"""
FSR extraction via Gemini.

  python src/extract_reports.py --dry-run
  python src/extract_reports.py --ids FSR-0001 FSR-0027 FSR-0028 FSR-0044
  python src/extract_reports.py --all
"""

import argparse, json, os, time
from pathlib import Path

# ── Model routing ─────────────────────────────────────────────────────────────
PFDS_IDS   = ["FSR-0014","FSR-0027","FSR-0028","FSR-0031","FSR-0037","FSR-0044"]
PFDS_CHAIN = ["gemini-3.7-flash", "gemini-3.6-flash"]
LITE_CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

FLASH_WAIT = 15   # seconds between Flash-chain calls (≤ 5 rpm)
LITE_WAIT  = 5    # seconds between Lite-chain calls  (≤ 15 rpm)

DATA_DIR = Path("data/HistoricalServices")
RAW_DIR  = Path("reports/raw")

# ── Extraction prompt (verbatim per spec) ─────────────────────────────────────
PROMPT = """\
You extract data from one field service report. Output only JSON that follows the schema.
Rules:
- Copy values exactly as written. Never calculate, convert, round, infer or fill in anything.
- If a field is not written in the report, use null (or [] for lists).
- total_stated is only a total the report prints. Do not add lines up yourself.
- cost_status: 'full_breakdown' = itemised lines AND a printed total; 'total_only' = just one
  total; 'partial_lines' = itemised lines but the total or some lines are missing or cut off;
  'no_cost' = no money amounts (e.g. 'invoice to follow', 'as per agreement').
- Do not convert currencies and do not guess a currency from the country.
- Parts: list every part or material with a quantity if given. Quantities such as '12 of 72 cloths replaced'
  mean quantity 12 and put '12 of 72' in notes. Plates are not cloths: keep the name as written.
- Add to notes, as short factual strings: 'invoice to follow', 'cost summary cut off', 'partial job',
  'VAT included', 'approximate hours (about)', 'quantity not stated', 'travel hours equal labour hours',
  and any number that looks inconsistent inside the report. Do not add opinions.
- Do not mix reports. Do not use outside knowledge.

Schema (every field null unless written):
{
  "report_id": str,
  "customer": str|null,
  "site": str|null,
  "country": str|null,
  "visit_date": str|null,
  "technician": str|null,
  "equipment_as_written": str|null,
  "equipment_family": "Larox PF-DS"|"Larox PF"|"ColumnCell"|"Concorde Cell"|"CellStation"|
                      "Filter Optimizer"|"MD series pump"|"Orion pump"|"other"|null,
  "job_summary": str|null,
  "tasks": [{"name": str, "hours": number|null}],
  "onsite_hours": number|null,
  "travel_hours": number|null,
  "parts": [{"name": str, "quantity": number|null, "unit": str|null,
             "unit_price": number|null, "line_total": number|null,
             "source": "parts_table"|"cost_table"|"text"}],
  "currency": str|null,
  "cost_lines": [{"description": str, "quantity": number|null,
                  "unit_rate": number|null, "total": number|null}],
  "subtotal": number|null,
  "vat_amount": number|null,
  "vat_rate_percent": number|null,
  "total_stated": number|null,
  "total_includes_vat": true|false|null,
  "cost_status": "full_breakdown"|"total_only"|"partial_lines"|"no_cost",
  "notes": [str]
}

Report:
"""


# ── Stage 1: dry-run analysis (no API) ────────────────────────────────────────
def dry_run():
    files = sorted(DATA_DIR.glob("FSR-*.txt"))
    print(f"Found {len(files)} reports in {DATA_DIR}\n")

    CUTOFF_CHARS = set("|$€£¥₹₩₽฿")
    sizes, no_newline, cutoffs = [], [], []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        sizes.append((len(text), f.stem))
        if not text.endswith("\n"):
            no_newline.append(f.stem)
        last_line = text.rstrip("\n").rsplit("\n", 1)[-1].rstrip()
        if last_line and last_line[-1] in CUTOFF_CHARS:
            cutoffs.append(f.stem)

    sizes.sort()
    print("5 shortest (chars):")
    for n, s in sizes[:5]:
        print(f"  {n:6d}  {s}")
    print("\n5 longest (chars):")
    for n, s in sizes[-5:]:
        print(f"  {n:6d}  {s}")

    print(f"\nNo trailing newline ({len(no_newline)}): {no_newline or 'none'}")
    print(f"Possibly cut off   ({len(cutoffs)}): {cutoffs or 'none'}")

    print("\nModel chain per report:")
    for _, stem in sizes:
        chain = PFDS_CHAIN if stem in PFDS_IDS else LITE_CHAIN
        print(f"  {stem}: {' → '.join(chain)}")


# ── Single-model API call with retry ──────────────────────────────────────────
def _call(client, model: str, content: str) -> str:
    """Call one model; retry 3× on 429/503; re-raise on daily quota."""
    from google.genai import types
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return r.text
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err:
                raise  # daily quota; let caller handle
            if any(x in err for x in ("429", "503", "SERVICE_UNAVAILABLE")):
                if attempt < 2:
                    print(f"    [{model} attempt {attempt+1}] rate-limit, wait 20 s")
                    time.sleep(20)
                    continue
            raise
    raise RuntimeError(f"{model}: all retries exhausted")


# ── Extract one report (try chain in order) ───────────────────────────────────
def _extract(client, fsr_id: str, report_text: str, exhausted: set) -> dict | None:
    chain = PFDS_CHAIN if fsr_id in PFDS_IDS else LITE_CHAIN
    content = PROMPT + report_text  # model reads the ID from the report header
    raw = None

    for model in [m for m in chain if m not in exhausted]:
        try:
            raw = _call(client, model, content)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"  daily quota hit: {model}")
                exhausted.add(model)
            else:
                print(f"  {fsr_id}/{model}: {str(e)[:100]}")
            continue

        # Try to parse JSON; retry once on bad JSON
        try:
            rec = json.loads(raw)
            rec["_model"] = model
            return rec
        except json.JSONDecodeError:
            try:
                raw = _call(client, model, content)  # one retry
                rec = json.loads(raw)
                rec["_model"] = model
                return rec
            except Exception:
                pass

        # Both attempts yielded invalid JSON; save error and give up
        err_path = RAW_DIR / f"{fsr_id}.error.txt"
        err_path.write_text(raw or "", encoding="utf-8")
        print(f"  {fsr_id}: invalid JSON, saved to {err_path.name}, moving on")
        return None

    print(f"  {fsr_id}: chain exhausted/failed, skipping")
    return None


# ── Process a list of report IDs ──────────────────────────────────────────────
def process(fsr_ids: list[str], verbose: bool = False):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running")

    from google import genai
    client = genai.Client(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    exhausted: set[str] = set()
    skipped:   list[str] = []
    # Track last call time per chain to enforce per-chain rate limits
    last_call = {"pfds": 0.0, "lite": 0.0}

    for fsr_id in fsr_ids:
        cache = RAW_DIR / f"{fsr_id}.json"
        if cache.exists():
            print(f"  {fsr_id}: cached")
            if verbose:
                print(json.dumps(json.loads(cache.read_text("utf-8")), indent=2, ensure_ascii=False))
            continue

        txt = DATA_DIR / f"{fsr_id}.txt"
        if not txt.exists():
            print(f"  {fsr_id}: source not found, skip")
            skipped.append(fsr_id)
            continue

        is_pfds = fsr_id in PFDS_IDS
        chain    = PFDS_CHAIN if is_pfds else LITE_CHAIN
        key      = "pfds" if is_pfds else "lite"
        wait     = FLASH_WAIT if is_pfds else LITE_WAIT

        if all(m in exhausted for m in chain):
            print(f"  {fsr_id}: chain fully exhausted, skip")
            skipped.append(fsr_id)
            continue

        # Rate-limit: wait if needed since last call on this chain
        elapsed = time.monotonic() - last_call[key]
        if elapsed < wait:
            time.sleep(wait - elapsed)

        print(f"  {fsr_id} ...", end=" ", flush=True)
        last_call[key] = time.monotonic()

        rec = _extract(client, fsr_id, txt.read_text(encoding="utf-8", errors="replace"), exhausted)
        if rec is None:
            skipped.append(fsr_id)
            continue

        cache.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved ({rec['_model']})")
        if verbose:
            print(json.dumps(rec, indent=2, ensure_ascii=False))

    # If a chain is fully exhausted, list what remains
    for name, chain in [("PFDS", PFDS_CHAIN), ("LITE", LITE_CHAIN)]:
        if all(m in exhausted for m in chain):
            remaining = [f.stem for f in sorted(DATA_DIR.glob("FSR-*.txt"))
                         if not (RAW_DIR / f"{f.stem}.json").exists()]
            if remaining:
                print(f"\n{name} chain exhausted. {len(remaining)} report(s) left: {remaining}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {skipped}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Stage 1: analysis, no API")
    g.add_argument("--ids", nargs="+", metavar="FSR-XXXX", help="Stage 2: specific reports")
    g.add_argument("--all", action="store_true", dest="run_all", help="Stage 3: all remaining")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
    elif args.ids:
        process(args.ids, verbose=True)
    else:
        all_ids = sorted(f.stem for f in sorted(DATA_DIR.glob("FSR-*.txt")))
        process(all_ids, verbose=False)


if __name__ == "__main__":
    main()
