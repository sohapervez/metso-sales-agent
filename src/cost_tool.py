"""
src/cost_tool.py — field-service cost estimation (no API, plain code).
Reads reports/records.json.

estimate_service_cost(equipment, job_or_part="", quantity=None, country=None) -> dict
"""

import json, re, statistics
from pathlib import Path

RECORDS_PATH = Path("reports/records.json")

# Exact alias matching only — no substring fallback, so 'larox pf' cannot
# accidentally hit 'Larox PF-DS' aliases.
EQUIPMENT_ALIASES: dict[str, list[str]] = {
    "Larox PF-DS":      ["pf-ds", "pf ds", "pfds", "tower press", "larox pf-ds"],
    "Larox PF":         ["larox pf", "pf 1.6", "pf 12", "pf 15", "pf 60"],
    "ColumnCell":       ["columncell", "column cell"],
    "Concorde Cell":    ["concorde cell", "concorde"],
    "CellStation":      ["cellstation", "cell station"],
    "Filter Optimizer": ["filter optimizer"],
    "MD series pump":   ["md series", "md series pump", "mdm", "mdr"],
    "Orion pump":       ["orion pump", "orion"],
}

_RECORDS: list[dict] | None = None

# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return re.sub(r'[^a-z0-9]', '', s.lower())

# Filler words stripped from job queries before matching
_FILLER = frozenset({
    "replace", "replacement", "replaced", "change", "changing",
    "service", "visit", "job", "of", "the", "a", "for", "on",
})
# Simple English plurals that appear in the data
_PLURALS = {"cloths": "cloth", "seals": "seal"}

def _job_words(job: str) -> list[str]:
    """Normalised keyword list: split, singularise, drop fillers."""
    words = [_PLURALS.get(w, w) for w in job.lower().split()]
    return [w for w in words if w not in _FILLER]


def _load() -> list[dict]:
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    return _RECORDS


def _match_family(query: str) -> str | None:
    """Return the known equipment_family for query, or None (→ search 'other' records).

    Matching order (most specific first):
    1. Explicit normalised patterns (pfds, towerpress, laroxpf).
    2. Normalised family name appears as substring of normalised input.
    3. Exact alias lookup (original behaviour).
    """
    norm = _normalise(query)

    # Explicit patterns — most specific first to avoid laroxpf matching laroxpfds
    if "pfds" in norm or "towerpress" in norm:
        return "Larox PF-DS"
    if "laroxpf" in norm:
        return "Larox PF"

    # Family name as substring of normalised input (longer names checked first)
    for fam in EQUIPMENT_ALIASES:
        if _normalise(fam) in norm:
            return fam

    # Exact alias fallback
    q = query.lower().strip()
    for fam, aliases in EQUIPMENT_ALIASES.items():
        if q == fam.lower() or q in aliases:
            return fam

    return None


def _eq_match(r: dict, query: str, family: str | None) -> bool:
    """True when a record matches the equipment query."""
    if family:
        return r.get("equipment_family") == family
    # For 'other' records: any significant token from the query appears in
    # equipment_as_written (len >= 3 to skip short stop-words).
    ew     = (r.get("equipment_as_written") or "").lower()
    tokens = [t for t in query.lower().replace("-", " ").split() if len(t) >= 3]
    return r.get("equipment_family") == "other" and bool(tokens) and any(t in ew for t in tokens)


def _word_match(word: str, blob: str) -> bool:
    """True when word appears as a whole word (or simple plural word+'s') in blob."""
    return bool(re.search(r'\b' + re.escape(word) + r's?\b', blob))

def _job_match(r: dict, job: str) -> bool:
    """True when ALL keywords from job (after dropping fillers, singularising) appear
    as whole words across the record's parts names, job_summary, or notes."""
    words = _job_words(job)
    if not words:
        return True  # empty query matches everything

    parts_text = " ".join((p.get("name") or "").lower() for p in (r.get("parts") or []))
    summary    = (r.get("job_summary") or "").lower()
    notes_text = " ".join((n or "").lower() for n in (r.get("notes") or []))
    blob       = f"{parts_text} {summary} {notes_text}"

    return all(_word_match(w, blob) for w in words)


def _parts_for_job(r: dict, job: str) -> list[dict]:
    """Parts whose name contains ALL job keywords as whole words; return all if job empty."""
    if not job:
        return r.get("parts") or []
    words = _job_words(job)
    if not words:
        return r.get("parts") or []
    return [p for p in (r.get("parts") or [])
            if all(_word_match(w, (p.get("name") or "").lower()) for w in words)]


def _summarise(r: dict, job: str) -> dict:
    return {
        "id":                 r["report_id"],
        "country":            r.get("country"),
        "country_as_written": r.get("country_as_written"),
        "country_inferred":   r.get("country_inferred"),
        "currency":           r.get("currency"),
        "onsite_hours":       r.get("onsite_hours"),
        "travel_hours":       r.get("travel_hours"),
        "matching_parts":     _parts_for_job(r, job),
        "total_stated":       r.get("total_stated"),
        "total_includes_vat": r.get("total_includes_vat"),
        "cost_status":        r.get("cost_status"),
        "notes":              r.get("notes") or [],
    }


def _build_estimate(
    usable: list[dict], partial: list[dict]
) -> tuple[dict, list[dict]]:
    """
    Returns:
      estimate       — {currency_label: {value} or {min, median, max, n}}
                       from full_breakdown / total_only records with non-null currency.
      part_estimates — list of partial sums from partial_lines records.
    """
    by_cur: dict[str, list[float]] = {}
    for r in usable:
        cur, tot = r.get("currency"), r.get("total_stated")
        if cur is not None and tot is not None:
            by_cur.setdefault(cur, []).append(float(tot))

    estimate: dict = {}
    for cur, vals in by_cur.items():
        label = "currency unclear: $" if cur == "USD?" else cur
        if len(vals) == 1:
            estimate[label] = {"value": vals[0]}
        else:
            estimate[label] = {
                "min":    min(vals),
                "median": statistics.median(vals),
                "max":    max(vals),
                "n":      len(vals),
            }

    part_estimates: list[dict] = []
    for r in partial:
        lines = [l for l in (r.get("cost_lines") or []) if l.get("total") is not None]
        if lines:
            part_estimates.append({
                "id":                  r["report_id"],
                "currency":            r.get("currency"),
                "sum_of_stated_lines": round(sum(l["total"] for l in lines), 2),
                "lines_summed":        [f"{l['description']}: {l['total']}" for l in lines],
                "label":               "partial",
            })

    return estimate, part_estimates


def estimate_service_cost(
    equipment:  str,
    job_or_part: str       = "",
    quantity:    int | None = None,
    country:     str | None = None,
) -> dict:

    records = _load()

    # 1. Match equipment family (or fall through to 'other' by name)
    family  = _match_family(equipment)
    matched = [r for r in records if _eq_match(r, equipment, family)]

    # 2. Split by job/part relevance
    if job_or_part:
        job_matched = [r for r in matched if _job_match(r, job_or_part)]
        other_jobs  = [r for r in matched if not _job_match(r, job_or_part)]
    else:
        job_matched, other_jobs = matched, []

    # 5. Categorise by cost_status and currency
    usable = [
        r for r in job_matched
        if r.get("cost_status") in ("full_breakdown", "total_only")
        and r.get("currency") is not None
        and r.get("total_stated") is not None
    ]
    partial = [r for r in job_matched if r.get("cost_status") == "partial_lines"]
    excluded = [
        {"id": r["report_id"],
         "reason": r.get("cost_status") or "no usable cost",
         "currency": r.get("currency")}
        for r in job_matched
        if r not in usable and r not in partial
    ]

    estimate, partial_estimates = _build_estimate(usable, partial)

    # 6. Country matching
    country_info: dict = {}
    if country:
        c_lo      = country.lower()
        from_here = [r for r in usable if (r.get("country") or "").lower() == c_lo]
        country_info["country_match"] = bool(from_here)
        if not from_here:
            country_info["countries_used"] = sorted(
                {r["country"] for r in usable if r.get("country")}
            )

    # 7. Quantity comparison (requested vs. each matching part's quantity)
    q_cmp: list[dict] = []
    if quantity is not None:
        for r in job_matched:
            for p in _parts_for_job(r, job_or_part):
                if p.get("quantity") is not None:
                    q_cmp.append({
                        "id":           r["report_id"],
                        "part":         p["name"],
                        "part_quantity": p["quantity"],
                        "requested":    quantity,
                    })

    # 10. Closest match: record with most similar part quantity among records
    #     that have at least partial cost lines
    closest: str | None = None
    if quantity is not None:
        pool = [r for r in job_matched
                if r.get("cost_status") in ("full_breakdown", "total_only", "partial_lines")]
        best_diff = float("inf")
        for r in pool:
            for p in _parts_for_job(r, job_or_part):
                q = p.get("quantity")
                if q is not None:
                    d = abs(float(q) - float(quantity))
                    if d < best_diff:
                        best_diff, closest = d, r["report_id"]

    # 9. Missing inputs
    missing = (["equipment"]   if not equipment   else []) + \
              (["job_or_part"] if not job_or_part else []) + \
              (["country"]     if country is None  else [])

    result = {
        "equipment_family":          family or f"other (matched by name: {equipment!r})",
        "records_considered":        len(matched),
        "records_used":              len(usable) + len(partial),
        "missing_inputs":            missing,
        "matched_records":           [_summarise(r, job_or_part) for r in job_matched],
        "excluded":                  excluded,
        "other_jobs_same_equipment": [_summarise(r, "") for r in other_jobs],
        "estimate":                  estimate,
        "partial_estimates":         partial_estimates,
        "closest_match":             closest,
    }
    if country_info:
        result.update(country_info)
    if q_cmp:
        result["quantity_comparison"] = q_cmp

    if not job_or_part:
        ask = ["What job or part is needed? (e.g. filter cloth, sparger seal, commissioning)"]
        if country is None:
            ask.append("Which country is this site in?")
        result["needs_clarification"] = True
        result["ask_user"]            = ask
        result["note"]                = "Do not quote a figure until the job is known"

    return result


# ── CLI tests ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TESTS = [
        ("PF-DS",         "filter cloth",  12, "Chile"),       # a
        ("ColumnCell",    "sparger seals",  4, "Chile"),       # b
        ("SALA VSP-750",  "",            None,  None),         # c
        ("Larox PF",      "filter cloth", None, "Australia"),  # d
        ("Concorde Cell", "",            None,  None),         # e
    ]
    for eq, job, qty, cnt in TESTS:
        print(f"\n{'='*60}")
        print(f"estimate_service_cost({eq!r}, {job!r}, qty={qty}, country={cnt!r})")
        print(json.dumps(estimate_service_cost(eq, job, qty, cnt),
                         indent=2, ensure_ascii=False))
