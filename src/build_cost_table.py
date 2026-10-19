"""
Stage after extraction: compile records and run validation checks.
No API calls.

  python src/build_cost_table.py
"""

import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR     = Path("reports/raw")
DATA_DIR    = Path("data/HistoricalServices")
RECORDS_OUT = Path("reports/records.json")
CSV_OUT     = Path("reports/records.csv")

# Mirror of routing constant from extract_reports.py
PFDS_IDS = {"FSR-0014","FSR-0027","FSR-0028","FSR-0031","FSR-0037","FSR-0044"}

# Currency normalisation — only symbols/codes actually present in the data
CURRENCY_MAP = {
    "€":   "EUR",
    "$":   "USD?",   # bare $ with no country hint
    "USD": "USD", "EUR": "EUR", "CAD": "CAD",
    "SEK": "SEK", "AUD": "AUD", "ZAR": "ZAR",
}

# Country normalisation — only abbreviations actually present
COUNTRY_MAP = {
    "AU":  "Australia",
    "FIN": "Finland",
}


# ── Load ─────────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    records = []
    for path in sorted(RAW_DIR.glob("FSR-*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def normalize_records(records: list[dict]) -> None:
    """Add *_as_written copies then normalise currency and country in-place.
    Never changes any other field or guesses from country."""
    for r in records:
        r["currency_as_written"] = r.get("currency")
        r["currency"] = CURRENCY_MAP.get(r.get("currency"), r.get("currency"))

        r["country_as_written"] = r.get("country")
        r["country"] = COUNTRY_MAP.get(r.get("country"), r.get("country"))


# ── Site-based country inference ─────────────────────────────────────────────
def _place(site: str) -> str:
    """Normalise a site string to a matchable place name:
    strip (bracket text), take before first comma, strip trailing 2-3-letter
    abbreviation, lowercase."""
    s = re.sub(r'\s*\(.*?\)', '', site)       # remove (Swed), (AU), …
    s = s.split(',')[0]                        # take before first comma
    s = re.sub(r'\s+[A-Z]{2,3}$', '', s)     # strip trailing abbrev like "AU"
    return s.strip().lower()


def build_site_country_map(records: list[dict]) -> dict[str, str]:
    """Build {place_name -> country} from records that already have a country.
    Only keeps place names that map to exactly one country."""
    place_to_countries: dict[str, set] = defaultdict(set)
    for r in records:
        country = r.get("country")
        site    = r.get("site")
        if country and site:
            place_to_countries[_place(site)].add(country)

    site_map, warnings = {}, []
    for place, countries in sorted(place_to_countries.items()):
        if len(countries) == 1:
            site_map[place] = next(iter(countries))
        else:
            warnings.append(f"  WARNING: {place!r} maps to multiple countries: {countries}")

    print("\n--- SITE_COUNTRY_MAP (auto-built) ---")
    for place, country in sorted(site_map.items()):
        print(f"  {place!r} -> {country!r}")
    for w in warnings:
        print(w)

    return site_map


def infer_countries(records: list[dict], site_map: dict[str, str]) -> None:
    """Fill country from site_map where country is None. Add country_inferred flag."""
    for r in records:
        if r.get("country") is not None:
            r["country_inferred"] = False          # already had a country
            continue
        site = r.get("site")
        if site:
            p = _place(site)
            if p in site_map:
                r["country"]          = site_map[p]
                r["country_inferred"] = True
                continue
        r["country_inferred"] = False              # still None, not in map


# ── CSV helper ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "report_id","customer","site",
    "country","country_as_written","country_inferred",
    "visit_date","technician",
    "equipment_as_written","equipment_family",
    "onsite_hours","travel_hours",
    "currency","currency_as_written",
    "subtotal","vat_rate_percent","vat_amount","total_stated",
    "total_includes_vat","cost_status","_model",
    "parts_text","notes_text",
]

def parts_text(parts: list) -> str:
    bits = []
    for p in parts:
        qty  = f"{p['quantity']}x " if p.get("quantity") is not None else ""
        unit = f" {p['unit']}"       if p.get("unit")     else ""
        price = f" @{p['unit_price']}" if p.get("unit_price") is not None else ""
        bits.append(f"{qty}{p['name']}{unit}{price}")
    return "; ".join(bits)


# ── Checks ───────────────────────────────────────────────────────────────────
def ok(label: str, passed: bool, detail: str = ""):
    status = "OK  " if passed else "FAIL"
    print(f"  [{status}] {label}" + (f": {detail}" if detail else ""))


def run_checks(records: list[dict]):
    print("\n" + "="*60)
    print("CHECKS")
    print("="*60)

    # ── a) 50 records, unique ids ─────────────────────────────────
    print("\na) 50 records, unique report_ids")
    ids = [r["report_id"] for r in records]
    dupes = [x for x in set(ids) if ids.count(x) > 1]
    ok("50 records",       len(records) == 50, str(len(records)))
    ok("unique report_ids", not dupes,          str(dupes) if dupes else "")

    # ── b) cost_lines sum vs total_stated ─────────────────────────
    print("\nb) cost_lines sum vs total_stated")
    mismatches = []
    for r in records:
        lines  = r.get("cost_lines") or []
        stated = r.get("total_stated")
        if not lines or stated is None:
            continue
        totals = [l["total"] for l in lines if l.get("total") is not None]
        if not totals:
            continue
        s = round(sum(totals), 2)
        diff = abs(s - stated)
        vat  = r.get("vat_amount")
        # allow mismatch that equals the VAT amount
        if diff > 0.01 and (vat is None or abs(diff - vat) > 0.01):
            mismatches.append((r["report_id"], s, stated, vat))

    if not mismatches:
        ok("no unexplained line-sum vs total mismatches", True)
    else:
        ok("no unexplained line-sum vs total mismatches", False)
        for rid, s, stated, vat in mismatches:
            print(f"       {rid}: sum={s}  total_stated={stated}  vat_amount={vat}")

    # ── c) counts ─────────────────────────────────────────────────
    print("\nc) counts")
    for label, key in [
        ("cost_status",      "cost_status"),
        ("currency",         "currency"),
        ("country",          "country"),
        ("equipment_family", "equipment_family"),
        ("_model",           "_model"),
    ]:
        counts = Counter(str(r.get(key)) for r in records)
        print(f"  {label}:")
        for val, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {n:3d}  {val}")

    # ── d) Larox PF-DS records ────────────────────────────────────
    print("\nd) equipment_family == 'Larox PF-DS'")
    pfds_records = [r for r in records if r.get("equipment_family") == "Larox PF-DS"]
    print(f"  Found {len(pfds_records)} PF-DS records")
    for r in pfds_records:
        cloths = [p for p in (r.get("parts") or []) if "cloth" in p.get("name","").lower()]
        cloth_str = "; ".join(
            f"{p['name']} qty={p.get('quantity')}" for p in cloths
        ) or "(none)"
        print(f"  {r['report_id']}  country={r.get('country')}  currency={r.get('currency')}")
        print(f"    filter-cloth parts: {cloth_str}")
        print(f"    onsite={r.get('onsite_hours')}  travel={r.get('travel_hours')}"
              f"  total_stated={r.get('total_stated')}  cost_status={r.get('cost_status')}")

    # ── e) partial_lines or no_cost ───────────────────────────────
    print("\ne) cost_status partial_lines or no_cost")
    flagged = [r for r in records
               if r.get("cost_status") in ("partial_lines", "no_cost")]
    if flagged:
        for r in flagged:
            print(f"  {r['report_id']}: {r['cost_status']}")
    else:
        print("  (none)")

    # ── f) FSR-0027 specifics ─────────────────────────────────────
    print("\nf) FSR-0027")
    r27 = next((r for r in records if r["report_id"] == "FSR-0027"), None)
    if r27:
        cloth = next((p for p in (r27.get("parts") or [])
                      if "cloth" in p.get("name","").lower()), None)
        ok("cost_status partial_lines", r27.get("cost_status") == "partial_lines",
           str(r27.get("cost_status")))
        ok("total_stated null",         r27.get("total_stated") is None,
           str(r27.get("total_stated")))
        ok("currency EUR",              r27.get("currency") == "EUR",
           str(r27.get("currency")))
        if cloth:
            ok("Filter Cloth quantity 12",   cloth.get("quantity") == 12,
               str(cloth.get("quantity")))
            ok("Filter Cloth unit_price 98",
               cloth.get("unit_price") in (98, 98.0),
               str(cloth.get("unit_price")))
        else:
            ok("Filter Cloth found", False, "no part with 'cloth' in name")
    else:
        print("  FSR-0027 not found")

    # ── g) specific field values ───────────────────────────────────
    print("\ng) specific field values")
    checks_g = {
        "FSR-0014": [("total_stated", 4367.0), ("currency", "CAD")],
        "FSR-0031": [("total_stated", 2580.44), ("vat_rate_percent", 24),
                     ("total_includes_vat", True)],
        "FSR-0044": [("total_stated", 19850),  ("currency", "SEK")],
    }
    by_id = {r["report_id"]: r for r in records}
    for fid, expectations in checks_g.items():
        r = by_id.get(fid)
        if not r:
            print(f"  {fid}: not found"); continue
        for field, expected in expectations:
            actual = r.get(field)
            # compare numerically for floats
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                passed = abs(float(actual) - float(expected)) < 0.001
            else:
                passed = actual == expected
            ok(f"{fid} {field}=={expected}", passed, f"actual={actual!r}")

    # ── h) routing vs equipment_family alignment ──────────────────
    print("\nh) routing vs equipment_family alignment")
    wrong_family = [r["report_id"] for r in records
                    if r["report_id"] not in PFDS_IDS
                    and r.get("equipment_family") == "Larox PF-DS"]
    wrong_routing = [r["report_id"] for r in records
                     if r["report_id"] in PFDS_IDS
                     and r.get("equipment_family") != "Larox PF-DS"]
    ok("non-PFDS IDs with family Larox PF-DS", not wrong_family,
       str(wrong_family) if wrong_family else "")
    ok("PFDS IDs without family Larox PF-DS",  not wrong_routing,
       str(wrong_routing) if wrong_routing else "")

    # ── i) SITE_COUNTRY_MAP fill results ─────────────────────────
    print("\ni) site-inferred country fill")
    TARGET = {
        "FSR-0018": "Australia",
        "FSR-0029": "Sweden",
        "FSR-0036": "Peru",
        "FSR-0038": "South Africa",
    }
    by_id = {r["report_id"]: r for r in records}

    filled   = [r for r in records if r.get("country_inferred")]
    still_null = [r for r in records if r.get("country") is None]

    ok("exactly 4 records filled", len(filled) == 4,
       f"got {[r['report_id'] for r in filled]}")
    ok("no previously-known country changed",
       all(not r.get("country_inferred") for r in records
           if r.get("country_as_written") is not None), "")

    for fid, expected in TARGET.items():
        r = by_id.get(fid)
        if r:
            actual = r.get("country")
            if actual is None:
                ok(f"{fid} -> {expected}", False, "place not in auto-built map — add manually if needed")
            else:
                ok(f"{fid} -> {expected}", actual == expected, f"actual={actual!r}")
        else:
            ok(f"{fid} found", False, "record missing")

    print(f"\n  Filled ({len(filled)}):")
    for r in filled:
        print(f"    {r['report_id']}  site={r.get('site')!r}"
              f"  country_as_written={r.get('country_as_written')!r}"
              f"  -> country={r.get('country')!r}")
    if still_null:
        print(f"\n  Still country=None ({len(still_null)}):")
        for r in still_null:
            print(f"    {r['report_id']}  site={r.get('site')!r}")

    print("\n  Country counts after fill:")
    for val, n in sorted(Counter(str(r.get("country")) for r in records).items(),
                         key=lambda x: -x[1]):
        print(f"    {n:3d}  {val}")


# ── Informational prints (not checks) ────────────────────────────────────────
def print_none_countries(records: list[dict]) -> None:
    """Print id, site, and first 3 source lines for records with country=None."""
    nulls = [r for r in records if r.get("country") is None]
    print(f"\n--- Records with country=None ({len(nulls)}) ---")
    for r in nulls:
        src_lines = (DATA_DIR / f"{r['report_id']}.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        print(f"  {r['report_id']}  site={r.get('site')!r}")
        for line in src_lines[:3]:
            print(f"    {line!r}")


def print_other_equipment(records: list[dict]) -> None:
    """Print id, equipment_as_written, country, cost_status for 'other' family."""
    others = [r for r in records if r.get("equipment_family") == "other"]
    print(f"\n--- equipment_family == 'other' ({len(others)}) ---")
    for r in others:
        print(f"  {r['report_id']}  written={r.get('equipment_as_written')!r}"
              f"  country={r.get('country')}  status={r.get('cost_status')}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    records = load_records()
    print(f"Loaded {len(records)} records from {RAW_DIR}")
    normalize_records(records)  # adds *_as_written; normalises currency/country
    site_map = build_site_country_map(records)
    infer_countries(records, site_map)  # fills country=None from site; adds country_inferred

    # Write records.json
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {RECORDS_OUT}")

    # Write records.csv (parts joined as text)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in CSV_FIELDS}
            row["parts_text"] = parts_text(r.get("parts") or [])
            row["notes_text"] = "; ".join(r.get("notes") or [])
            w.writerow(row)
    print(f"Wrote {CSV_OUT}")

    print_other_equipment(records)

    run_checks(records)


if __name__ == "__main__":
    main()
