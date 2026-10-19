"""
Fix known transcription errors in cached Gemini output.
No API calls. Backs up tables/raw/ first, patches fields, then rebuilds .md files.
"""

import re, shutil, sys
from pathlib import Path

RAW_DIR    = Path("tables/raw")
BACKUP_DIR = Path("tables/raw_backup")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def change_field(line: str, field: str, new_val: str) -> tuple[str, bool]:
    """
    Replace 'field = <old>' with 'field = new_val' in a semicolon-delimited line.
    [^;\\n]* stops before the next semicolon or newline so we never eat the delimiter.
    Returns (new_line, was_changed). Raises if the field name is not found.
    """
    pat = re.compile(rf'({re.escape(field)} = )([^;\n]*)')
    m = pat.search(line)
    if not m:
        raise ValueError(f"Field not found: {field!r}\n  in: {line[:80]!r}")
    old_val = m.group(2)
    if old_val == new_val:
        return line, False
    return line[:m.start(2)] + new_val + line[m.end(2):], True


def patch(path: Path, prefix: str, field_fixes: dict[str, str]) -> None:
    """
    Find every line starting with prefix, apply field_fixes to it, write back.
    Prints each change as:  [file] [prefix]  field -> 'new_val'.
    Raises if no matching line is found, or if any field is missing.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            found = True
            for field, new_val in field_fixes.items():
                line, changed = change_field(line, field, new_val)
                if changed:
                    print(f"  {path.name}  [{prefix}]  {field} -> {new_val!r}")
            lines[i] = line
    if not found:
        raise ValueError(f"Prefix not found: {prefix!r} in {path}")
    path.write_text("".join(lines), encoding="utf-8")


def remove_token(path: Path, token: str) -> None:
    """Remove every occurrence of token from the file. Raises if absent."""
    text = path.read_text(encoding="utf-8")
    count = text.count(token)
    if count == 0:
        raise ValueError(f"Token not found: {token!r} in {path}")
    path.write_text(text.replace(token, ""), encoding="utf-8")
    print(f"  {path.name}  Removed {count}× {token!r}")


def delete_block(path: Path, start_prefix: str, end_prefix: str) -> None:
    """
    Delete lines from the line that starts with start_prefix up to
    (but NOT including) the line that starts with end_prefix.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    si = ei = None
    for i, line in enumerate(lines):
        if si is None and line.startswith(start_prefix):
            si = i
        elif si is not None and line.startswith(end_prefix):
            ei = i
            break
    if si is None:
        raise ValueError(f"Block start not found: {start_prefix!r}")
    if ei is None:
        raise ValueError(f"Block end not found: {end_prefix!r}")
    path.write_text("".join(lines[:si] + lines[ei:]), encoding="utf-8")
    print(f"  {path.name}  Deleted lines {si+1}–{ei} (block: {start_prefix!r}…)")


# ---------------------------------------------------------------------------
# Rebuild combined .md files from cached raw text
# ---------------------------------------------------------------------------

def rebuild_mds() -> None:
    """Re-create tables/<brochure>.md from cached raw files (no API calls)."""
    sys.path.insert(0, "src")
    from read_tables import PAGES, write_brochure_md  # main() is guarded
    print("\n=== Rebuilding .md files ===")
    for brochure, page_list in PAGES.items():
        pages = {}
        for n in page_list:
            raw = RAW_DIR / f"{brochure}-p{n}.txt"
            if raw.exists():
                pages[n] = raw.read_text(encoding="utf-8")
        if pages:
            write_brochure_md(brochure, pages)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(ok: bool, label: str) -> None:
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {label}")


def run_checks() -> None:
    print("\n=== Checks ===")
    p6  = (RAW_DIR / "larox-pf-brochure-b-p6.txt").read_text()
    p5  = (RAW_DIR / "larox-pf-ds-p5.txt").read_text()
    p10 = (RAW_DIR / "orion-series-pump-p10.txt").read_text()

    BAD = ("Required floor area", "Hydraulic unit motor", "Hydraulic unit oil tank")

    # (a) No "not available" in the three key fields on any PF 12 / PF 15 line
    bad = [l for l in p6.splitlines()
           if ("LAROX PF 12 SERIES" in l or "LAROX PF 15 SERIES" in l)
           and any(f"{f} = not available" in l for f in BAD)]
    check(not bad, "p6: no 'not available' in floor area / hydraulic motor / oil tank")

    # (b) PF 12 model 25 and PF 15 model 40 → 18.5 kW / 3000 l
    l25 = next((l for l in p6.splitlines() if l.startswith("LAROX PF 12 SERIES 25:")), "")
    check("Water pump motor = 18.5 (3000) kW (rpm)" in l25
          and "Pressing water tank = 3000 l" in l25,
          "p6: PF 12 model 25 → 18.5 kW, 3000 l")

    l40 = next((l for l in p6.splitlines() if l.startswith("LAROX PF 15 SERIES 40:")), "")
    check("Water pump motor = 18.5 (3000) kW (rpm)" in l40
          and "Pressing water tank = 3000 l" in l40,
          "p6: PF 15 model 40 → 18.5 kW, 3000 l")

    # (c) p5: Chambers 16 fixed; "Electric motors" gone from entire file
    l16 = next((l for l in p5.splitlines() if l.startswith("Chambers 16:")), "")
    check("Pressing water pump IEC = 18.5 kW" in l16,
          "p5: Chambers 16 Pressing water pump IEC = 18.5 kW")
    check("Pressing water tank = 3.0 m³" in l16,
          "p5: Chambers 16 Pressing water tank = 3.0 m³")
    check("Electric motors" not in p5,
          "p5: 'Electric motors' removed from all lines")

    # (d) Chart block gone from p10
    check("[Chart labels" not in p10, "p10: chart block removed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 0: back up before touching anything
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(RAW_DIR, BACKUP_DIR)
    print(f"Backed up {RAW_DIR} → {BACKUP_DIR}\n")

    # ---- Fix 1: larox-pf-brochure-b-p6.txt ---------------------------------
    print("=== Fix 1: p6 — PF 12 ===")
    p6 = RAW_DIR / "larox-pf-brochure-b-p6.txt"

    # Models 9.5–25: floor area, hydraulic motor, oil tank
    for m in ("9.5", "12.5", "16", "19", "22", "25"):
        patch(p6, f"LAROX PF 12 SERIES {m}:", {
            "Required floor area":     "44.5 m²",
            "Hydraulic unit motor":    "22 (1500) kW (rpm)",
            "Hydraulic unit oil tank": "300 l",
        })

    # Models 9.5–22: water pump and pressing tank
    for m in ("9.5", "12.5", "16", "19", "22"):
        patch(p6, f"LAROX PF 12 SERIES {m}:", {
            "Water pump motor":    "11 (3000) kW (rpm)",
            "Pressing water tank": "1500 l",
        })

    # Models 25–38: water pump and pressing tank
    for m in ("25", "28", "32", "35", "38"):
        patch(p6, f"LAROX PF 12 SERIES {m}:", {
            "Water pump motor":    "18.5 (3000) kW (rpm)",
            "Pressing water tank": "3000 l",
        })

    print("\n=== Fix 1: p6 — PF 15 ===")

    # Models 25–35: floor area, hydraulic motor, oil tank
    for m in ("25", "30", "35"):
        patch(p6, f"LAROX PF 15 SERIES {m}:", {
            "Required floor area":     "84 m²",
            "Hydraulic unit motor":    "30 (1500) kW (rpm)",
            "Hydraulic unit oil tank": "300 l",
        })

    # Models 25–40: water pump and pressing tank
    for m in ("25", "30", "35", "40"):
        patch(p6, f"LAROX PF 15 SERIES {m}:", {
            "Water pump motor":    "18.5 (3000) kW (rpm)",
            "Pressing water tank": "3000 l",
        })

    # Models 45–60: water pump and pressing tank (verify already-correct values)
    for m in ("45", "50", "55", "60"):
        patch(p6, f"LAROX PF 15 SERIES {m}:", {
            "Water pump motor":    "37 (3000) kW (rpm)",
            "Pressing water tank": "6000 l",
        })

    # ---- Fix 2: larox-pf-brochure-b-p8.txt ---------------------------------
    print("\n=== Fix 2: p8 — sizes 72–156 ===")
    p8 = RAW_DIR / "larox-pf-brochure-b-p8.txt"
    for sz in ("72", "84", "96", "108", "120", "132", "144", "156"):
        patch(p8, f"{sz}:", {
            "Filter length":      "not listed per size (brochure prints 8,136 mm once, under size 60)",
            "Filter width":       "not listed per size (brochure prints 5,144 mm once, under size 60)",
            "Required floor area":"not listed per size (brochure prints 175.5 m² once, under size 60)",
        })

    # ---- Fix 3: larox-pf-ds-p5.txt -----------------------------------------
    print("\n=== Fix 3: p5 — Chambers 16 and remove Electric motors ===")
    p5 = RAW_DIR / "larox-pf-ds-p5.txt"
    patch(p5, "Chambers 16:", {
        "Pressing water pump IEC":  "18.5 kW, 3000 rpm, 50 Hz",
        "Pressing water pump NEMA": "25 hp, 3600 rpm, 60 Hz",
        "Pressing water tank":      "3.0 m³",
    })
    # Remove from all lines (Chambers 18's 0.13 m³ is not touched — different field)
    remove_token(p5, "Electric motors = not available; ")

    # ---- Fix 4: orion-series-pump-p10.txt -----------------------------------
    print("\n=== Fix 4: p10 — delete chart block ===")
    p10 = RAW_DIR / "orion-series-pump-p10.txt"
    delete_block(p10, "[Chart labels and axes]", "Connection dimensions /")

    # Rebuild and verify
    rebuild_mds()
    run_checks()


if __name__ == "__main__":
    main()
