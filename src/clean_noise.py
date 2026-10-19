"""
Clean noise lines (footer tags, image labels, chart axes) from cached Gemini output.
No API calls. Never deletes table-row lines (lines containing ' = ').
"""

import shutil, sys
from pathlib import Path

RAW_DIR    = Path("tables/raw")
BACKUP_DIR = Path("tables/raw_backup2")


# ---------------------------------------------------------------------------
# Safety guard and helpers
# ---------------------------------------------------------------------------

def has_data(line: str) -> bool:
    """A table-row line contains ' = '. These must never be deleted."""
    return " = " in line


def count_data_lines(path: Path) -> int:
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if has_data(l))


def _safe(removed: list[str], path: Path) -> None:
    """Raise if any line we are about to delete contains ' = '."""
    for l in removed:
        if has_data(l):
            raise ValueError(f"[ABORT] Would delete data line in {path.name}: {l!r}")


def remove_from_to_eof(path: Path, start_line: str) -> None:
    """Delete from start_line (exact) through end of file."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    si = next((i for i, l in enumerate(lines) if l.rstrip("\n") == start_line), None)
    if si is None:
        print(f"  [WARN] {path.name}: marker not found: {start_line!r} — skipping")
        return
    removed = lines[si:]
    _safe(removed, path)
    print(f"  {path.name}: removing lines {si+1}–{len(lines)}:")
    for l in removed:
        print(f"    - {l.rstrip()!r}")
    path.write_text("".join(lines[:si]), encoding="utf-8")


def remove_block(path: Path, start_line: str, end_line: str) -> None:
    """Delete from start_line to end_line (both inclusive, both exact match)."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    si = next((i for i, l in enumerate(lines) if l.rstrip("\n") == start_line), None)
    if si is None:
        print(f"  [WARN] {path.name}: start not found: {start_line!r} — skipping")
        return
    ei = next((i for i, l in enumerate(lines) if i >= si and l.rstrip("\n") == end_line), None)
    if ei is None:
        print(f"  [WARN] {path.name}: end not found: {end_line!r} — skipping")
        return
    removed = lines[si:ei + 1]
    _safe(removed, path)
    print(f"  {path.name}: removing lines {si+1}–{ei+1}:")
    for l in removed:
        print(f"    - {l.rstrip()!r}")
    path.write_text("".join(lines[:si] + lines[ei + 1:]), encoding="utf-8")


def remove_second_occurrence(path: Path, target: str) -> None:
    """Remove the second (and later) occurrence of a line matching target exactly."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen, new_lines = False, []
    removed_count = 0
    for l in lines:
        if l.rstrip("\n") == target:
            if not seen:
                seen = True
                new_lines.append(l)
            else:
                _safe([l], path)
                print(f"  {path.name}: removing duplicate: {l.rstrip()!r}")
                removed_count += 1
        else:
            new_lines.append(l)
    if not seen:
        print(f"  [WARN] {path.name}: line not found: {target!r}")
    elif removed_count == 0:
        print(f"  [WARN] {path.name}: no duplicate found for: {target!r}")
    path.write_text("".join(new_lines), encoding="utf-8")


def remove_block_keep_subblock(path: Path, start_line: str, end_line: str,
                                keep_start: str, keep_end: str) -> None:
    """
    Delete lines from start_line to end_line (inclusive) EXCEPT the contiguous
    sub-block from keep_start to keep_end (inclusive).
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    si  = next((i for i, l in enumerate(lines) if l.rstrip("\n") == start_line), None)
    ei  = next((i for i, l in enumerate(lines) if l.rstrip("\n") == end_line),   None)
    ksi = next((i for i, l in enumerate(lines) if l.rstrip("\n") == keep_start), None)
    kei = next((i for i, l in enumerate(lines) if l.rstrip("\n") == keep_end),   None)
    for label, val in (("start", si), ("end", ei), ("keep_start", ksi), ("keep_end", kei)):
        if val is None:
            print(f"  [WARN] {path.name}: {label} marker not found: "
                  f"{dict(start=start_line, end=end_line, keep_start=keep_start, keep_end=keep_end)[label]!r}"
                  " — skipping")
            return
    removed = lines[si:ksi] + lines[kei + 1:ei + 1]
    _safe(removed, path)
    print(f"  {path.name}: removing chart block (keeping {keep_start!r}…{keep_end!r}):")
    for l in removed:
        print(f"    - {l.rstrip()!r}")
    path.write_text("".join(lines[:si] + lines[ksi:kei + 1] + lines[ei + 1:]), encoding="utf-8")


# ---------------------------------------------------------------------------
# Rebuild combined .md files from cached raw text
# ---------------------------------------------------------------------------

def rebuild_mds() -> None:
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

def run_checks(before_counts: dict) -> None:
    print("\n=== Checks ===")

    # (1) Forbidden strings must not appear in any raw file
    forbidden = ("Text on images", "Image text:", "Labels on machine")
    for raw in sorted(RAW_DIR.glob("*.txt")):
        text = raw.read_text(encoding="utf-8")
        for term in forbidden:
            ok = term not in text
            if not ok:
                print(f"  [FAIL] {raw.name} still contains: {term!r}")
    # If we got here with no FAIL lines, all are clean
    all_clean = all(
        term not in raw.read_text(encoding="utf-8")
        for raw in RAW_DIR.glob("*.txt")
        for term in forbidden
    )
    if all_clean:
        print(f"  [OK]   No raw file contains: {forbidden}")

    # (2) ' = ' line count unchanged for each edited file
    for path, before in before_counts.items():
        after = count_data_lines(path)
        ok = before == after
        tag = "[OK]  " if ok else "[FAIL]"
        print(f"  {tag} {path.name}: data lines before={before}, after={after}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 0: back up before any edits
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(RAW_DIR, BACKUP_DIR)
    print(f"Backed up {RAW_DIR} → {BACKUP_DIR}\n")

    # Files we will edit
    p5_brochure = RAW_DIR / "larox-pf-brochure-b-p5.txt"
    p7_brochure = RAW_DIR / "larox-pf-brochure-b-p7.txt"
    p8_brochure = RAW_DIR / "larox-pf-brochure-b-p8.txt"
    p11_orion   = RAW_DIR / "orion-series-pump-p11.txt"

    # Count data lines BEFORE edits for the safety check
    before = {p: count_data_lines(p) for p in (p5_brochure, p7_brochure, p8_brochure, p11_orion)}

    # ---- Fix 1: larox-pf-brochure-b-p5.txt ---------------------------------
    # Delete "Image text:" through EOF (page number noise at the bottom)
    print("=== Fix 1: larox-pf-brochure-b-p5.txt ===")
    remove_from_to_eof(p5_brochure, "Image text:")

    # ---- Fix 2: larox-pf-brochure-b-p7.txt ---------------------------------
    # Step a: delete the "Text on images:" block (page-number block in the middle)
    # Step b: delete the second (duplicate) "Larox PF 60 Series" title line
    print("\n=== Fix 2: larox-pf-brochure-b-p7.txt ===")
    remove_block(p7_brochure, "Text on images:", "7")
    remove_second_occurrence(p7_brochure, "Larox PF 60 Series")

    # ---- Fix 3: larox-pf-brochure-b-p8.txt ---------------------------------
    # Delete "Labels on machine:" through EOF
    print("\n=== Fix 3: larox-pf-brochure-b-p8.txt ===")
    remove_from_to_eof(p8_brochure, "Labels on machine:")

    # ---- Fix 4: orion-series-pump-p11.txt ----------------------------------
    # Delete chart axis lines ("ft" through "25000") but keep the MM/MR model
    # name list (MM100…MR 500) which sits as a clean block inside the chart.
    print("\n=== Fix 4: orion-series-pump-p11.txt ===")
    remove_block_keep_subblock(
        p11_orion,
        start_line="ft",
        end_line="25000",
        keep_start="MM100",
        keep_end="MR 500",
    )

    rebuild_mds()
    run_checks(before)


if __name__ == "__main__":
    main()
