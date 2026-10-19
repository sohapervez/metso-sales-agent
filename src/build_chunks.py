"""
Stage 1: Parse clean/ and tables/ markdown files into index/chunks.json.
No API calls needed.
"""

import json, re
from pathlib import Path

# ============================================================
# Settings — edit these to change behaviour
# ============================================================

DISPLAY = {
    "ac-filter":           "AC filter",
    "cellstation":         "CellStation",
    "columncell":          "ColumnCell",
    "concorde-cell":       "Concorde Cell",
    "filter-optimizer":    "Filter optimizer",
    "floatforce-upgrade":  "FloatForce upgrade",
    "gm-filter":           "GM filter",
    "larox-pf-brochure-a": "Larox PF filter (2021 brochure)",
    "larox-pf-brochure-b": "Larox PF filter (2024 brochure)",
    "larox-pf-ds":         "Larox PF-DS",
    "md-series-pump":      "MD series mill discharge pumps",
    "orion-series-pump":   "Orion series slurry pumps",
}

# Optional prefix for row chunks; fallback = DISPLAY name
PREFIX = {
    ("larox-pf-brochure-b", 5): "Larox PF 1.6 Series",
    ("larox-pf-brochure-b", 6): "Larox PF 12 / PF 15 Series",
    ("larox-pf-brochure-b", 7): "Larox PF 60 Series",
    ("larox-pf-brochure-b", 8): "Larox PF 60 Model 2025",
    ("larox-pf-ds",          5): "Larox PF-DS tower press filter",
    ("orion-series-pump",    5): "Orion slurry pumps, materials",
    ("orion-series-pump",   10): "Orion heavy duty pumps (HM, HR)",
    ("orion-series-pump",   11): "Orion mining duty pumps (MM, MR)",
}

# Pages to skip entirely; value = reason (printed in report)
EXCLUDE = {
    ("larox-pf-brochure-a", 3): "older brochure, flattened/broken tables, superseded by brochure-b",
    ("larox-pf-brochure-a", 4): "older brochure, flattened/broken tables, superseded by brochure-b",
    ("larox-pf-brochure-a", 5): "older brochure, flattened/broken tables, superseded by brochure-b",
    ("concorde-cell",        5): "diagram labels only",
}

# Optional edition tag per brochure
EDITION = {
    "larox-pf-brochure-a": "2021",
    "larox-pf-ds":         "2025",
}

# Use tables/ for these pages; drop the clean/ version of the same page
TABLE_PAGES = {
    "ac-filter":           [2],
    "concorde-cell":       [5],
    "filter-optimizer":    [3],
    "larox-pf-brochure-b": [5, 6, 7, 8],
    "larox-pf-ds":         [5],
    "md-series-pump":      [2],
    "orion-series-pump":   [5, 10, 11],
}

# Notes: attached to matching chunks; printed values are never changed
NOTE_RULES = [
    {
        "brochure": "larox-pf-ds", "page": 5, "type": "row",
        "model_prefix": "Chambers 18",
        "note": (
            "NOTE (possible printing error in the brochure): the 25 mm chamber volume is "
            "printed as 0.13 m³, but neighbouring models show 0.811 and 1.014 m³ "
            "(the pattern suggests about 0.91). Confirm with Metso before quoting."
        ),
    },
    {
        "brochure": "orion-series-pump", "page": 11, "type": "row",
        "model_prefix": "MR350",
        "note": (
            "NOTE (possible printing error in the brochure): 1 600 kg is listed as 5 732 lbs, "
            "but 1 600 kg is about 3 527 lbs. Confirm with Metso."
        ),
    },
    {
        "brochure": "larox-pf-brochure-b", "page": 5, "type": "row",
        "model_prefix": None,  # None = every row on this page
        "note": (
            "NOTE: the brochure prints chamber volume in m² for this table; "
            "a volume should be m³. Quote it as printed and mention this."
        ),
    },
    {
        "brochure": "larox-pf-brochure-b", "page": 4, "type": "text",
        "model_prefix": None,
        "note": (
            "NOTE: this page says 1.6–180 m² filtration area and 40, 45 and 65 mm chamber "
            "depth, but the tables on pages 5-8 and the older 2021 brochure say up to 168 m² "
            "and 33, 45, 60 and 75 mm. Flag this conflict to the user."
        ),
    },
]

# Family aliases used by search.py for keyword boost
FAMILY_ALIASES = {
    "ac-filter":           ["ac filter"],
    "cellstation":         ["cellstation", "exactlevel"],
    "columncell":          ["columncell", "microcel", "sonicsparger"],
    "concorde-cell":       ["concorde", "blast tube"],
    "filter-optimizer":    ["filter optimizer"],
    "floatforce-upgrade":  ["floatforce"],
    "gm-filter":           ["gm filter"],
    "larox-pf-brochure-a": ["larox pf", "pf 1.6", "pf 12", "pf 15", "pf 60"],
    "larox-pf-brochure-b": ["larox pf", "pf 1.6", "pf 12", "pf 15", "pf 60"],
    "larox-pf-ds":         ["pf-ds", "pf ds", "tower press"],
    "md-series-pump":      ["md series", "mdm", "mdr"],
    "orion-series-pump":   ["orion"],
}

CLEAN_DIR  = Path("clean")
TABLES_DIR = Path("tables")
INDEX_DIR  = Path("index")


# ============================================================
# Parsing helpers
# ============================================================

def parse_md(path: Path) -> dict[int, str]:
    """Return {page_num: content_text} from a ## Page N markdown file."""
    pages, page, buf = {}, None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^##\s+Page\s+(\d+)\s*$', raw)
        if m:
            if page is not None:
                pages[page] = "\n".join(buf).strip()
            page, buf = int(m.group(1)), []
        elif raw.startswith("# ") or raw.startswith("Source:"):
            pass
        elif page is not None and not re.match(r'^\d+\s*$', raw):
            buf.append(raw)
    if page is not None:
        pages[page] = "\n".join(buf).strip()
    return pages


def is_row(line: str) -> bool:
    """True when the line has a ':' before its first ' = ' (table row format)."""
    eq = line.find(" = ")
    if eq == -1:
        return False
    col = line.find(":")
    return col != -1 and col < eq


def row_model(line: str) -> str:
    """Label before the first ':', with '•' and whitespace stripped."""
    col = line.find(":")
    return line[:col].replace("•", "").strip() if col != -1 else ""


def _merge_atoms(atoms: list[str], max_len: int, sep: str = "\n") -> list[str]:
    """Greedy-merge a list of atoms into chunks ≤ max_len chars."""
    chunks, cur = [], ""
    for s in atoms:
        candidate = (cur + sep + s) if cur else s
        if cur and len(candidate) > max_len:
            chunks.append(cur)
            cur = s
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks or atoms


def split_text(text: str, max_len: int = 1200) -> list[str]:
    """
    Split text into chunks ≤ max_len chars without cutting mid-sentence.
    Stage 1: blank-line paragraphs.
    Stage 2: sentence-split ('. ') any para still > max_len.
    Stage 3: line-split any piece still > max_len (e.g. dense label dumps).
    Stage 4: greedy-merge short adjacent pieces back up to max_len.
    """
    if len(text) <= max_len:
        return [text]

    # Stage 1
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Stage 2
    stage2: list[str] = []
    for para in paras:
        if len(para) <= max_len:
            stage2.append(para)
        else:
            sents = [s.strip() for s in re.split(r'(?<=\.)\s+', para) if s.strip()]
            stage2.extend(_merge_atoms(sents, max_len, sep=" ") if len(sents) > 1 else [para])

    # Stage 3
    stage3: list[str] = []
    for piece in stage2:
        if len(piece) <= max_len:
            stage3.append(piece)
        else:
            lines = [s.strip() for s in piece.splitlines() if s.strip()]
            stage3.extend(_merge_atoms(lines, max_len, sep="\n") if len(lines) > 1 else [piece])

    # Stage 4: merge short pieces back up to max_len
    return _merge_atoms(stage3, max_len, sep="\n\n") or [text]


# ============================================================
# Alias generation
# ============================================================

def get_aliases(model: str, brochure: str, page: int):
    """Return (aliases: list[str], pairs: list[[str, str]]) for a row model."""
    if not model:
        return [], []

    # "LAROX PF 12 SERIES 9.5" → ("pf 12", "9.5")
    m = re.match(r'LAROX PF (\d+) SERIES ([\d.]+)', model, re.IGNORECASE)
    if m:
        return [], [[f"pf {m.group(1)}", m.group(2)]]

    # "Chambers 16" in pf-ds → ("pf-ds", "16")
    m = re.match(r'Chambers\s+(\d+)$', model, re.IGNORECASE)
    if m and brochure == "larox-pf-ds":
        return [], [["pf-ds", m.group(1)]]

    # Pure number — look up series from PREFIX
    if re.match(r'^[\d.]+$', model):
        pstr = PREFIX.get((brochure, page), "")
        if "PF 1.6" in pstr:
            return [], [["pf 1.6", model]]
        if "PF 60" in pstr:
            return [], [["pf 60", model]]
        return [], []

    # Named model (e.g. "AC 2000", "HM100"): save if ≥ 3 alphanumerics
    if len(re.sub(r'[^a-zA-Z0-9]', '', model)) >= 3:
        return [model.lower()], []

    return [], []


# ============================================================
# Note matching
# ============================================================

def find_note(brochure: str, page: int, ctype: str, model: str) -> str:
    for rule in NOTE_RULES:
        if rule["brochure"] != brochure or rule["page"] != page or rule["type"] != ctype:
            continue
        pfx = rule["model_prefix"]
        if pfx is not None and not model.startswith(pfx):
            continue
        return rule["note"]
    return ""


# ============================================================
# Chunk builder
# ============================================================

def make_id(raw: str, seen: dict) -> str:
    """Return raw on first use, raw:N on subsequent uses (N=1,2,...)."""
    n = seen.get(raw, 0)
    seen[raw] = n + 1
    return raw if n == 0 else f"{raw}:{n}"


def build_chunks() -> list[dict]:
    chunks, seen_ids = [], {}

    for md_path in sorted(CLEAN_DIR.glob("*.md")):
        brochure = md_path.stem
        display  = DISPLAY.get(brochure, brochure)
        edition  = EDITION.get(brochure, "")

        clean_pages = parse_md(md_path)
        tbl_path    = TABLES_DIR / f"{brochure}.md"
        tbl_pages   = parse_md(tbl_path) if tbl_path.exists() else {}
        tbl_set     = set(TABLE_PAGES.get(brochure, []))

        for page in sorted(set(clean_pages) | set(tbl_pages)):
            if (brochure, page) in EXCLUDE:
                continue

            use_tbl = page in tbl_set and page in tbl_pages
            source  = "tables" if use_tbl else "clean"
            content = tbl_pages[page] if use_tbl else clean_pages.get(page, "")
            if not content.strip():
                continue

            hdr    = f"[{display} | page {page}]"
            prefix = PREFIX.get((brochure, page), display)

            if use_tbl:
                # One ROW chunk per data line; one TEXT chunk for the rest
                rest = []
                for line in content.splitlines():
                    line = line.rstrip()
                    if not line:
                        continue
                    if is_row(line):
                        model = row_model(line)
                        note  = find_note(brochure, page, "row", model)
                        text  = f"{hdr} {prefix} — {line}"
                        if note:
                            text += f"\n{note}"
                        uid = make_id(f"{brochure}:p{page}:{model}", seen_ids)
                        al, pr = get_aliases(model, brochure, page)
                        chunks.append({
                            "id": uid, "brochure": brochure, "display": display,
                            "edition": edition, "page": page, "source": source,
                            "type": "row", "model": model,
                            "text": text, "embed_text": text,
                            "note": note, "aliases": al, "pairs": pr,
                        })
                    else:
                        rest.append(line)

                rest_text = "\n".join(rest).strip()
                if len(rest_text) >= 60:
                    note = find_note(brochure, page, "text", "")
                    subs = [s for s in split_text(rest_text) if len(s.strip()) >= 60]
                    for idx, sub in enumerate(subs):
                        text = f"{hdr} {sub}"
                        if note:
                            text += f"\n{note}"
                        if len(subs) == 1:
                            uid = make_id(f"{brochure}:p{page}:text", seen_ids)
                        else:
                            uid = make_id(f"{brochure}:p{page}:text{idx+1}", seen_ids)
                        chunks.append({
                            "id": uid, "brochure": brochure, "display": display,
                            "edition": edition, "page": page, "source": source,
                            "type": "text", "model": "",
                            "text": text, "embed_text": text,
                            "note": note, "aliases": [], "pairs": [],
                        })

            else:
                # One (or more) TEXT chunks from clean/
                note = find_note(brochure, page, "text", "")
                subs = [s for s in split_text(content) if len(s.strip()) >= 60]
                for idx, sub in enumerate(subs):
                    text = f"{hdr} {sub}"
                    if note:
                        text += f"\n{note}"
                    if len(subs) == 1:
                        uid = make_id(f"{brochure}:p{page}:text", seen_ids)
                    else:
                        uid = make_id(f"{brochure}:p{page}:text{idx+1}", seen_ids)
                    chunks.append({
                        "id": uid, "brochure": brochure, "display": display,
                        "edition": edition, "page": page, "source": source,
                        "type": "text", "model": "",
                        "text": text, "embed_text": text,
                        "note": note, "aliases": [], "pairs": [],
                    })

    return chunks


# ============================================================
# Report
# ============================================================

def report(chunks: list[dict]) -> None:
    from collections import Counter

    print(f"\n{'='*60}")
    print(f"TOTAL CHUNKS: {len(chunks)}")

    print("\n--- Per brochure ---")
    for b, n in sorted(Counter(c["brochure"] for c in chunks).items()):
        row_n = sum(1 for c in chunks if c["brochure"] == b and c["type"] == "row")
        txt_n = sum(1 for c in chunks if c["brochure"] == b and c["type"] == "text")
        print(f"  {b}: {n}  (row={row_n} text={txt_n})")

    print("\n--- Per source ---")
    for src in ("clean", "tables"):
        print(f"  {src}: {sum(1 for c in chunks if c['source']==src)}")
    print("--- Per type ---")
    for t in ("row", "text"):
        print(f"  {t}: {sum(1 for c in chunks if c['type']==t)}")

    lens = [len(c["text"]) for c in chunks]
    print(f"\n--- Text lengths ---")
    imin = lens.index(min(lens))
    print(f"  Shortest: {min(lens)} chars  id={chunks[imin]['id']}")
    print("  5 longest:")
    for length, c in sorted(zip(lens, chunks), key=lambda x: -x[0])[:5]:
        print(f"    {length:5d} chars  id={c['id']}")

    print("\n--- Excluded pages ---")
    for (b, p), reason in sorted(EXCLUDE.items()):
        print(f"  {b} p{p}: {reason}")

    print("\n--- Checks ---")

    tbl_page_set = {(b, p) for b, pages in TABLE_PAGES.items() for p in pages}
    bad_clean = [c["id"] for c in chunks
                 if c["source"] == "clean" and (c["brochure"], c["page"]) in tbl_page_set]
    print(f"  {'OK' if not bad_clean else 'FAIL'}: no clean/ chunk for any tables/ page "
          f"{bad_clean or ''}")

    empty = [c["id"] for c in chunks if not c["text"].strip()]
    print(f"  {'OK' if not empty else 'FAIL'}: every chunk has non-empty text  {empty or ''}")

    ids = [c["id"] for c in chunks]
    dupes = [x for x in set(ids) if ids.count(x) > 1]
    print(f"  {'OK' if not dupes else 'FAIL'}: all ids unique  {dupes or ''}")

    note_hits = {}
    for i, rule in enumerate(NOTE_RULES):
        matched = [c["id"] for c in chunks if c["note"] == rule["note"]]
        if not matched:
            print(f"  WARNING: Note rule {i+1} found no matches "
                  f"({rule['brochure']} p{rule['page']} {rule['type']})")
        note_hits[i+1] = matched
    all_ok = all(note_hits[i] for i in range(1, 5))
    print(f"  {'OK' if all_ok else 'FAIL'}: exactly 4 note types attached")
    for i in range(1, 5):
        ids_str = note_hits[i][:4]
        more = f"…+{len(note_hits[i])-4}" if len(note_hits[i]) > 4 else ""
        print(f"    Note {i} ({len(note_hits[i])} chunks): {ids_str}{more}")

    ac_ok = any(c["brochure"] == "ac-filter" and c["page"] == 2 and c["model"] == "AC 2000"
                for c in chunks)
    print(f"  {'OK' if ac_ok else 'FAIL'}: row 'AC 2000' exists in ac-filter p2")

    print("\n--- Sample chunks ---")
    for label, cands in (("ROW",  [c for c in chunks if c["type"] == "row"]),
                          ("TEXT", [c for c in chunks if c["type"] == "text"]),
                          ("NOTE", [c for c in chunks if c["note"]])):
        if cands:
            c = cands[0]
            print(f"\n[{label}] id={c['id']}")
            print(f"  text[:250]: {c['text'][:250]!r}")
            if c.get("aliases") or c.get("pairs"):
                print(f"  aliases={c['aliases']}  pairs={c['pairs']}")


# ============================================================
# Main
# ============================================================

def main():
    INDEX_DIR.mkdir(exist_ok=True)

    print("Building chunks …")
    chunks = build_chunks()

    out = INDEX_DIR / "chunks.json"
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}  ({len(chunks)} chunks)")

    aliases_out = INDEX_DIR / "aliases.json"
    aliases_out.write_text(
        json.dumps({"family": FAMILY_ALIASES}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {aliases_out}")

    report(chunks)


if __name__ == "__main__":
    main()
