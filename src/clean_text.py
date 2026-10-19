"""
Step A: Clean brochure Markdown files.

- NFKC-normalizes text (turns ligatures like ﬁ → fi)
- Removes soft hyphens (U+00AD)
- Preserves "## Page N" markers
- Writes cleaned files to clean/ with the same filenames
- Prints a per-file report and runs a final sanity check
"""

import unicodedata
from pathlib import Path

SRC_DIR = Path("data/brochures")
OUT_DIR = Path("clean")

# Unicode ranges to check at the end
LIGATURES = set(range(0xFB00, 0xFB07))  # ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ
SOFT_HYPHEN = "­"


def clean(text: str) -> tuple[str, int, int]:
    """Return (cleaned_text, ligature_count, soft_hyphen_count)."""
    # Count ligatures before normalization
    lig_count = sum(1 for ch in text if ord(ch) in LIGATURES)

    # NFKC normalization expands ligatures and other compatibility chars
    text = unicodedata.normalize("NFKC", text)

    # Count and remove soft hyphens
    sh_count = text.count(SOFT_HYPHEN)
    text = text.replace(SOFT_HYPHEN, "")

    return text, lig_count, sh_count


def main():
    OUT_DIR.mkdir(exist_ok=True)

    md_files = sorted(SRC_DIR.glob("*.md"))
    total_files = len(md_files)
    total_lig = 0
    total_sh = 0

    for md_path in md_files:
        raw = md_path.read_text(encoding="utf-8")
        cleaned, lig_count, sh_count = clean(raw)

        out_path = OUT_DIR / md_path.name
        out_path.write_text(cleaned, encoding="utf-8")

        total_lig += lig_count
        total_sh += sh_count
        print(f"  {md_path.name}: {lig_count} ligatures, {sh_count} soft hyphens")

    print(f"\nProcessed {total_files} files  |  "
          f"total ligatures replaced: {total_lig}  |  "
          f"total soft hyphens removed: {total_sh}")

    # Sanity check: no banned characters should remain in clean/
    issues = []
    for out_path in OUT_DIR.glob("*.md"):
        text = out_path.read_text(encoding="utf-8")
        bad = [hex(ord(ch)) for ch in text if ord(ch) in LIGATURES or ch == SOFT_HYPHEN]
        if bad:
            issues.append(f"  {out_path.name}: still contains {bad}")

    if issues:
        print("\n[FAIL] Residual characters found:")
        print("\n".join(issues))
    else:
        print("\n[OK] No ligatures (U+FB00–FB06) or soft hyphens remain in clean/")


if __name__ == "__main__":
    main()
