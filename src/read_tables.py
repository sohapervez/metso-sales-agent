"""
Step B: Render PDF table pages to images and transcribe them with Gemini.

Usage:
  python src/read_tables.py          # dry run + process ONLY ac-filter p2, then stop
  python src/read_tables.py --all    # process all pages (already-cached pages are skipped)
"""

import os, sys, time
from pathlib import Path
import pymupdf as fitz                # PyMuPDF (newer import name)
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Edit this dict to change which pages are processed (1-based page numbers)
# ---------------------------------------------------------------------------
PAGES = {
    "ac-filter":           [2],
    "concorde-cell":       [5],
    "filter-optimizer":    [3],
    "larox-pf-brochure-b": [5, 6, 7, 8],
    "larox-pf-ds":         [5],
    "md-series-pump":      [2],
    "orion-series-pump":   [5, 10, 11],
}

PDF_DIR   = Path("data/brochures")
CLEAN_DIR = Path("clean")
IMG_DIR   = Path("tables/images")
RAW_DIR   = Path("tables/raw")
OUT_DIR   = Path("tables")

# Model preference order; fall through on 429/503
MODELS     = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
DPI        = 200
PAGE_WAIT  = 15   # seconds between pages (rate limit: 5 req/min)
RETRY_WAIT = 20   # seconds before a retry on 429/503
MAX_TRIES  = 3    # retries per model before trying the next model

PROMPT = (
    "This is a page from a Metso product brochure. Transcribe ALL the information\n"
    "on it as plain text. Rules:\n"
    "- For tables: write one line per product model (or per table column),\n"
    "  in the form 'Model name: field = value unit; field = value unit; ...'.\n"
    "  Every value must be attached to its model name. Repeat the table title\n"
    "  and any notes above or below the table.\n"
    "- If one value applies to several models (merged cell), write it for each\n"
    "  of those models.\n"
    "- If a cell is empty or shows '-', write 'not available'.\n"
    "- Copy numbers exactly as printed, including decimals and units. Do not\n"
    "  fix, round, or guess anything, even if a number looks wrong.\n"
    "- If you cannot read something clearly, write '[unclear]' instead of guessing.\n"
    "- For diagrams and charts: write down only the text, labels and numbers that\n"
    "  are visible. Do not interpret or add anything.\n"
    "- Output plain text only. No commentary."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pdf(brochure: str) -> Path | None:
    """Return the PDF path for a brochure, or None if not found."""
    # 1. Direct name match
    direct = PDF_DIR / f"{brochure}.pdf"
    if direct.exists():
        return direct

    # 2. Fall back to the "Source:" line in the cleaned .md
    md = CLEAN_DIR / f"{brochure}.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("Source:"):
                filename = line.split("Source:", 1)[1].strip()
                candidate = PDF_DIR / filename
                if candidate.exists():
                    return candidate
    return None


def render_page(pdf_path: Path, page_num: int, out_path: Path) -> None:
    """Render a single PDF page (1-based) to PNG at ~200 DPI."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    doc.close()


def call_gemini(client: genai.Client, img_path: Path) -> str:
    """Send the image to Gemini. Retry on 429/503; fall back through MODELS."""
    img_bytes = img_path.read_bytes()

    for model in MODELS:
        for attempt in range(MAX_TRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        PROMPT,
                    ],
                )
                return response.text
            except Exception as e:
                err = str(e)
                retryable = any(code in err for code in ("429", "503",
                                "RESOURCE_EXHAUSTED", "SERVICE_UNAVAILABLE"))
                if retryable and attempt < MAX_TRIES - 1:
                    print(f"    [{model}] {err[:80]} — retry in {RETRY_WAIT}s")
                    time.sleep(RETRY_WAIT)
                else:
                    print(f"    [{model}] failed: {err[:120]}")
                    break  # move to next model

    raise RuntimeError("All models and retries exhausted")


def write_brochure_md(brochure: str, page_texts: dict) -> None:
    """Write tables/<brochure>.md combining text for all processed pages."""
    lines = [f"# {brochure}", "Source: tables read from the PDF by Gemini"]
    for n, text in sorted(page_texts.items()):
        lines += [f"\n## Page {n}", text]
    out = OUT_DIR / f"{brochure}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run():
    print("=== DRY RUN: PDF matching and page list ===")
    for brochure, page_list in PAGES.items():
        pdf = find_pdf(brochure)
        label = str(pdf) if pdf else "PDF NOT FOUND"
        print(f"  {brochure:30s}  {label}  pages={page_list}")
    print()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pages(only_test: bool = False):
    """Render pages and call Gemini. only_test=True stops after ac-filter p2."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[int, str]] = {}
    first_api_call = True  # skip the pre-wait for the very first page

    for brochure, page_list in PAGES.items():
        pdf = find_pdf(brochure)
        if not pdf:
            print(f"[SKIP] {brochure}: PDF not found")
            continue

        page_texts: dict[int, str] = {}

        for n in page_list:
            raw_path = RAW_DIR / f"{brochure}-p{n}.txt"
            img_path = IMG_DIR / f"{brochure}-p{n}.png"

            # Cache hit — no API call needed
            if raw_path.exists():
                print(f"  [cached] {brochure} p{n}")
                page_texts[n] = raw_path.read_text(encoding="utf-8")
                continue

            print(f"  Processing {brochure} p{n} ...")

            # Render the page to PNG
            render_page(pdf, n, img_path)

            # Rate-limit gap between API calls
            if not first_api_call:
                print(f"    Waiting {PAGE_WAIT}s ...")
                time.sleep(PAGE_WAIT)
            first_api_call = False

            try:
                text = call_gemini(client, img_path)
                raw_path.write_text(text, encoding="utf-8")
                page_texts[n] = text
                print(f"    Saved raw text : {raw_path}")
                print(f"    Image saved    : {img_path}")

                # In test mode, show the text and stop immediately
                if only_test and brochure == "ac-filter" and n == 2:
                    print("\n--- Transcribed text ---")
                    print(text)
                    print("--- End of text ---")
                    print("\nTest page done. Check the output above.")
                    print("Run with --all to continue with the remaining pages.")
                    # Write the partial brochure .md so nothing is lost
                    page_texts[n] = text
                    write_brochure_md(brochure, page_texts)
                    return

            except RuntimeError as e:
                print(f"    [ERROR] {e} — skipping {brochure} p{n}")

        if page_texts:
            all_results[brochure] = page_texts

    # Write one .md per brochure after all pages are done
    print("\n=== Writing output .md files ===")
    for brochure, page_texts in all_results.items():
        write_brochure_md(brochure, page_texts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_all = "--all" in sys.argv
    dry_run()

    if run_all:
        print("=== Full run: all pages (cached pages skipped) ===")
        process_pages(only_test=False)
    else:
        print("=== Test run: processing only ac-filter p2 ===")
        process_pages(only_test=True)


if __name__ == "__main__":
    main()
