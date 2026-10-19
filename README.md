# Metso Sales Assistant

A command-line sales assistant for Metso. A salesperson runs it during a call with a mining customer; it answers by calling one of three tools, chosen by the model through tool calling.

**Tools**
- `search_brochures` — product questions, grounded in the brochures (vector embeddings with keyword boost for model names)
- `estimate_service_cost` — rough cost of a field service visit from the historical reports (plain Python filtering; no currency conversion; missing totals left missing)
- `analyze_flotation_data` — checks concentrate readings against the flotation CSV (pandas; optional task)

Agent model: `gemini-3.5-flash-lite` (constant `MODEL` in `src/agent.py`). Embeddings: `gemini-embedding-001` (768 dims).

---

## Architecture

```
data/brochures/ (12 PDFs)
  └─ clean_text.py     ──► clean/               (Unicode normalisation; no API)
  └─ read_tables.py    ──► tables/raw/           (Gemini reads table/image pages; cached)
       + fix_tables.py                            (scripted field corrections; no API)
       + clean_noise.py                           (removes noise lines; no API)
                        ──► tables/              (rebuilt .md files)
  └─ build_chunks.py   ──► index/chunks.json     (190 chunks)
                            index/aliases.json    (keyword-boost aliases; no API)
  └─ embed_chunks.py   ──► index/embeddings.npy  (Gemini embedding API; cached)
                                                       ↓
                                               search_brochures

data/HistoricalServices/ (50 .txt reports)
  └─ extract_reports.py  ──► reports/raw/        (Gemini extracts JSON; cached)
  └─ build_cost_table.py ──► reports/records.json + records.csv  (no API)
                                                       ↓
                                          estimate_service_cost

data/flotation_process_data_hourly.csv
  └─ flotation_tool.py (pandas; no API)
                                                       ↓
                                         analyze_flotation_data

src/agent.py: system prompt + 3 tools + tool-calling loop + CLI
```

---

## Requirements

Python 3.13 (tested on 3.13.5).

```
pip install -r requirements.txt
```

Alternative without pinned versions:
```
pip install google-genai numpy pandas pymupdf
```

---

## Setup

1. Place the provided `data/` folder in the project root. It must contain:
   - `brochures/` — PDF brochures and their .md text extractions
   - `HistoricalServices/` — `FSR-0001.txt` through `FSR-0050.txt`
   - `flotation_process_data_hourly.csv`

   `data/` is not included in the repository.

2. Export your API key:
   ```
   export GEMINI_API_KEY="your_key_here"
   ```
   Never commit the key.

**Note:** The generated folders (`clean/`, `tables/`, `index/`, `reports/`) are git-ignored and are rebuilt by the build steps below. The provided `data/` folder is required and is not included.

---

## Build steps

Run once from the project root, in this order.

**a) Clean brochure text**
```
python src/clean_text.py
```
Reads `data/brochures/*.md`. Applies Unicode NFKC normalisation (expands ligatures, removes soft hyphens). Writes to `clean/`. No API calls.

**b) Transcribe table and image pages**
```
python src/read_tables.py --all
```
Reads the 12 brochure PDFs. Renders 12 pages to PNG (`tables/images/`) and sends each to Gemini to transcribe tables and diagrams. Uses `gemini-3.8-flash`, with `gemini-3.7-flash` and `gemini-3.6-flash` as fallbacks when a model is busy or out of daily quota. Writes raw text to `tables/raw/` (cached; already-cached pages are skipped on re-run) and assembled Markdown to `tables/*.md`. Calls the Gemini API. Free-tier rate limits (5 req/min) apply; the script waits between pages. If a daily limit is hit, re-run later — cached pages are skipped automatically.

Then apply scripted corrections to the cached raw text (no API calls):

```
python src/fix_tables.py
```
Patches known field errors in `tables/raw/` (merged-cell values the model repeated incorrectly, wrong motor ratings). Backs up `tables/raw/` to `tables/raw_backup/`, updates the files, and rebuilds `tables/*.md`.

```
python src/clean_noise.py
```
Removes noise lines (footer tags, chart axis labels, image-label blocks) from `tables/raw/`. Backs up to `tables/raw_backup2/`, updates the files, and rebuilds `tables/*.md`.

Both scripts are scripted corrections; no lines were edited by hand. The current `tables/raw/` already has both sets of corrections applied, so re-running these scripts on the existing files is a no-op. Gemini output can differ slightly on a fresh run, so the correction scripts (`fix_tables.py`, `clean_noise.py`) may need adjusting; the corrected `tables/raw/` folder is not included in the repo.

**c) Build chunks**
```
python src/build_chunks.py
```
Reads `clean/` and `tables/`. Splits content into 190 chunks, assigns model aliases, and attaches NOTE annotations for known printing errors. Writes `index/chunks.json` and `index/aliases.json`. No API calls.

**d) Embed chunks**
```
python src/embed_chunks.py
```
Reads `index/chunks.json`. Embeds each chunk with `gemini-embedding-001` (768 dims). Writes `index/embeddings.npy` and a per-run checkpoint `index/embed_cache.json`. Results are cached by hash — re-running is cheap. Calls the Gemini embeddings API; rate-limited to ~15,000 tokens/min.

**e) Extract service reports**
```
python src/extract_reports.py --all
```
Reads `data/HistoricalServices/FSR-*.txt`. Sends each report to Gemini (the 6 PF-DS reports use `gemini-3.7-flash`, fallback `gemini-3.6-flash`; the other 44 use `gemini-3.5-flash-lite`, fallback `gemini-3.1-flash-lite`) and extracts structured JSON. Writes `reports/raw/FSR-XXXX.json` (cached; already-cached reports are skipped). Calls the Gemini API; free-tier rate limits may require waiting or re-running.

**f) Build cost table**
```
python src/build_cost_table.py
```
Reads `reports/raw/`. Normalises currencies and country names; infers missing countries from site names. Writes `reports/records.json` and `reports/records.csv`, then runs validation checks. No API calls.

---

## Run the agent

```
python src/agent.py                                              # interactive
python src/agent.py --verbose                                    # show tool calls and result summaries
python src/agent.py --script tests/demo_script.txt --verbose     # batch demo
python src/agent.py --offline                                    # offline checks only, no API
```

In interactive mode, type `exit` or `quit` to end the session. In `--script` mode, a blank line starts a new conversation.

**Example prompts**
```
We have Larox pressure filters but performance drops whenever our feed grade changes
```
```
A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press.
What should I budget for the service visit?
```
```
The customer is getting 3.1% silica and 64.5% iron in their concentrate.
They think that's normal. What do you think, and what could we suggest?
```

---

## Project layout

```
test_agent/
├── README.md                this file
├── .gitignore               ignores data/, generated folders, .env
├── requirements.txt         Python dependencies (pinned versions)
├── src/
│   ├── agent.py             agent: system prompt, tool loop, CLI
│   ├── search.py            search_brochures: embedding lookup + keyword boost
│   ├── cost_tool.py         estimate_service_cost: plain Python filtering
│   ├── flotation_tool.py    analyze_flotation_data: pandas analysis
│   ├── clean_text.py        build step a: Unicode normalisation
│   ├── read_tables.py       build step b: PDF → image → Gemini transcription
│   ├── fix_tables.py        build step b (post): scripted field corrections
│   ├── clean_noise.py       build step b (post): noise-line removal
│   ├── build_chunks.py      build step c: chunk and annotate brochure content
│   ├── embed_chunks.py      build step d: embed chunks via Gemini
│   ├── extract_reports.py   build step e: extract report JSON via Gemini
│   ├── build_cost_table.py  build step f: compile and validate cost records
│   ├── profile_flotation.py optional task: profiles the flotation CSV (gaps, repeated values, correlations)
│   └── test_queries.py      retrieval test: prints top search results for sample queries
├── tests/
│   └── demo_script.txt      batch demo prompts
├── data/                    ← NOT in repo; place the provided folder here
│   ├── brochures/           12 PDFs + .md text extractions
│   ├── HistoricalServices/  50 FSR .txt files
│   └── flotation_process_data_hourly.csv
├── clean/                   ← generated; git-ignored
├── tables/                  ← generated; git-ignored
├── index/                   ← generated; git-ignored
└── reports/                 ← generated; git-ignored
```

---

## Design notes

- Models read messy input once; plain Python handles all filtering and arithmetic.
- Every model answer is cached to disk; re-running a build step skips already-processed items.
- Brochure printing errors and conflicts are kept as printed and flagged with NOTE lines in the chunk text, so the agent passes them to the user.
- No currency conversion; missing totals are not filled; countries inferred from site names are flagged as `country_inferred`.
- System prompt rules refer to tool output fields (e.g. `country_match`, `records_used`), not to specific machines, places, or report IDs.
- `MODEL` is a single constant in `src/agent.py`; switching to a paid model is a one-line change.
- The brochure search uses a dual-query approach (brochure-style query and customer wording), merged by highest score.

---

## Known limitations

- Phrasing varies between runs; all numbers come from code, not the model.
- Many currency groups rest on a single report; about 40% of reports state no cost.
- The 6 PF-DS reports were processed with `gemini-3.7-flash` (fallback `gemini-3.6-flash`); the other 44 with `gemini-3.5-flash-lite` (fallback `gemini-3.1-flash-lite`). Results were spot-checked and code-validated, not verified line by line.
- Flotation findings are correlations from one plant (largest absolute Spearman ≈ 0.22); feed columns are mostly repeated values; silica and iron concentrate are strongly inversely linked.
- Free-tier rate limits apply to `read_tables.py --all`, `embed_chunks.py`, and `extract_reports.py --all`.
