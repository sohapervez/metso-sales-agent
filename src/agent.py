"""
src/agent.py — Metso sales-assistant agent.

  python src/agent.py                        # interactive chat
  python src/agent.py --verbose              # show tool calls + token counts
  python src/agent.py --script tests/f.txt   # batch (blank line = new conversation)
  python src/agent.py --offline              # Stage 1 offline checks, no API
"""

import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search import search, MIN_SCORE
from src.cost_tool import estimate_service_cost as _cost_fn
from src.flotation_tool import analyze_flotation as _flotation_fn

MODEL      = "gemini-3.5-flash-lite"
N_HISTORY  = 6        # user turns kept in context
TOKEN_CAP  = 12_000   # max estimated tokens per 60-second rolling window

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a sales assistant for Metso. A salesperson is talking to you during a call with a mining customer.
You have three tools. Decide for every message which one to use.

WHEN TO USE WHICH TOOL
- Product questions, specifications, comparisons, recommendations, 'which product for this problem':
  call search_brochures.
- Budget, cost or duration of a service visit, repair, inspection or part replacement:
  call estimate_service_cost.
- Concentrate quality readings (% silica, % iron) and whether they are normal or how to improve:
  call analyze_flotation_data. If a reading is missing, ask for it.
- If a message needs both (for example a recommendation and the cost of servicing it), call both.
- Greetings, thanks, or questions about neither: answer briefly without tools. Never answer product facts
  or costs from your own memory.
- If the user asks something that neither the brochures nor the service reports can cover (for example the
  price of a product, or a machine not found), say so plainly instead of guessing.

ASKING FOR MISSING INFORMATION
- For a cost question, you need the equipment and the job or part. If either is missing, ask ONE short
  question before calling the tool. If the country is missing, ask once for it too, but if the user does
  not know, proceed.
- For a product problem, search first and recommend if the results support it. Ask a short clarifying
  question only when the results are weak or several products fit equally well. When you ask, say what
  you would do with the answer.
- Ask at most one question per turn.

RULES FOR PRODUCT ANSWERS (search_brochures)
- Use only the returned text. Quote specifications exactly as written, with the unit, and name the
  brochure and page. Never invent or estimate a specification.
- Do not add explanations, conditions or reasons that are not in the returned text (for example, do not
  say a range "depends on" something unless the text says so).
- State each specification exactly as listed in its own table row. Never link two different rows or
  fields together (for example, do not say one value "depends on" another field).
- If the customer's own wording (for example a technical term) does not appear in the returned text, say
  so and explain that the match is an interpretation of what the brochure does say.
- When recommending, name one specific product and tie the reason to what the customer said. If the
  brochure only partly matches the customer's wording, say which part is a direct match and which part is
  an interpretation.
- If a returned text contains a line starting with NOTE, pass that warning on to the user in plain words
  (printing errors or conflicts between brochure pages). Do not silently pick one value.
- If weak_match is true, or the results do not contain what was asked, say that the brochures do not
  cover it. Do not fill the gap.
- If different editions or variants of a product appear with different values, say which one each value
  belongs to and ask which one the customer has if it matters.

RULES FOR COST ANSWERS (estimate_service_cost)
- Use only figures returned by the tool. Never calculate, convert or guess any amount. Never convert
  currencies; show each currency separately, as written.
- Always say how many reports each figure is based on (records_used) and name the report ids.
- If country_match is false, say that no matching report exists for the country asked, name the countries
  in countries_used, and say that local rates may differ.
- If a record has country_inferred true, say that its country was not stated in full in the report and
  was filled in from other reports that list the same site.
- If total_includes_vat is true, say the total includes VAT. If currency is 'USD?', say the currency is
  unclear ($).
- Lead with the closest_match report: its country, quantity compared to the quantity asked, on-site and
  travel hours, and its total. If partial_estimates is present, call it a partial sum of the stated lines
  because the total is missing or cut off, and list the lines that were summed.
- Then list the other matching reports as context, each with country, currency, number of cloths or parts
  if stated, and whether it is a total only. Do not scale a total by quantity. Do not compare totals across
  currencies, and say totals with and without VAT are not like for like.
- A report total covers the whole visit (labour, travel, parts), not just the requested part. Say so.
- Say which matching reports were excluded and why (for example no cost stated).
- Put the "rough estimate, not a quote" label at the end of the answer.
- If needs_clarification is true, ask the questions in ask_user first and do not quote a figure.
- Do not add reasons, causes or comparisons that the tool result does not state.

RULES FOR PLANT DATA ANSWERS (analyze_flotation_data)
- Use only numbers returned by the tool and give the number of hours each figure is based on.
- Say where the readings sit in the plant's history in plain words (percentile, share of hours), and
  answer the user's question about whether it is normal directly from those figures.
- Answer "is it normal" strictly from the returned percentiles and shares: say which percentile each
  reading is at and the share of hours as bad or worse. Do not call a reading normal if it sits in the
  worse part of the history; say it occurs but is worse than most hours.
- Describe the feed columns as mostly repeated values; do not say why.
- Present differences between similar hours and best hours as patterns in this plant's history, never
  as proven causes or instructions. Quote values as recorded; the data has no units, so do not add any.
- Present the differences as patterns to check, not causes, and give one or two suggestions worded as
  "worth checking".
- Do not interpret beyond the numbers (for example, do not describe columns as running harder).
- Always state the caveats returned by the tool, in plain words, especially the data limitations.
- Say that the customer's own reagent, air and level settings were not provided, and ask for them if
  a more specific comparison is wanted.
- Do not diagnose the customer's plant.

STYLE
- Short and direct, written for a busy salesperson on a call. Plain sentences; a short list only when
  comparing several reports. End with sources: brochure names and pages, or report ids.
- Write amounts and units as plain text (for example EUR 4,200, 8.4 hours). Never use LaTeX, $ signs
  for formatting, or markdown math.\
"""

# ── Tool declarations (dicts; converted to SDK objects at runtime) ─────────────
TOOL_DECLS = [
    {
        "name": "search_brochures",
        "description": (
            "Search the Metso product brochures. Use this for ANY question about products: what a product "
            "does, specifications, sizes, capacities, materials, features, benefits, how products compare, "
            "which product fits a problem or application, or recommendations. The brochures are the ONLY source "
            "of product facts. Do NOT use this tool for the price or cost of a service visit. "
            "query: write the search text in brochure-style technical terms describing the PROBLEM or the "
            "PRODUCT FEATURE (e.g. 'adaptive control of filter cycle under changing slurry conditions'), not the "
            "customer's casual wording. Keep exact model names, numbers and units from the user (e.g. 'AC 2000', "
            "'PF-DS', '25 mm'). "
            "customer_wording: the user's own words, copied as written."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Brochure-style technical query describing the product feature or problem.",
                },
                "customer_wording": {
                    "type": "string",
                    "description": "The user's own words, copied as written.",
                },
            },
            "required": ["query", "customer_wording"],
        },
    },
    {
        "name": "estimate_service_cost",
        "description": (
            "Estimate the cost of a field service visit from the historical service reports (not the brochures). "
            "Use this when the user asks what to budget, how much a service visit, repair, inspection or part "
            "replacement costs, or how long a past job took. Do NOT use it for product specifications. "
            "equipment: the machine as the user names it (for example a filter, flotation cell, column or pump). "
            "job_or_part: the work or part (for example 'filter cloth replacement', 'sparger seals'). "
            "quantity: how many parts or units, if stated. country: where the visit is, if stated. "
            "It returns matching past reports, costs grouped by currency (never converted), what was excluded "
            "and why, whether any report matches the requested country, and what inputs are missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equipment":   {"type": "string", "description": "The machine as the user names it."},
                "job_or_part": {"type": "string", "description": "The work or part being done or replaced."},
                "quantity":    {"type": "integer", "description": "How many parts or units, if stated."},
                "country":     {"type": "string",  "description": "Where the service visit is, if stated."},
            },
            "required": ["equipment"],
        },
    },
    {
        "name": "analyze_flotation_data",
        "description": (
            "Assess a customer's flotation concentrate quality against historical plant data "
            "(hourly readings from an iron ore flotation plant). Use this when the user gives concentrate "
            "readings such as % silica and % iron and asks whether they are normal, what they mean, or what "
            "could be improved. It computes from the data: where the readings sit in the plant's history, "
            "how hours with similar results differ from the best hours in process settings, and caveats. "
            "Do NOT use it for product specifications or service costs. "
            "silica: % silica in the concentrate. iron: % iron in the concentrate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "silica": {"type": "number", "description": "% silica in the concentrate."},
                "iron":   {"type": "number", "description": "% iron in the concentrate."},
            },
            "required": ["silica", "iron"],
        },
    },
]

# Estimated tokens for tool declarations; computed once at import time (change 4)
_DECL_TOKENS = sum(len(json.dumps(d)) for d in TOOL_DECLS if isinstance(d, dict)) // 4

# ── Thinking / reasoning filter ───────────────────────────────────────────────

# Lines that look like chain-of-thought preamble rather than a final answer.
# "Let me" and "I need to" are intentionally excluded — they can open real answers.
_THINK_LINE = re.compile(
    r'^(wait[,!.]?\s|plan:\s*|the user is asking|i should |'
    r'actually,\s|okay,?\s|so,?\s|hmm\b|right,\s|first,\s|looking at\s)',
    re.IGNORECASE,
)

def _strip_thinking(text: str) -> str:
    """Remove a leading block of chain-of-thought lines from model output.

    Returns the original text unchanged when:
    - No reasoning lines are detected.
    - The text is a single paragraph (no blank-line break) — too compact to
      safely split into preamble + answer.
    - Stripping would remove all content.
    Never returns an empty string.
    """
    if not text:
        return text
    # Single paragraph: no blank line separating reasoning from answer → leave as-is.
    if '\n\n' not in text.strip():
        return text

    lines = text.split('\n')
    cut = 0  # index one past the last preamble line
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue        # blank — keep scanning
        if _THINK_LINE.match(s):
            cut = i + 1     # this line is preamble
        else:
            break           # first genuine content line; stop

    if cut == 0:
        return text         # nothing stripped

    result = '\n'.join(lines[cut:]).strip()
    return result if result else text  # never return empty


# ── Tool A: search_brochures ───────────────────────────────────────────────────

def _run_search_brochures(query: str, customer_wording: str) -> dict:
    """Dual-query search; merge by chunk id (keep highest score); return top 6 compact."""
    r1 = search(query, k=6)
    r2 = search(customer_wording, k=6)

    by_id: dict[str, dict] = {}
    for r in r1 + r2:
        if r["id"] not in by_id or r["score"] > by_id[r["id"]]["score"]:
            by_id[r["id"]] = r

    merged = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)[:6]
    best   = merged[0]["score"] if merged else 0.0

    results = []
    for r in merged:
        m = re.match(r'\[([^|]+)\|', r.get("text", ""))
        display = m.group(1).strip() if m else r["brochure"]
        entry: dict = {
            "source": f"{display} p{r['page']}",
            "score":  round(r["score"], 3),
            "text":   r["text"][:1200],
        }
        if r.get("low_confidence"):
            entry["low_confidence"] = True
        results.append(entry)

    out: dict = {"best_score": round(best, 3), "results": results}
    if best < MIN_SCORE:
        out["weak_match"] = True
    return out


# ── Tool B: estimate_service_cost ─────────────────────────────────────────────

_TOP_FIELDS = [
    "records_used", "records_considered", "country_match", "countries_used",
    "estimate", "partial_estimates", "closest_match", "quantity_comparison",
    "excluded", "needs_clarification", "ask_user", "missing_inputs", "note",
]
_REC_FIELDS = [
    "id", "country", "country_as_written", "country_inferred",
    "currency", "currency_as_written", "onsite_hours", "travel_hours",
    "matching_parts", "total_stated", "total_includes_vat", "cost_status", "notes",
]


def _compact_cost(raw: dict) -> dict:
    """Strip nulls, drop tasks/job_summary, cap at ~6000 chars."""
    out = {k: raw[k] for k in _TOP_FIELDS if raw.get(k) is not None}

    recs = []
    for r in raw.get("matched_records") or []:
        rec = {k: r[k] for k in _REC_FIELDS if r.get(k) is not None}
        for k in ("matching_parts", "notes"):
            if rec.get(k) == []:
                del rec[k]
        recs.append(rec)
    if recs:
        out["matched_records"] = recs

    s = json.dumps(out, ensure_ascii=False)
    while len(s) > 6000 and len(out.get("matched_records") or []) > 1:
        out["matched_records"].pop()
        s = json.dumps(out, ensure_ascii=False)

    return out


def _run_estimate_cost(equipment: str, job_or_part: str = "",
                        quantity=None, country=None) -> dict:
    if quantity is not None:          # change 8: ensure int
        quantity = int(quantity)
    return _compact_cost(_cost_fn(equipment, job_or_part, quantity, country))


# ── Tool C: analyze_flotation_data ───────────────────────────────────────────

def _compact_flotation(raw: dict) -> dict:
    """Keep the fields the agent needs; drop by_month from sub-objects; cap at ~5000 chars."""
    pos = raw.get("position", {})
    sim = raw.get("similar_hours", {})
    gd  = raw.get("good_hours", {})
    out = {
        "position": pos,
        "similar_hours": {"count": sim.get("count")},
        "good_hours": {
            "count":                   gd.get("count"),
            "silica_q25_threshold":    gd.get("silica_q25_threshold"),
            "iron_q75_threshold":      gd.get("iron_q75_threshold"),
            "good_hours_concentrated": gd.get("good_hours_concentrated"),
        },
        "differences":            raw.get("differences", {}),
        "month_split":            raw.get("month_split", {}),
        "caveats":                raw.get("caveats", []),
        "customer_settings_needed": raw.get("customer_settings_needed", {}),
    }
    # enforce ~5000 char cap by trimming caveats if needed
    s = json.dumps(out, ensure_ascii=False)
    while len(s) > 5000 and len(out["caveats"]) > 1:
        out["caveats"].pop()
        s = json.dumps(out, ensure_ascii=False)
    return out


def _run_analyze_flotation(silica: float, iron: float) -> dict:
    return _compact_flotation(_flotation_fn(silica=silica, iron=iron))


# ── Agent class ───────────────────────────────────────────────────────────────

class Agent:

    def __init__(self, verbose: bool = False):
        self.verbose      = verbose
        # History stores only plain text turns: user text + model final text (change 1)
        self.history      = []
        self.sys_fallback = False
        self._client      = None
        self._tok_window  = []

    @property
    def client(self):
        if self._client is None:
            from google import genai
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise SystemExit("Set GEMINI_API_KEY before running")
            self._client = genai.Client(api_key=key)
        return self._client

    # ── Token budget ─────────────────────────────────────────────────────────

    def _est_tokens(self, contents) -> int:
        # change 4: count system prompt + declarations + every part type
        total = len(SYSTEM_PROMPT) // 4 + _DECL_TOKENS
        for c in contents:
            for p in c.parts:
                if getattr(p, "text", None):
                    total += max(1, len(p.text) // 4)
                elif getattr(p, "function_call", None):
                    fc = p.function_call
                    total += max(1, len(json.dumps(
                        {"name": fc.name, "args": dict(fc.args)}
                    )) // 4)
                elif getattr(p, "function_response", None):
                    fr = p.function_response
                    total += max(1, len(json.dumps(dict(fr.response))) // 4)
        return total

    def _rate_limit(self, est: int):
        now = time.monotonic()
        self._tok_window = [(t, n) for t, n in self._tok_window if now - t < 60]
        # change 6: guard against empty window before indexing [0]
        if self._tok_window and sum(n for _, n in self._tok_window) + est > TOKEN_CAP:
            wait = 61 - (now - self._tok_window[0][0])
            print(f"  [rate limit] waiting {wait:.0f}s ...")
            time.sleep(max(0, wait))
            self._tok_window = []
        self._tok_window.append((time.monotonic(), est))

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_api(self, raw_contents: list, with_tools: bool = True):
        """Prepend sys-prefix in fallback mode, retry on 429/503.
        with_tools=False for the final no-tool round (change 10)."""
        from google import genai
        from google.genai import types

        def _contents():
            if self.sys_fallback:
                # change 3: Part.from_text(text=...)
                return [
                    types.Content(role="user",  parts=[types.Part.from_text(text=f"Instructions:\n{SYSTEM_PROMPT}")]),
                    types.Content(role="model", parts=[types.Part.from_text(text="Understood.")]),
                ] + list(raw_contents)
            return list(raw_contents)

        def _cfg():
            kwargs: dict = dict(
                system_instruction=None if self.sys_fallback else SYSTEM_PROMPT,
                temperature=0.2,
            )
            if with_tools:
                kwargs["tools"] = [types.Tool(function_declarations=[
                    types.FunctionDeclaration(**d) for d in TOOL_DECLS if isinstance(d, dict)
                ])]
                kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
            return types.GenerateContentConfig(**kwargs)

        retries = 0
        while retries < 3:
            try:
                return self.client.models.generate_content(
                    model=MODEL, contents=_contents(), config=_cfg()
                )
            except Exception as e:
                err = str(e)
                # change 7: daily quota only when error says "PerDay" or "per day"
                if "RESOURCE_EXHAUSTED" in err and any(p in err for p in ("PerDay", "per day")):
                    raise SystemExit(f"Daily quota exhausted: {err}")
                if any(x in err for x in ("429", "503", "RESOURCE_EXHAUSTED")):
                    print(f"  [retry {retries+1}/3] rate limit, waiting 20s ...")
                    time.sleep(20)
                    retries += 1
                    continue
                # change 2: precise fallback-trigger phrases
                if not self.sys_fallback and any(
                    phrase in err.lower()
                    for phrase in ("developer instruction", "system instruction", "instruction is not enabled")
                ):
                    print("  [fallback] system_instruction rejected; switching to history mode")
                    self.sys_fallback = True
                    continue  # retry; not a rate-limit retry
                raise
        raise RuntimeError("API failed after 3 retries")

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "search_brochures":
            return _run_search_brochures(
                query           = args.get("query", ""),
                customer_wording= args.get("customer_wording", ""),
            )
        if name == "estimate_service_cost":
            return _run_estimate_cost(
                equipment   = args.get("equipment", ""),
                job_or_part = args.get("job_or_part", ""),
                quantity    = args.get("quantity"),
                country     = args.get("country"),
            )
        if name == "analyze_flotation_data":
            return _run_analyze_flotation(
                silica = float(args.get("silica", 0)),
                iron   = float(args.get("iron",   0)),
            )
        return {"error": f"unknown tool: {name}"}

    def _result_summary(self, name: str, r: dict) -> str:
        if name == "search_brochures":
            return f"best_score={r.get('best_score')}, weak={r.get('weak_match', False)}, n={len(r.get('results', []))}"
        if name == "estimate_service_cost":
            return f"records_used={r.get('records_used')}, country_match={r.get('country_match')}"
        if name == "analyze_flotation_data":
            pos = r.get("position", {})
            return (f"n_similar={r.get('similar_hours', {}).get('count')}, "
                    f"n_good={r.get('good_hours', {}).get('count')}, "
                    f"silica_pct={pos.get('silica_percentile_rank')}")
        return ""

    # ── Main chat method ──────────────────────────────────────────────────────

    def chat(self, user_msg: str) -> str:
        from google.genai import types

        # change 3: Part.from_text(text=...)
        user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])
        raw_contents = list(self.history) + [user_content]

        final_text: str = ""

        for _round in range(4):
            # change 4: estimate + rate-limit before EVERY API call
            est = self._est_tokens(raw_contents)
            if self.verbose:
                print(f"  [~{est} tokens]")
            self._rate_limit(est)

            try:
                resp = self._call_api(raw_contents)
            except SystemExit:
                raise
            except Exception as e:
                return f"[API error: {e}]"

            # change 5: guard against empty candidates
            if not resp.candidates or resp.candidates[0].content is None:
                return "[no answer from the model, try again]"

            model_content = resp.candidates[0].content
            raw_contents.append(model_content)

            fcalls = [p for p in model_content.parts if getattr(p, "function_call", None)]

            if not fcalls:
                final_text = _strip_thinking("".join(
                    p.text for p in model_content.parts
                    if getattr(p, "text", None) and not getattr(p, "thought", False)
                ))
                break

            resp_parts = []
            for p in fcalls:
                fc   = p.function_call
                name = fc.name
                args = dict(fc.args)
                if self.verbose:
                    print(f"  [→] {name}({json.dumps(args, ensure_ascii=False)[:120]})")
                try:
                    result = self._dispatch(name, args)
                except Exception as exc:
                    result = {"error": str(exc)[:200]}
                if self.verbose:
                    print(f"  [←] {self._result_summary(name, result)}")
                resp_parts.append(
                    types.Part.from_function_response(name=name, response={"result": result})
                )

            tool_content = types.Content(role="user", parts=resp_parts)
            raw_contents.append(tool_content)
        else:
            # change 10: tool rounds exhausted — one final call without tools
            est = self._est_tokens(raw_contents)
            if self.verbose:
                print(f"  [~{est} tokens, final no-tool call]")
            self._rate_limit(est)
            try:
                resp = self._call_api(raw_contents, with_tools=False)
                if not resp.candidates or resp.candidates[0].content is None:
                    final_text = "[no answer from the model, try again]"
                else:
                    final_text = _strip_thinking("".join(
                        p.text for p in resp.candidates[0].content.parts
                        if getattr(p, "text", None) and not getattr(p, "thought", False)
                    ))
            except Exception as e:
                final_text = f"[Error on final call: {e}]"

        # change 1: store only user text + model final text in history (no tool parts)
        self.history.append(types.Content(role="user",  parts=[types.Part.from_text(text=user_msg)]))
        self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=final_text or "(no response)")]))

        # Trim to last N_HISTORY user turns
        ui = [i for i, c in enumerate(self.history) if c.role == "user"]
        if len(ui) > N_HISTORY:
            self.history = self.history[ui[-N_HISTORY]:]

        return final_text

    def reset(self):
        self.history = []


# ── Stage 1 offline checks (no API) ──────────────────────────────────────────

def run_offline_checks():
    print("=" * 60)
    print("STAGE 1 — OFFLINE CHECKS")
    print("=" * 60)

    # 1. Tool declarations
    print("\nTool declarations:")
    for d in TOOL_DECLS:
        if not isinstance(d, dict):
            continue
        props = list(d["parameters"]["properties"].keys())
        req   = d["parameters"]["required"]
        print(f"  {d['name']}")
        print(f"    params:   {props}")
        print(f"    required: {req}")
        print(f"    desc[:80]: {d['description'][:80]}...")

    # 2. change 9: show raw top-level keys and confirm compact has partial_estimates
    print("\nTest estimate_service_cost (raw and compact):")
    raw_cost = _cost_fn("Larox PF-DS", "filter cloth", 12, "Chile")
    print(f"  Raw keys:      {list(raw_cost.keys())}")
    compact = _run_estimate_cost("Larox PF-DS", "filter cloth", 12, "Chile")
    s = json.dumps(compact, ensure_ascii=False)
    print(f"  records_used   = {compact.get('records_used')}")
    print(f"  country_match  = {compact.get('country_match')}")
    print(f"  estimate keys  = {list(compact.get('estimate', {}).keys())}")
    print(f"  closest_match  = {compact.get('closest_match')}")
    key = "partial_estimates"
    print(f"  '{key}' present: {'YES ✓' if key in compact else 'NO ✗'}")
    print(f"  output length  = {len(s)} chars")

    # 3. search_brochures needs the embedding API
    print("\nTest _run_search_brochures: skipped (requires GEMINI_API_KEY)")

    # 3b. analyze_flotation_data compact output check
    print("\nTest analyze_flotation_data (compact):")
    flotation_result = _run_analyze_flotation(silica=2.0, iron=65.0)
    s_fl = json.dumps(flotation_result, ensure_ascii=False)
    print(f"  keys:          {list(flotation_result.keys())}")
    print(f"  n_similar      = {flotation_result.get('similar_hours', {}).get('count')}")
    print(f"  n_good         = {flotation_result.get('good_hours', {}).get('count')}")
    print(f"  output length  = {len(s_fl)} chars (cap ~5000)")

    # 4. System prompt integrity checks
    sp = len(SYSTEM_PROMPT)
    print(f"\nSystem prompt: {sp} chars (~{sp // 4} tokens)")
    for word in ["Chile", "PF-DS", "2,135", "3.1", "64.5"]:
        status = "FAIL ✗" if word in SYSTEM_PROMPT else "OK ✓"
        print(f"  '{word}' in system prompt: {status}")

    # 5. _DECL_TOKENS
    print(f"\nTool decl tokens (estimated): {_DECL_TOKENS}")

    # 6. _strip_thinking before/after demo
    def _demo(label: str, sample: str) -> None:
        result = _strip_thinking(sample)
        changed = result != sample.strip()
        print(f"\n  [{label}]  stripped={changed}")
        print(f"    IN : {sample.splitlines()[0]!r}{'...' if len(sample.splitlines()) > 1 else ''}")
        print(f"    OUT: {result.splitlines()[0]!r}{'...' if len(result.splitlines()) > 1 else ''}")

    print("\n_strip_thinking demos:")
    _demo(
        "multi-para reasoning → stripped",
        "Wait, let me check the records.\n"
        "The user is asking about filter cloth costs.\n"
        "Plan: call estimate_service_cost.\n"
        "\n"
        "Based on 4 past service reports, the budget is approximately "
        "CAD 4,367 or EUR 2,580.",
    )
    _demo(
        "single paragraph → unchanged",
        "Wait, let me check — the cost is approximately CAD 4,367.",
    )
    _demo(
        "no reasoning → unchanged",
        "Based on 4 past service reports, the budget is CAD 4,367.",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Metso sales assistant")
    ap.add_argument("--verbose", action="store_true", help="Show tool calls and token counts")
    ap.add_argument("--script",  metavar="FILE",      help="Batch file: blank line = new conversation")
    ap.add_argument("--offline", action="store_true", help="Stage 1 offline checks only")
    args = ap.parse_args()

    if args.offline or not os.environ.get("GEMINI_API_KEY"):
        run_offline_checks()
        if not args.offline:
            print("\nNo GEMINI_API_KEY found. Set it to run interactive or script mode:")
            print("  export GEMINI_API_KEY=<your-key>")
            print("  python src/agent.py --verbose")
            print("  python src/agent.py --script tests/demo_script.txt --verbose")
        return

    if args.script:
        text  = Path(args.script).read_text(encoding="utf-8")
        convs = [c.strip() for c in text.split("\n\n") if c.strip()]
        for ci, conv in enumerate(convs, 1):
            print(f"\n{'='*60}\nConversation {ci}\n{'='*60}")
            agent = Agent(verbose=args.verbose)
            for line in conv.splitlines():
                if line.strip():
                    print(f"\nYou: {line}")
                    print(f"Agent: {agent.chat(line)}")
            mode = "history-fallback" if agent.sys_fallback else "system_instruction"
            print(f"[mode: {mode}]")
        return

    agent = Agent(verbose=args.verbose)
    print("Metso Sales Assistant  (type 'exit' to quit)")
    reported_mode = False
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in ("exit", "quit"):
            break
        answer = agent.chat(msg)
        if not reported_mode:
            mode = "history-fallback" if agent.sys_fallback else "system_instruction"
            print(f"  [mode: {mode}]")
            reported_mode = True
        print(f"Agent: {answer}")
    print("Goodbye.")


if __name__ == "__main__":
    main()
