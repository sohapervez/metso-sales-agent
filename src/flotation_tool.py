"""
flotation_tool.py
Loads flotation process data and analyses a customer's concentrate reading.
pandas / numpy only — no external API calls.
"""

import json
import pandas as pd
import numpy as np

DATA_PATH = "data/flotation_process_data_hourly.csv"

SILICA_COL = "% Silica Concentrate"
IRON_COL   = "% Iron Concentrate"

# Columns excluded from process-variable analysis (low-frequency lab samples)
EXCLUDE = {"% Iron Feed", "% Silica Feed"}


def _load():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _percentile_rank(series, value):
    """Share of values <= value, as a percentage."""
    return round((series <= value).mean() * 100, 2)


def _spearman_corr(x, y):
    """Spearman correlation via rank-then-Pearson (numpy/pandas only)."""
    return float(x.rank().corr(y.rank(), method="pearson"))


def analyze_flotation(silica: float, iron: float) -> dict:
    """
    Analyse a customer concentrate reading against the plant's history.

    Parameters
    ----------
    silica : customer's % Silica Concentrate reading
    iron   : customer's % Iron Concentrate reading

    Returns
    -------
    dict with keys: position, similar_hours, good_hours, differences,
                    month_split, monthly_context, caveats, customer_settings_needed
    """
    df = _load()
    si = df[SILICA_COL]
    fe = df[IRON_COL]
    n  = len(df)

    # ── process columns (excludes feed + concentrate targets) ────────────────
    proc_cols = [c for c in df.columns
                 if c != "date" and c not in EXCLUDE
                 and c not in (SILICA_COL, IRON_COL)]

    # ── masks ────────────────────────────────────────────────────────────────
    si_q25 = si.quantile(0.25)
    fe_q75 = fe.quantile(0.75)

    mask_similar = (si.between(silica - 0.3, silica + 0.3) &
                    fe.between(iron   - 0.7, iron   + 0.7))
    mask_good    = (si <= si_q25) & (fe >= fe_q75)

    # ── 1. position ──────────────────────────────────────────────────────────
    def _stats(series):
        return {
            "mean":   round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "p05":    round(float(series.quantile(0.05)), 4),
            "p95":    round(float(series.quantile(0.95)), 4),
        }

    position = {
        "silica_percentile_rank": _percentile_rank(si, silica),
        "iron_percentile_rank":   _percentile_rank(fe, iron),
        "share_hours_silica_gte": round(float((si >= silica).mean()), 4),
        "share_hours_iron_lte":   round(float((fe <= iron).mean()), 4),
        "share_hours_both":       round(float(((si >= silica) & (fe <= iron)).mean()), 4),
        "silica_stats":           _stats(si),
        "iron_stats":             _stats(fe),
        "n_rows":                 n,
    }

    # ── 2. similar_hours ─────────────────────────────────────────────────────
    sim_months = (df[mask_similar]["date"]
                  .dt.to_period("M").astype(str)
                  .value_counts().sort_index()
                  .to_dict())
    similar_hours = {
        "count":       int(mask_similar.sum()),
        "by_month":    sim_months,
    }

    # ── 3. good_hours ────────────────────────────────────────────────────────
    good_months = (df[mask_good]["date"]
                   .dt.to_period("M").astype(str)
                   .value_counts().sort_index()
                   .to_dict())
    good_total = int(mask_good.sum())
    # flag if any single month holds more than 50 % of good hours
    good_concentrated = False
    good_concentrated_month = None
    if good_total > 0:
        top_month, top_count = max(good_months.items(), key=lambda kv: kv[1])
        if top_count / good_total > 0.50:
            good_concentrated       = True
            good_concentrated_month = top_month
    good_hours = {
        "count":                   good_total,
        "silica_q25_threshold":    round(float(si_q25), 4),
        "iron_q75_threshold":      round(float(fe_q75), 4),
        "by_month":                good_months,
        "good_hours_concentrated": good_concentrated,
    }

    # ── 4. differences ───────────────────────────────────────────────────────
    rows_similar = df[mask_similar]
    rows_good    = df[mask_good]

    diff_records = []
    for col in proc_cols:
        col_std  = float(df[col].std())
        med_sim  = float(rows_similar[col].median()) if len(rows_similar) else float("nan")
        med_good = float(rows_good[col].median())    if len(rows_good)    else float("nan")
        diff     = med_good - med_sim
        std_diff = diff / col_std if col_std > 0 else float("nan")
        spear    = _spearman_corr(df[col], si)
        diff_records.append({
            "column":               col,
            "median_similar_hours": round(med_sim,  4),
            "median_good_hours":    round(med_good, 4),
            "difference":           round(diff,     4),
            "std_diff":             round(std_diff, 4),
            "spearman_with_silica": round(spear,    4),
        })

    # sort by absolute standardised difference, take top 6
    diff_records.sort(key=lambda r: abs(r["std_diff"]), reverse=True)
    differences = {"top_6_by_abs_std_diff": diff_records[:6]}

    # ── 5. monthly_context ───────────────────────────────────────────────────
    monthly = (df.set_index("date")[[SILICA_COL, IRON_COL]]
                 .resample("ME").mean().round(4))
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly_context = monthly.rename(columns={
        SILICA_COL: "mean_silica_conc",
        IRON_COL:   "mean_iron_conc",
    }).to_dict(orient="index")

    # ── 6. caveats ───────────────────────────────────────────────────────────
    # feed column repeat shares
    feed_repeat = {}
    for col in ["% Iron Feed", "% Silica Feed"]:
        share = round(float((df[col] == df[col].shift()).mean()), 4)
        feed_repeat[col] = share

    # largest absolute Spearman correlation with silica concentrate (process cols only)
    max_abs_spearman = max(abs(r["spearman_with_silica"]) for r in diff_records)

    # silica-iron concentrate Spearman
    si_fe_spearman = round(_spearman_corr(si, fe), 4)

    # largest time gap
    gaps        = df["date"].diff().dropna()
    largest_gap = str(gaps.max())

    date_range = (df["date"].min().strftime("%Y-%m-%d"),
                  df["date"].max().strftime("%Y-%m-%d"))

    caveats = [
        (f"Feed columns are mostly repeated values "
         f"(repeat share: % Iron Feed {feed_repeat['% Iron Feed']:.1%}, "
         f"% Silica Feed {feed_repeat['% Silica Feed']:.1%})."),
        (f"Correlations between process variables and silica concentrate are weak: "
         f"largest absolute Spearman found is {max_abs_spearman:.3f}."),
        "Correlation is not causation — observed patterns may reflect confounders "
        "or operating regimes rather than direct control levers.",
        (f"Data covers one plant, {date_range[0]} to {date_range[1]}, "
         f"{n} hourly rows; patterns may not generalise to other plants or periods."),
        (f"% Silica Concentrate and % Iron Concentrate are strongly inversely "
         f"correlated (Spearman = {si_fe_spearman}); improving one typically "
         f"moves the other."),
        f"There is at least one data gap of {largest_gap}; "
        f"continuity across this period cannot be assumed.",
        "The CSV provides no column units; all values are reported exactly as recorded.",
    ]
    # conditional: warn if good hours are time-concentrated
    if good_concentrated:
        caveats.append(
            f"The good hours are concentrated in {good_concentrated_month}, "
            f"so the settings patterns may be confounded with time-of-period effects."
        )

    # ── 7. customer_settings_needed ──────────────────────────────────────────
    customer_settings_needed = {
        "value": True,
        "note": (
            "The customer's own reagent doses, air flows, and column levels were "
            "not provided. The differences above are patterns from this plant's "
            "historical data, not a diagnosis of the customer's process."
        ),
    }

    month_split = {
        "similar_hours_by_month": sim_months,
        "good_hours_by_month":    good_months,
        "good_hours_concentrated": good_concentrated,
    }

    return {
        "position":                  position,
        "similar_hours":             similar_hours,
        "good_hours":                good_hours,
        "differences":               differences,
        "month_split":               month_split,
        "monthly_context":           monthly_context,
        "caveats":                   caveats,
        "customer_settings_needed":  customer_settings_needed,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = analyze_flotation(silica=3.1, iron=64.5)

    # compact pretty-print: JSON with 2-space indent
    print(json.dumps(result, indent=2))
