"""
Flotation process data profiler.
Covers: basic shape/types/missing, per-column stats, frozen values,
correlations with concentrate outputs, customer point analysis, monthly trends.
"""

import pandas as pd
import numpy as np

DATA = "data/flotation_process_data_hourly.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

silica_col = "% Silica Concentrate"
iron_col   = "% Iron Concentrate"

# ═══════════════════════════════════════════════════════════════════════════════
# A) OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("A) OVERVIEW")
print("=" * 70)
print(f"Shape          : {df.shape}")
print(f"Date range     : {df['date'].min()}  →  {df['date'].max()}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

dup = df.duplicated(subset="date").sum()
print(f"\nDuplicate timestamps: {dup}")

gaps = df["date"].diff().dropna()
big_gaps = gaps[gaps > pd.Timedelta("1h")]
print(f"\nGaps > 1 h: {len(big_gaps)} total")
print("5 largest gaps:")
print(big_gaps.nlargest(5).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# B) PER-COLUMN STATS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B) PER-COLUMN STATISTICS")
print("=" * 70)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
pcts = df[num_cols].quantile([0.05, 0.50, 0.95])
summary = pd.DataFrame({
    "min"   : df[num_cols].min(),
    "p05"   : pcts.loc[0.05],
    "median": pcts.loc[0.50],
    "mean"  : df[num_cols].mean(),
    "p95"   : pcts.loc[0.95],
    "max"   : df[num_cols].max(),
})
print(summary.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# C) FROZEN VALUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C) FROZEN VALUES")
print("=" * 70)

def longest_run(series):
    """Return the length of the longest run of equal consecutive values."""
    runs = (series != series.shift()).cumsum()
    return runs.value_counts().max()

frozen = {}
for col in num_cols:
    s = df[col].dropna()
    run   = longest_run(s)
    share = (s == s.shift()).mean()
    frozen[col] = {"longest_run": run, "repeat_share": round(share, 4)}

frozen_df = pd.DataFrame(frozen).T
print(frozen_df.to_string())
high_freeze = frozen_df[frozen_df["repeat_share"] > 0.30].index.tolist()
print(f"\nColumns with repeat_share > 30%: {high_freeze}")

# ═══════════════════════════════════════════════════════════════════════════════
# D) CORRELATIONS WITH CONCENTRATE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D) CORRELATIONS WITH CONCENTRATE COLUMNS")
print("=" * 70)

other_cols = [c for c in num_cols if c not in (silica_col, iron_col)]

def spearman_series(df_sub, target_series):
    """Spearman correlation of each column with target via rank then Pearson."""
    ranked = df_sub.rank()
    ranked_target = target_series.rank()
    return ranked.corrwith(ranked_target, method="pearson")

for target in (silica_col, iron_col):
    pearson  = df[other_cols].corrwith(df[target], method="pearson")
    spearman = spearman_series(df[other_cols], df[target])
    corr = pd.DataFrame({"pearson": pearson, "spearman": spearman})
    corr["abs_pearson"] = corr["pearson"].abs()

    print(f"\n--- Top 10 by |Pearson| for '{target}' ---")
    print(corr.sort_values("abs_pearson", ascending=False).head(10)
              .drop(columns="abs_pearson").to_string())

    corr["abs_spearman"] = corr["spearman"].abs()
    print(f"\n--- Top 10 by |Spearman| for '{target}' ---")
    print(corr.sort_values("abs_spearman", ascending=False).head(10)
              .drop(columns=["abs_pearson", "abs_spearman"]).to_string())

p_both = df[silica_col].corr(df[iron_col], method="pearson")
s_both = df[silica_col].rank().corr(df[iron_col].rank(), method="pearson")
print(f"\nCorrelation between '{silica_col}' and '{iron_col}':")
print(f"  Pearson  = {p_both:.4f}")
print(f"  Spearman = {s_both:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# E) CUSTOMER READING (silica 3.1, iron 64.5)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E) CUSTOMER READING  (silica=3.1, iron=64.5)")
print("=" * 70)

cust_si, cust_fe = 3.1, 64.5
si = df[silica_col].dropna()
fe = df[iron_col].dropna()

# percentile rank: share of values <= x, expressed as a percentage
pct_si = (si <= cust_si).mean() * 100
pct_fe = (fe <= cust_fe).mean() * 100
print(f"Percentile of silica=3.1 within '{silica_col}': {pct_si:.2f}%")
print(f"Percentile of iron=64.5  within '{iron_col}'  : {pct_fe:.2f}%")

share_si_ge = (df[silica_col] >= cust_si).mean()
share_fe_le = (df[iron_col]   <= cust_fe).mean()
share_both  = ((df[silica_col] >= cust_si) & (df[iron_col] <= cust_fe)).mean()
print(f"Share of hours silica >= 3.1         : {share_si_ge:.4f} ({share_si_ge*100:.2f}%)")
print(f"Share of hours iron   <= 64.5        : {share_fe_le:.4f} ({share_fe_le*100:.2f}%)")
print(f"Share of hours BOTH                  : {share_both:.4f}  ({share_both*100:.2f}%)")

near = ((df[silica_col].between(cust_si - 0.3, cust_si + 0.3)) &
        (df[iron_col].between(cust_fe - 0.7, cust_fe + 0.7))).sum()
print(f"Hours within ±0.3 silica & ±0.7 iron : {near}")

# ═══════════════════════════════════════════════════════════════════════════════
# F) MONTHLY MEANS OF CONCENTRATE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("F) MONTHLY MEAN OF CONCENTRATE COLUMNS")
print("=" * 70)

monthly = (df.set_index("date")[[silica_col, iron_col]]
             .resample("ME")
             .mean()
             .round(4))
monthly.index = monthly.index.strftime("%Y-%m")
print(monthly.to_string())
