"""
=============================================================
Trading Game #2 — Beating Passive Strategies
Step 2: Cointegration Testing & Pair Selection
=============================================================
Author   : Margaux
Course   : Commodities Markets & Models — ESILV
Reference: Palazzi, R.B. (2025), Journal of Futures Markets

This script:
  - Tests all asset pairs for cointegration (Engle-Granger)
  - Ranks pairs by p-value
  - Selects the best pair(s) involving ALI=F
  - Plots the spread and z-score of the best pair
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA  (output from step1_data.py)
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  TRADING GAME #2  |  Step 2: Cointegration Testing")
print("=" * 65)

log_prices = pd.read_csv("log_prices.csv", index_col=0, parse_dates=True)
tickers    = list(log_prices.columns)
BENCHMARK  = "ALI=F"

print(f"[✓] Loaded log_prices: {log_prices.shape[0]} days × {log_prices.shape[1]} assets")
print(f"    Tickers: {tickers}\n")

# ─────────────────────────────────────────────────────────────
# 2.  ENGLE-GRANGER COINTEGRATION TEST — ALL PAIRS
# ─────────────────────────────────────────────────────────────
# H0: no cointegration  →  reject if p-value < 0.05

print("--- Running Engle-Granger cointegration tests on all pairs ---")

results = []
pairs   = list(combinations(tickers, 2))

for (t1, t2) in pairs:
    try:
        score, pval, _ = coint(log_prices[t1], log_prices[t2])
        results.append({
            "Asset 1": t1,
            "Asset 2": t2,
            "t-stat":  round(score, 4),
            "p-value": round(pval, 4),
            "Cointegrated (5%)": pval < 0.05,
            "Involves ALI=F": (t1 == BENCHMARK or t2 == BENCHMARK),
        })
    except Exception as e:
        print(f"  [!] Skipped ({t1}, {t2}): {e}")

coint_df = pd.DataFrame(results).sort_values("p-value")

print(f"\n[✓] Tested {len(pairs)} pairs")
print(f"    Cointegrated at 5%: {coint_df['Cointegrated (5%)'].sum()} pairs")

print("\n--- Top 15 most cointegrated pairs ---")
print(coint_df.head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 3.  BEST PAIRS INVOLVING ALI=F (our benchmark)
# ─────────────────────────────────────────────────────────────

ali_pairs = coint_df[coint_df["Involves ALI=F"]].copy()

print("\n--- Pairs involving ALI=F (Aluminium Futures) ---")
print(ali_pairs.to_string(index=False))

# Select best pair (lowest p-value involving ALI=F)
best_row = ali_pairs.iloc[0]
ASSET1   = best_row["Asset 1"]
ASSET2   = best_row["Asset 2"]
print(f"\n[✓] Best pair selected: ({ASSET1}, {ASSET2})  |  p-value = {best_row['p-value']}")

# Fallback: if no cointegrated pair with ALI=F, use overall best pair
if best_row["p-value"] > 0.05:
    fallback = coint_df.iloc[0]
    ASSET1   = fallback["Asset 1"]
    ASSET2   = fallback["Asset 2"]
    print(f"[!] No cointegrated pair with ALI=F at 5%. Using: ({ASSET1}, {ASSET2}) instead.")

# Save the full cointegration table
coint_df.to_csv("cointegration_results.csv", index=False)
print("[✓] Saved → cointegration_results.csv")

# ─────────────────────────────────────────────────────────────
# 4.  SPREAD CONSTRUCTION  (OLS hedge ratio)
# ─────────────────────────────────────────────────────────────

y = log_prices[ASSET1].values
x = add_constant(log_prices[ASSET2].values)

model  = OLS(y, x).fit()
beta   = model.params[1]   # hedge ratio
alpha  = model.params[0]   # intercept

spread = log_prices[ASSET1] - beta * log_prices[ASSET2] - alpha

print(f"\n--- Spread Construction ---")
print(f"    Model   : log({ASSET1}) = {alpha:.4f} + {beta:.4f} × log({ASSET2}) + ε")
print(f"    R²      : {model.rsquared:.4f}")
print(f"    Hedge β : {beta:.4f}")

# ADF test on the spread  (should be stationary)
adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(spread)
print(f"\n--- ADF Test on Spread (H0: spread has unit root) ---")
print(f"    ADF stat : {adf_stat:.4f}")
print(f"    p-value  : {adf_pval:.4f}  →  {'STATIONARY ✓' if adf_pval < 0.05 else 'NOT stationary ✗'}")

# ─────────────────────────────────────────────────────────────
# 5.  Z-SCORE
# ─────────────────────────────────────────────────────────────

LOOKBACK = 60   # rolling window (days) — will be optimised in step 3

roll_mean = spread.rolling(LOOKBACK).mean()
roll_std  = spread.rolling(LOOKBACK).std()
zscore    = (spread - roll_mean) / roll_std

# ─────────────────────────────────────────────────────────────
# 6.  PLOTS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
fig.suptitle(
    f"Step 2 — Cointegration Analysis: {ASSET1} / {ASSET2}",
    fontsize=13, fontweight="bold"
)

# -- Normalised log-prices
ax1 = axes[0]
(log_prices[[ASSET1, ASSET2]] - log_prices[[ASSET1, ASSET2]].iloc[0]).plot(ax=ax1)
ax1.set_title("Log-Prices (demeaned at start)")
ax1.set_ylabel("Log-price")
ax1.legend()
ax1.grid(True, alpha=0.25)

# -- Spread
ax2 = axes[1]
spread.plot(ax=ax2, color="darkorange", linewidth=1.0)
roll_mean.plot(ax=ax2, color="black", linewidth=1.2, linestyle="--", label="Rolling mean")
ax2.fill_between(spread.index,
                 roll_mean - roll_std, roll_mean + roll_std,
                 alpha=0.15, color="grey", label="±1 std")
ax2.set_title(f"Spread  =  log({ASSET1}) − {beta:.4f}·log({ASSET2})")
ax2.set_ylabel("Spread")
ax2.legend()
ax2.grid(True, alpha=0.25)

# -- Z-score
ax3 = axes[2]
zscore.plot(ax=ax3, color="steelblue", linewidth=0.9, label="Z-score")
ax3.axhline( 1.0, color="red",   linestyle="--", linewidth=1.2, label="+1 threshold (default)")
ax3.axhline(-1.0, color="green", linestyle="--", linewidth=1.2, label="−1 threshold (default)")
ax3.axhline( 0.0, color="black", linestyle="-",  linewidth=0.8)
ax3.set_title(f"Z-Score  (lookback = {LOOKBACK} days)")
ax3.set_ylabel("Z-Score")
ax3.legend()
ax3.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("step2_cointegration.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure saved → step2_cointegration.png")

# ─────────────────────────────────────────────────────────────
# 7.  SAVE OUTPUTS  (used by step 3)
# ─────────────────────────────────────────────────────────────

spread.to_csv("spread.csv", header=["spread"])
zscore.to_csv("zscore.csv", header=["zscore"])

pair_meta = pd.DataFrame([{
    "asset1": ASSET1, "asset2": ASSET2,
    "beta": beta, "alpha": alpha,
    "coint_pval": best_row["p-value"],
    "adf_pval": adf_pval
}])
pair_meta.to_csv("pair_metadata.csv", index=False)

print("[✓] Saved → spread.csv, zscore.csv, pair_metadata.csv")
print("\n>>> Step 2 complete.  Next → run step3_strategy.py")
print("=" * 65)
