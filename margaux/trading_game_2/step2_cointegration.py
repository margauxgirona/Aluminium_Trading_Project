import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

log_prices = pd.read_csv("log_prices.csv", index_col=0, parse_dates=True)
tickers    = list(log_prices.columns)
BENCHMARK  = "ALI=F"

print(f"Loaded: {log_prices.shape[0]} days x {log_prices.shape[1]} assets")

# run engle-granger test on all pairs
print("Running cointegration tests...")

results = []
for t1, t2 in combinations(tickers, 2):
    try:
        score, pval, _ = coint(log_prices[t1], log_prices[t2])
        results.append({
            "Asset 1": t1, "Asset 2": t2,
            "t-stat": round(score, 4),
            "p-value": round(pval, 4),
            "Cointegrated": pval < 0.05,
            "With benchmark": (t1 == BENCHMARK or t2 == BENCHMARK),
        })
    except:
        pass

coint_df = pd.DataFrame(results).sort_values("p-value")
print(f"Tested {len(results)} pairs — {coint_df['Cointegrated'].sum()} cointegrated at 5%")
print("\nTop 15 pairs:")
print(coint_df.head(15).to_string(index=False))

# best pair involving ALI=F
ali_pairs = coint_df[coint_df["With benchmark"]]
print("\nPairs with ALI=F:")
print(ali_pairs.to_string(index=False))

best  = ali_pairs.iloc[0]
A1    = best["Asset 1"]
A2    = best["Asset 2"]
print(f"\nSelected pair: ({A1}, {A2}) | p-value = {best['p-value']}")

if best["p-value"] > 0.05:
    best = coint_df.iloc[0]
    A1   = best["Asset 1"]
    A2   = best["Asset 2"]
    print(f"No cointegrated pair with benchmark — using ({A1}, {A2}) instead")

coint_df.to_csv("cointegration_results.csv", index=False)

# OLS to get hedge ratio
y     = log_prices[A1].values
x     = add_constant(log_prices[A2].values)
model = OLS(y, x).fit()
beta  = model.params[1]
alpha = model.params[0]

spread = log_prices[A1] - beta * log_prices[A2] - alpha

print(f"\nSpread model: log({A1}) = {alpha:.4f} + {beta:.4f} * log({A2})")
print(f"R² = {model.rsquared:.4f}")

# ADF test on spread
adf_stat, adf_pval = adfuller(spread)[:2]
print(f"ADF test on spread: stat={adf_stat:.4f}, p={adf_pval:.4f} -> {'stationary' if adf_pval < 0.05 else 'NOT stationary'}")

# z-score with 60-day lookback (will be optimised later)
LOOKBACK  = 60
roll_mean = spread.rolling(LOOKBACK).mean()
roll_std  = spread.rolling(LOOKBACK).std()
zscore    = (spread - roll_mean) / roll_std

# plots
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"Cointegration Analysis: {A1} / {A2}", fontsize=13, fontweight="bold")

(log_prices[[A1, A2]] - log_prices[[A1, A2]].iloc[0]).plot(ax=axes[0])
axes[0].set_title("Log-Prices (demeaned)")
axes[0].grid(alpha=0.25)

spread.plot(ax=axes[1], color="darkorange", lw=1.0)
roll_mean.plot(ax=axes[1], color="black", lw=1.2, linestyle="--", label="Rolling mean")
axes[1].fill_between(spread.index, roll_mean - roll_std, roll_mean + roll_std,
                     alpha=0.15, color="grey")
axes[1].set_title(f"Spread = log({A1}) - {beta:.4f}*log({A2})")
axes[1].legend()
axes[1].grid(alpha=0.25)

zscore.plot(ax=axes[2], color="steelblue", lw=0.9)
axes[2].axhline( 1, color="red",   linestyle="--", lw=1.2)
axes[2].axhline(-1, color="green", linestyle="--", lw=1.2)
axes[2].axhline( 0, color="black", lw=0.8)
axes[2].set_title(f"Z-Score (lookback={LOOKBACK}d)")
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("step2_cointegration.png", dpi=150, bbox_inches="tight")
plt.close()

spread.to_csv("spread.csv", header=["spread"])
zscore.to_csv("zscore.csv", header=["zscore"])

pd.DataFrame([{"asset1": A1, "asset2": A2, "beta": beta, "alpha": alpha,
               "coint_pval": best["p-value"], "adf_pval": adf_pval}]).to_csv("pair_metadata.csv", index=False)

print("\nSaved: cointegration_results.csv, spread.csv, zscore.csv, pair_metadata.csv")
