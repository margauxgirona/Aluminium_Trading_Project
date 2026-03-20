import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BENCHMARK = "ALI=F"

ASSETS = [
    "PICK",
    "XME",
    "XLB",
    "AA",
    "CENX",
    "KALU",
    "CSTM",
    "RIO",
    "NHYDY",
    "ACH",
    "BHP",
    "HINDALCO.NS",
    "S32.AX",
    "FCX",
    "NEM",
]

ALL_TICKERS = [BENCHMARK] + ASSETS

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

print("=" * 65)
print("  TRADING GAME #2  |  Step 1: Data Collection")
print("=" * 65)
print(f"  Period : {START_DATE}  to  {END_DATE}")
print(f"  Assets : {len(ALL_TICKERS)} tickers")
print("-" * 65)

raw        = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=True)
prices_raw = raw["Close"].copy()

print(f"\n[OK] Download complete: {prices_raw.shape[0]} days x {prices_raw.shape[1]} assets")

print("\n--- Data Cleaning ---")

nan_pct       = prices_raw.isna().mean()
valid_tickers = nan_pct[nan_pct <= 0.20].index.tolist()
dropped       = [t for t in ALL_TICKERS if t not in valid_tickers]

if dropped:
    print(f"  [!] Dropped (>20% missing): {dropped}")
else:
    print("  [OK] All assets passed the missing-data filter")

prices = prices_raw[valid_tickers].copy()
prices = prices.ffill(limit=3)
prices = prices.dropna()

print(f"  [OK] After cleaning : {prices.shape[0]} days x {prices.shape[1]} assets")
print(f"  [OK] Effective period: {prices.index[0].date()}  to  {prices.index[-1].date()}")

if BENCHMARK not in prices.columns:
    raise ValueError(
        f"Benchmark {BENCHMARK} was dropped. Consider using AA as proxy."
    )

if len(prices.columns) < 10:
    raise ValueError(
        f"Only {len(prices.columns)} assets remaining. Dataset too sparse for analysis."
    )

log_prices  = np.log(prices)
log_returns = log_prices.diff().dropna()

print(f"  [OK] Log-returns computed: {log_returns.shape[0]} observations")

ann_return = log_returns.mean() * 252
ann_vol    = log_returns.std()  * np.sqrt(252)
sharpe     = ann_return / ann_vol
skewness   = log_returns.skew()
kurtosis   = log_returns.kurt()

cum_ret    = (1 + log_returns).cumprod()
roll_max   = cum_ret.cummax()
drawdown   = (cum_ret - roll_max) / roll_max
max_dd     = drawdown.min()
calmar     = ann_return / max_dd.abs()

stats = pd.DataFrame({
    "Ann. Return (%)":      (ann_return * 100).round(2),
    "Ann. Volatility (%)":  (ann_vol    * 100).round(2),
    "Sharpe Ratio":          sharpe.round(3),
    "Max Drawdown (%)":     (max_dd     * 100).round(2),
    "Calmar Ratio":          calmar.round(3),
    "Skewness":              skewness.round(3),
    "Excess Kurtosis":       kurtosis.round(3),
}).sort_values("Ann. Return (%)", ascending=False)

print("\n" + "=" * 65)
print("  DESCRIPTIVE STATISTICS")
print("=" * 65)
print(stats.to_string())
print("=" * 65)

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle(
    "Trading Game #2 — Aluminium Universe: Data Overview",
    fontsize=14, fontweight="bold"
)

ax1 = axes[0]
prices_norm = prices / prices.iloc[0] * 100

for col in prices_norm.columns:
    if col == BENCHMARK:
        ax1.plot(prices_norm.index, prices_norm[col],
                 color="crimson", linewidth=2.5, zorder=5,
                 label=f"{BENCHMARK} — Aluminium Futures")
    else:
        ax1.plot(prices_norm.index, prices_norm[col],
                 linewidth=0.8, alpha=0.40, color="steelblue")

ax1.plot([], [], color="steelblue", linewidth=0.8, alpha=0.40,
         label=f"Related assets (n={len(valid_tickers)-1})")
ax1.set_title("Normalised Prices (base = 100)", fontsize=11)
ax1.set_ylabel("Normalised Price")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

ax2 = axes[1]
corr = log_returns.corr()
im   = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(valid_tickers)))
ax2.set_yticks(range(len(valid_tickers)))
ax2.set_xticklabels(valid_tickers, rotation=45, ha="right", fontsize=7)
ax2.set_yticklabels(valid_tickers, fontsize=7)
ax2.set_title("Correlation Matrix of Daily Log-Returns", fontsize=11)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(len(valid_tickers)):
    for j in range(len(valid_tickers)):
        val   = corr.values[i, j]
        color = "white" if abs(val) > 0.65 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=5, color=color)

plt.tight_layout()
plt.savefig("step1_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Figure saved: step1_overview.png")

prices.to_csv("prices_clean.csv")
log_prices.to_csv("log_prices.csv")
log_returns.to_csv("log_returns.csv")

print("\n[OK] Files saved:")
print("      prices_clean.csv")
print("      log_prices.csv")
print("      log_returns.csv")
print(f"\n  Final universe ({len(valid_tickers)} assets): {valid_tickers}")
print("\n>>> Step 1 complete. Run step2_cointegration.py next.")
print("=" * 65)
