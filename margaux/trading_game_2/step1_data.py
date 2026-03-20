import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# benchmark + asset universe for aluminum (G5)
BENCHMARK = "ALI=F"

ASSETS = [
    "PICK", "XME", "XLB",
    "AA", "CENX", "KALU", "CSTM",
    "RIO", "NHYDY", "ACH",
    "BHP", "HINDALCO.NS", "S32.AX",
    "FCX", "NEM",
]

ALL_TICKERS = [BENCHMARK] + ASSETS
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

print(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}...")

raw    = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)
prices = raw["Close"].copy()

print(f"Raw data: {prices.shape[0]} days x {prices.shape[1]} assets")

# drop assets with more than 20% missing values
nan_pct = prices.isna().mean()
keep    = nan_pct[nan_pct <= 0.20].index.tolist()
dropped = [t for t in ALL_TICKERS if t not in keep]
if dropped:
    print(f"Dropped (too many NaN): {dropped}")

prices = prices[keep].ffill(limit=3).dropna()
print(f"After cleaning: {prices.shape[0]} days x {prices.shape[1]} assets")
print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")

if BENCHMARK not in prices.columns:
    raise ValueError(f"{BENCHMARK} was dropped, check data source")

# log prices and returns
log_prices  = np.log(prices)
log_returns = log_prices.diff().dropna()

# descriptive stats
ann_ret = log_returns.mean() * 252
ann_vol = log_returns.std()  * np.sqrt(252)
sharpe  = ann_ret / ann_vol
cum     = (1 + log_returns).cumprod()
mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
calmar  = ann_ret / mdd.abs()

stats = pd.DataFrame({
    "Ann. Return (%)":     (ann_ret * 100).round(2),
    "Ann. Volatility (%)": (ann_vol * 100).round(2),
    "Sharpe Ratio":         sharpe.round(3),
    "Max Drawdown (%)":    (mdd * 100).round(2),
    "Calmar Ratio":         calmar.round(3),
    "Skewness":             log_returns.skew().round(3),
    "Excess Kurtosis":      log_returns.kurt().round(3),
}).sort_values("Ann. Return (%)", ascending=False)

print("\nDescriptive Statistics:")
print(stats.to_string())

# plots
fig, axes = plt.subplots(2, 1, figsize=(14, 11))
fig.suptitle("Aluminium Universe — Data Overview", fontsize=13, fontweight="bold")

norm = prices / prices.iloc[0] * 100
for col in norm.columns:
    if col == BENCHMARK:
        axes[0].plot(norm.index, norm[col], color="crimson", lw=2.5, zorder=5, label=f"{BENCHMARK} (benchmark)")
    else:
        axes[0].plot(norm.index, norm[col], lw=0.8, alpha=0.4, color="steelblue")
axes[0].plot([], [], color="steelblue", lw=0.8, alpha=0.4, label=f"Related assets (n={len(keep)-1})")
axes[0].set_title("Normalised Prices (base 100)")
axes[0].set_ylabel("Price")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

corr = log_returns.corr()
im   = axes[1].imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1].set_xticks(range(len(keep)))
axes[1].set_yticks(range(len(keep)))
axes[1].set_xticklabels(keep, rotation=45, ha="right", fontsize=7)
axes[1].set_yticklabels(keep, fontsize=7)
axes[1].set_title("Return Correlation Matrix")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
for i in range(len(keep)):
    for j in range(len(keep)):
        v = corr.values[i, j]
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=5, color="white" if abs(v) > 0.65 else "black")

plt.tight_layout()
plt.savefig("step1_overview.png", dpi=150, bbox_inches="tight")
plt.close()

prices.to_csv("prices_clean.csv")
log_prices.to_csv("log_prices.csv")
log_returns.to_csv("log_returns.csv")

print(f"\nSaved: prices_clean.csv, log_prices.csv, log_returns.csv")
print(f"Final universe: {keep}")
