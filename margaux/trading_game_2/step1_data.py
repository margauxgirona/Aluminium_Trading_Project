"""
=============================================================
Trading Game #2 — Beating Passive Strategies
Step 1: Data Collection & Cleaning — Aluminium Sector
=============================================================
Author   : Margaux
Course   : Commodities Markets & Models — ESILV
Reference: Palazzi, R.B. (2025), Journal of Futures Markets

Run this script first. It will:
  - Download 7 years of price data from Yahoo Finance
  - Clean and align the dataset
  - Compute log-returns
  - Save 3 CSV files used by the next steps
=============================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────
# The benchmark is Aluminium Futures (ALI=F).
# We build a universe of 15 related assets following the
# "Golden Rule": securities strongly tied to the commodity.

BENCHMARK = "ALI=F"

ASSETS = [
    # --- Sector ETFs ---
    "PICK",     # iShares MSCI Global Metals & Mining Producers ETF
    "XME",      # SPDR S&P Metals & Mining ETF
    "XLB",      # Materials Select Sector SPDR Fund

    # --- Pure-play aluminium producers ---
    "AA",       # Alcoa Corp.
    "CENX",     # Century Aluminum
    "KALU",     # Kaiser Aluminum
    "CSTM",     # Constellium
    "ARNC",     # Arconic Inc. (aluminium components & structures)

    # --- Major miners with large aluminium exposure ---
    "RIO",      # Rio Tinto Group
    "NHYDY",    # Norsk Hydro ASA (ADR)
    "ACH",      # Aluminum Corp. of China (Chalco, ADR)
    "BHP",      # BHP Group

    # --- Broader industrial metals complex ---
    "VALE",     # Vale SA (diversified metals)
    "FCX",      # Freeport-McMoRan (copper/metals)
    "NEM",      # Newmont Corporation
]

ALL_TICKERS = [BENCHMARK] + ASSETS   # 16 tickers total

# ─────────────────────────────────────────────────────────────
# 2.  PARAMETERS
# ─────────────────────────────────────────────────────────────

START_DATE   = "2018-01-01"
END_DATE     = "2024-12-31"   # ~7 years of daily data
MAX_NAN_PCT  = 0.20           # drop asset if > 20% values missing

# ─────────────────────────────────────────────────────────────
# 3.  DOWNLOAD  (Yahoo Finance via yfinance)
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  TRADING GAME #2  |  Step 1: Data Collection")
print("=" * 65)
print(f"  Period : {START_DATE}  →  {END_DATE}")
print(f"  Assets : {len(ALL_TICKERS)} tickers")
print("-" * 65)

raw    = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=True)
prices_raw = raw["Close"].copy()

print(f"\n[✓] Download complete: {prices_raw.shape[0]} days  ×  {prices_raw.shape[1]} assets")

# ─────────────────────────────────────────────────────────────
# 4.  CLEANING
# ─────────────────────────────────────────────────────────────

print("\n--- Data Cleaning ---")

# 4.1  Remove assets with too many missing values
nan_pct       = prices_raw.isna().mean()
valid_tickers = nan_pct[nan_pct <= MAX_NAN_PCT].index.tolist()
dropped       = [t for t in ALL_TICKERS if t not in valid_tickers]

if dropped:
    print(f"  [!] Dropped (>{int(MAX_NAN_PCT*100)}% NaN): {dropped}")
else:
    print("  [✓] All assets passed the missing-data filter")

prices = prices_raw[valid_tickers].copy()

# 4.2  Forward-fill up to 3 days (handles occasional exchange holidays)
prices = prices.ffill(limit=3)

# 4.3  Drop any remaining rows with NaN (keep only common trading dates)
prices = prices.dropna()

print(f"  [✓] After cleaning : {prices.shape[0]} days  ×  {prices.shape[1]} assets")
print(f"  [✓] Effective period: {prices.index[0].date()}  →  {prices.index[-1].date()}")

# 4.4  Ensure benchmark survived cleaning
if BENCHMARK not in prices.columns:
    raise ValueError(
        f"\n[ERROR] Benchmark '{BENCHMARK}' was dropped due to excessive missing data.\n"
        "        Consider using 'AA' or 'PICK' as an alternative benchmark.\n"
    )

# ─────────────────────────────────────────────────────────────
# 5.  LOG PRICES  &  LOG-RETURNS
# ─────────────────────────────────────────────────────────────

log_prices  = np.log(prices)
log_returns = log_prices.diff().dropna()

print(f"  [✓] Log-returns computed: {log_returns.shape[0]} observations")

# ─────────────────────────────────────────────────────────────
# 6.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────

ann_return = log_returns.mean() * 252
ann_vol    = log_returns.std()  * np.sqrt(252)
sharpe     = ann_return / ann_vol
skewness   = log_returns.skew()
kurtosis   = log_returns.kurt()

stats = pd.DataFrame({
    "Ann. Return (%)":    (ann_return * 100).round(2),
    "Ann. Volatility (%)": (ann_vol   * 100).round(2),
    "Sharpe Ratio":        sharpe.round(3),
    "Skewness":            skewness.round(3),
    "Excess Kurtosis":     kurtosis.round(3),
    "Missing (%)":         (nan_pct[valid_tickers] * 100).round(1),
}).sort_values("Ann. Return (%)", ascending=False)

print("\n" + "=" * 65)
print("  DESCRIPTIVE STATISTICS")
print("=" * 65)
print(stats.to_string())
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 7.  PLOTS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle(
    "Trading Game #2 — Aluminium Universe: Data Overview",
    fontsize=14, fontweight="bold", y=1.01
)

# ── Plot 1: Normalised prices (base 100)
ax1 = axes[0]
prices_norm = prices / prices.iloc[0] * 100

for col in prices_norm.columns:
    if col == BENCHMARK:
        ax1.plot(prices_norm.index, prices_norm[col],
                 color="crimson", linewidth=2.5, zorder=5,
                 label=f"{BENCHMARK} — Aluminium Futures (benchmark)")
    else:
        ax1.plot(prices_norm.index, prices_norm[col],
                 linewidth=0.8, alpha=0.40, color="steelblue")

ax1.plot([], [], color="steelblue", linewidth=0.8, alpha=0.40,
         label=f"Related assets (n={len(valid_tickers)-1})")
ax1.set_title("Normalised Prices  (base = 100 at start)", fontsize=11)
ax1.set_ylabel("Normalised Price")
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.25)

# ── Plot 2: Correlation heatmap
ax2 = axes[1]
corr = log_returns.corr()
im   = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(valid_tickers)))
ax2.set_yticks(range(len(valid_tickers)))
ax2.set_xticklabels(valid_tickers, rotation=45, ha="right", fontsize=8)
ax2.set_yticklabels(valid_tickers, fontsize=8)
ax2.set_title("Correlation Matrix of Daily Log-Returns", fontsize=11)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(len(valid_tickers)):
    for j in range(len(valid_tickers)):
        val   = corr.values[i, j]
        color = "white" if abs(val) > 0.65 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=6, color=color)

plt.tight_layout()
plt.savefig("step1_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure saved → step1_overview.png")

# ─────────────────────────────────────────────────────────────
# 8.  SAVE CLEANED DATA  (used by all subsequent steps)
# ─────────────────────────────────────────────────────────────

prices.to_csv("prices_clean.csv")
log_prices.to_csv("log_prices.csv")
log_returns.to_csv("log_returns.csv")

print("\n[✓] CSV files saved:")
print("      prices_clean.csv   — adjusted close prices")
print("      log_prices.csv     — log prices")
print("      log_returns.csv    — daily log-returns")
print(f"\n  Valid tickers: {valid_tickers}")
print("\n>>> Step 1 complete.  Next → run step2_cointegration.py")
print("=" * 65)
