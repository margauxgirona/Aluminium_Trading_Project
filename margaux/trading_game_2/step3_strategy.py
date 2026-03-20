"""
=============================================================
Trading Game #2 — Beating Passive Strategies
Step 3: Pairs Trading Strategy, Optimisation & Backtest
=============================================================
Author   : Margaux
Course   : Commodities Markets & Models — ESILV
Reference: Palazzi, R.B. (2025), Journal of Futures Markets

This script implements the full strategy:
  1. Train/test split (75% / 25%)
  2. Grid search: optimise lookback & z-score threshold
  3. Out-of-sample backtest with:
       - Volatility filter
       - Trailing stop-loss
       - Minimum holding period
       - Transaction costs
  4. Benchmark comparison (buy-and-hold ALI=F)
  5. Performance metrics & plots
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score of the spread."""
    mu  = spread.rolling(lookback).mean()
    sig = spread.rolling(lookback).std()
    return (spread - mu) / sig


def compute_spread(log_prices: pd.DataFrame, a1: str, a2: str,
                   beta: float, alpha: float) -> pd.Series:
    """Spread = log(A1) - beta*log(A2) - alpha."""
    return log_prices[a1] - beta * log_prices[a2] - alpha


def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sharpe ratio (assuming risk-free = 0)."""
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(freq)


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum drawdown from peak."""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min()


def run_strategy(spread: pd.Series,
                 log_prices: pd.DataFrame,
                 asset1: str, asset2: str, beta: float,
                 lookback: int, threshold: float,
                 transaction_cost: float = 0.002,
                 trailing_stop_factor: float = 0.025,
                 min_holding_period: int = 5,
                 vol_lookback: int = 30,
                 vol_threshold: float = 1.5) -> pd.Series:
    """
    Run the pairs trading strategy on a given period.

    Signal logic (from Palazzi 2025):
        z > +threshold  →  short spread  (short A1, long A2)
        z < -threshold  →  long spread   (long A1, short A2)
        |z| < threshold →  flat

    Returns: daily strategy returns (pd.Series)
    """
    zscore = compute_zscore(spread, lookback)

    # Rolling volatility of the spread (for vol filter)
    spread_vol     = spread.rolling(vol_lookback).std()
    spread_vol_avg = spread_vol.rolling(vol_lookback).mean()

    n       = len(spread)
    signal  = np.zeros(n)
    returns = np.zeros(n)
    position  = 0          # current position: +1, -1, 0
    hold_days = 0          # days in current position
    entry_price = np.nan   # spread value at entry (for stop-loss)

    # Daily log-returns of each leg
    r1 = log_prices[asset1].diff().values
    r2 = log_prices[asset2].values   # we use prices for stop-loss reference

    for t in range(lookback, n):
        z   = zscore.iloc[t]
        vol = spread_vol.iloc[t]
        avg = spread_vol_avg.iloc[t]

        if np.isnan(z) or np.isnan(vol) or np.isnan(avg):
            returns[t] = 0.0
            continue

        # ── Volatility filter: skip if vol is too high ──
        high_vol = vol > vol_threshold * avg

        # ── Trailing stop-loss check ──
        stop_hit = False
        if position != 0 and not np.isnan(entry_price):
            current_spread = spread.iloc[t]
            move = (current_spread - entry_price) * position
            if move < -trailing_stop_factor * abs(entry_price + 1e-8):
                stop_hit = True

        # ── Minimum holding period ──
        if position != 0:
            hold_days += 1
        can_close = (hold_days >= min_holding_period) or stop_hit

        # ── Signal generation ──
        new_signal = 0
        if not high_vol:
            if z <= -threshold:
                new_signal = 1    # long spread
            elif z >= threshold:
                new_signal = -1   # short spread

        # ── Position transitions ──
        if position == 0 and new_signal != 0:
            # Open position
            position    = new_signal
            signal[t]   = position
            hold_days   = 0
            entry_price = spread.iloc[t]
            returns[t]  = -transaction_cost   # entry cost

        elif position != 0 and can_close and (new_signal != position or stop_hit):
            # Close or reverse position
            returns[t]  = -transaction_cost   # exit cost
            position    = new_signal
            signal[t]   = position
            hold_days   = 0
            entry_price = spread.iloc[t] if position != 0 else np.nan
            if new_signal != 0:
                returns[t] -= transaction_cost  # re-entry cost

        elif position != 0:
            # Hold: return = position * daily change in spread
            # spread change ≈ r1 - beta*r2 (log-return approximation)
            signal[t]  = position
            spread_ret = log_prices[asset1].diff().iloc[t] \
                       - beta * log_prices[asset2].diff().iloc[t]
            returns[t] = position * spread_ret

    return pd.Series(returns, index=spread.index, name="strategy_returns"), \
           pd.Series(signal,  index=spread.index, name="signal")


# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  TRADING GAME #2  |  Step 3: Strategy & Backtest")
print("=" * 65)

log_prices = pd.read_csv("log_prices.csv",    index_col=0, parse_dates=True)
pair_meta  = pd.read_csv("pair_metadata.csv")

ASSET1 = pair_meta["asset1"].iloc[0]
ASSET2 = pair_meta["asset2"].iloc[0]
BETA   = pair_meta["beta"].iloc[0]
ALPHA  = pair_meta["alpha"].iloc[0]
BENCHMARK = "ALI=F"

print(f"[✓] Pair: ({ASSET1}, {ASSET2})  |  β = {BETA:.4f}")

spread = compute_spread(log_prices, ASSET1, ASSET2, BETA, ALPHA)

# ─────────────────────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT  (75% / 25%)
# ─────────────────────────────────────────────────────────────

SPLIT_RATIO = 0.75
split_idx   = int(len(spread) * SPLIT_RATIO)

spread_train = spread.iloc[:split_idx]
spread_test  = spread.iloc[split_idx:]
prices_train = log_prices.iloc[:split_idx]
prices_test  = log_prices.iloc[split_idx:]

print(f"[✓] Train: {spread_train.index[0].date()} → {spread_train.index[-1].date()} ({split_idx} days)")
print(f"[✓] Test : {spread_test.index[0].date()}  → {spread_test.index[-1].date()} ({len(spread_test)} days)")

# ─────────────────────────────────────────────────────────────
# 3.  GRID SEARCH — OPTIMISE LOOKBACK & THRESHOLD (in-sample)
# ─────────────────────────────────────────────────────────────

print("\n--- Grid Search (in-sample optimisation) ---")

LOOKBACK_GRID  = [20, 30, 40, 60, 90, 120]
THRESHOLD_GRID = [0.5, 0.7, 1.0, 1.2, 1.5]

best_sharpe   = -np.inf
best_lookback = LOOKBACK_GRID[0]
best_threshold = THRESHOLD_GRID[0]
grid_results  = []

for lb, th in itertools.product(LOOKBACK_GRID, THRESHOLD_GRID):
    rets, _ = run_strategy(
        spread_train, prices_train, ASSET1, ASSET2, BETA,
        lookback=lb, threshold=th,
        transaction_cost=0.002,
        trailing_stop_factor=0.025,
        min_holding_period=5,
        vol_lookback=30,
        vol_threshold=1.5
    )
    sr = sharpe_ratio(rets.dropna())
    grid_results.append({"lookback": lb, "threshold": th, "sharpe": round(sr, 4)})
    if sr > best_sharpe:
        best_sharpe    = sr
        best_lookback  = lb
        best_threshold = th

grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
print(f"\n  Top 5 parameter combinations:")
print(grid_df.head(5).to_string(index=False))
print(f"\n[✓] Best: lookback = {best_lookback},  threshold = {best_threshold},  Sharpe = {best_sharpe:.3f}")

# ─────────────────────────────────────────────────────────────
# 4.  OUT-OF-SAMPLE BACKTEST
# ─────────────────────────────────────────────────────────────

print("\n--- Out-of-Sample Backtest ---")

# Need the full spread to compute rolling stats across the split boundary
oos_returns, oos_signals = run_strategy(
    spread, log_prices, ASSET1, ASSET2, BETA,
    lookback=best_lookback, threshold=best_threshold,
    transaction_cost=0.002,
    trailing_stop_factor=0.025,
    min_holding_period=5,
    vol_lookback=30,
    vol_threshold=1.5
)

# Restrict to OOS period
oos_returns = oos_returns.iloc[split_idx:]
oos_signals = oos_signals.iloc[split_idx:]

# ─────────────────────────────────────────────────────────────
# 5.  BENCHMARK: BUY-AND-HOLD ALI=F
# ─────────────────────────────────────────────────────────────

if BENCHMARK in log_prices.columns:
    bh_returns = log_prices[BENCHMARK].diff().iloc[split_idx:]
else:
    # If benchmark was dropped, use Asset1
    bh_returns = log_prices[ASSET1].diff().iloc[split_idx:]

bh_returns = bh_returns.dropna()

# ─────────────────────────────────────────────────────────────
# 6.  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────

def performance_report(returns: pd.Series, name: str, n_trades: int = None) -> dict:
    rets    = returns.dropna()
    cum     = (1 + rets).cumprod()
    ann_ret = rets.mean() * 252
    mdd     = max_drawdown(cum)
    calmar  = round(ann_ret / abs(mdd), 3) if mdd != 0 else np.nan
    result  = {
        "Strategy":            name,
        "Ann. Return (%)":     round(ann_ret * 100, 2),
        "Ann. Volatility (%)": round(rets.std() * np.sqrt(252) * 100, 2),
        "Sharpe Ratio":        round(sharpe_ratio(rets), 3),
        "Max Drawdown (%)":    round(mdd * 100, 2),
        "Calmar Ratio":        calmar,
        "Total Return (%)":    round((cum.iloc[-1] - 1) * 100, 2),
        "# Trades":            n_trades if n_trades is not None else "N/A",
    }
    return result

n_trades       = int((oos_signals.diff().abs() > 0).sum())
perf_strategy  = performance_report(oos_returns, "Pairs Trading (Active)", n_trades)
perf_benchmark = performance_report(bh_returns,  "Buy-and-Hold ALI=F")

perf_df = pd.DataFrame([perf_strategy, perf_benchmark]).set_index("Strategy")

print("\n" + "=" * 65)
print("  OUT-OF-SAMPLE PERFORMANCE COMPARISON")
print("=" * 65)
print(perf_df.to_string())
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 7.  PLOTS
# ─────────────────────────────────────────────────────────────

oos_cum = (1 + oos_returns.fillna(0)).cumprod()
bh_cum  = (1 + bh_returns.fillna(0)).cumprod()
zscore_oos = compute_zscore(spread, best_lookback).iloc[split_idx:]

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.suptitle(
    f"Trading Game #2 — Out-of-Sample Results\n"
    f"Pair: {ASSET1} / {ASSET2}  |  Lookback={best_lookback}  |  Threshold={best_threshold}",
    fontsize=13, fontweight="bold"
)

# ── Cumulative returns
ax1 = axes[0]
oos_cum.plot(ax=ax1, color="steelblue", linewidth=2.0, label="Pairs Trading (Active)")
bh_cum.plot( ax=ax1, color="crimson",   linewidth=2.0, label=f"Buy-and-Hold {BENCHMARK}", linestyle="--")
ax1.set_title("Cumulative Returns — Out-of-Sample", fontsize=11)
ax1.set_ylabel("Portfolio Value (start = 1)")
ax1.legend()
ax1.grid(True, alpha=0.25)

# ── Z-score and signals
ax2 = axes[1]
zscore_oos.plot(ax=ax2, color="darkorange", linewidth=0.9, label="Z-score", alpha=0.8)
ax2.axhline( best_threshold, color="red",   linestyle="--", linewidth=1.2, label=f"+{best_threshold} (short)")
ax2.axhline(-best_threshold, color="green", linestyle="--", linewidth=1.2, label=f"−{best_threshold} (long)")
ax2.axhline( 0.0, color="black", linewidth=0.7)
# shade long/short zones
ax2.fill_between(zscore_oos.index, zscore_oos, 0,
                 where=(oos_signals == 1),  alpha=0.2, color="green", label="Long")
ax2.fill_between(zscore_oos.index, zscore_oos, 0,
                 where=(oos_signals == -1), alpha=0.2, color="red",   label="Short")
ax2.set_title(f"Z-Score & Trading Signals  (lookback={best_lookback} days)", fontsize=11)
ax2.set_ylabel("Z-Score")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# ── Drawdown
ax3 = axes[2]
drawdown = (oos_cum / oos_cum.cummax() - 1) * 100
drawdown.plot(ax=ax3, color="steelblue", linewidth=1.0, label="Strategy Drawdown")
bh_dd = (bh_cum / bh_cum.cummax() - 1) * 100
bh_dd.plot(ax=ax3, color="crimson", linewidth=1.0, linestyle="--", label="Buy-and-Hold Drawdown")
ax3.set_title("Drawdown (%)", fontsize=11)
ax3.set_ylabel("Drawdown (%)")
ax3.legend()
ax3.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("step3_backtest.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure saved → step3_backtest.png")

# ─────────────────────────────────────────────────────────────
# 8.  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

oos_returns.to_csv("oos_strategy_returns.csv", header=["strategy_returns"])
bh_returns.to_csv( "oos_bh_returns.csv",        header=["bh_returns"])
perf_df.to_csv("performance_summary.csv")
grid_df.to_csv("grid_search_results.csv", index=False)

print("[✓] Saved: oos_strategy_returns.csv, oos_bh_returns.csv,")
print("           performance_summary.csv, grid_search_results.csv")
print("\n>>> Step 3 complete. All results ready for your report.")
print("=" * 65)
