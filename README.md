# VOO Constituent SMA Backtest

Institutional-style Python backtest for a constituent-level VOO trend-following strategy, built to research-grade data standards rather than around a strategy claim.

Repository: `https://github.com/Aroesler1/voo-constituent-sma-backtest`

## What it does

- **Point-in-time universe construction** from CRSP, with S&P membership history and delisting returns compounded into the final return of names that exit
- **Retail-implementable cost model**: EDGE effective spreads (Ardia, Guidotti & Kroencke, JFE 2024), opening-auction slippage, participation-based impact, and FINRA regulatory fees. Making costs the centrepiece rather than an afterthought is supported by ["Implementation Risk in Portfolio Backtesting"](https://arxiv.org/abs/2603.20319) (Yin, Miki, Lesnichenko & Gural, 2026), which ran 15 strategies across five open-source backtesting engines and found the engines agree exactly at zero cost — "isolating transaction-cost implementation as the sole source of disagreement" — with divergence reaching 3.71% for high-turnover strategies, which at 4.5x annual turnover is the regime this strategy sits in
- **Multiple-testing-aware validation**: Deflated Sharpe Ratio over the SMA-length sweep, with the full configuration grid as the trial pool
- **A universe integrity audit** (`audit_universe.py`) that checks the CRSP ticker→PERMNO mapping is point-in-time correct, and a resolver (`permno_resolution.py`) that fixes it

That last item is what this repository is currently most useful for. The strategy is a 200-day SMA — deliberately simple. The infrastructure around it is the substance, and running the audit against it produced the finding below.

## Regenerated results (2026-09): the strategy loses to the index

Rerun after the data-integrity audit below, on 1,148 CRSP-resolved constituents, 7,300 trading days, 21,836 trades.

| metric | Strategy | S&P 500 total return |
|---|---|---|
| CAGR | 8.40% | **9.98%** |
| Annualised volatility | 16.85% | 19.27% |
| Sharpe | 0.363 | **0.399** |
| Max drawdown | 54.4% | 55.2% |
| Annual turnover | 4.54 | 0.03 |
| Annualised cost | **758 bps** | 0 |

**The strategy underperforms simply holding the index, and does not reduce drawdown to compensate** (54.4% against 55.2%). Costs are the mechanism: 14.9 bps per trade at 4.5x annual turnover compounds to 758 bps a year, which is more than the entire gap to the benchmark.

### Family-wise error control across the sweep

The Deflated Sharpe asks whether the *selected* configuration beats the expected best of N noise strategies. Romano-Wolf stepwise testing asks a different question: across the whole sweep, which configurations beat the benchmark, holding the probability of **any** false rejection at 5%?

| null hypothesis | configurations significant |
|---|---|
| beats cash | **5 of 5** |
| beats the S&P 500 | **0 of 5** |

That contrast is the point. Tested against cash, every SMA length looks significant — but that says only that equities beat T-bills, which is a null no long-equity strategy can fail. Tested against actually holding the index, none of them clear the bar, and every configuration has a *negative* mean excess return (t between −0.88 and −1.35).

Choosing the weak null would have produced five significant results and a much better-looking repository. The benchmark-relative test is the one reported.

The Deflated Sharpe over the 9-configuration sweep is 0.985, meaning the *selected configuration* is unlikely to be the best of nine noise strategies. That is worth stating precisely: it says the configuration search did not manufacture the result. It does not say the result is good, and here it is not.

### Probability of Backtest Overfitting

The Deflated Sharpe asks whether the selected configuration's Sharpe survives the fact that several were tried. The **Probability of Backtest Overfitting** asks a sharper question about the selection procedure itself: pick the best configuration in sample, and how often does it land in the bottom half out of sample? Bailey, Borwein, López de Prado and Zhu, ["The Probability of Backtest Overfitting"](https://doi.org/10.2139/ssrn.2326253) (*Journal of Computational Finance* 20(4), 2017).

`statistics_mt.probability_of_backtest_overfitting` implements combinatorially symmetric cross-validation: the daily return panel is cut into 16 contiguous blocks, all C(16,8) = 12,870 ways of using half for training and the complement for testing are enumerated, and PBO is the share of those splits in which the in-sample winner ranked in the bottom half out of sample. CSCV is used rather than a single train/test cut because one cut point is arbitrary, and because the symmetric design means every block trains exactly as often as it tests.

Two things about reading the number honestly. It is computed over the **five SMA lengths that already exist** (150/175/200/225/250) — widening the sweep would make PBO look more interesting without making it more informative. And with five configurations the out-of-sample rank takes only five values, so the logit is quantised and PBO is coarse: it is an indicator, not a precise probability.

Reproduce, from the per-configuration daily returns `main.py` writes:

```bash
python run_pbo.py
```

This is the expected outcome for a 200-day SMA on index constituents, and it is reported rather than buried. The contribution of this repository is the infrastructure and the audit, not the strategy.

### What the literature says

This result is not news, and presenting it as this repository's finding would overstate it. Zakamulin has made the same case twice on far longer samples: ["The Real-Life Performance of Market Timing with Moving Average and Time-Series Momentum Rules"](https://doi.org/10.2139/ssrn.2242795) (*Journal of Asset Management* 15, 2014) argues that the published performance of moving-average timing rules contains considerable data-mining bias and ignores market frictions, and that the advantage largely disappears in out-of-sample tests carrying realistic transaction costs; ["A Comprehensive Look at the Empirical Performance of Moving Average Trading Strategies"](https://doi.org/10.2139/ssrn.2677212) (2015) reaches the same conclusion over **155 years** of data, finding no single optimal lookback and no reliable out-of-sample edge. What is contributed here is not the conclusion but the audit trail behind it: a point-in-time CRSP universe with the ticker-to-PERMNO collisions actually resolved, a cost model built from a published spread estimator rather than a flat assumption, and the multiple-testing statistics reported below.

### Caveats on these numbers

- **The sample effectively ends 2024-12-31, not 2026.** CRSP's coverage stops there while the configured end date runs later, so every current constituent is flagged as needing a "recent tail" it cannot get. An earlier draft of this section described that as a 32% coverage gap; it was not. 1,148 tickers resolved and 1,064 actually traded, with trades running 1996-10-15 to 2024-12-16. Only **51** tickers genuinely never appear in CRSP's name history, and most of those are delisted shells (`AAMRQ`, `ABKFQ`) rather than live constituents.
- **EODHD was removed entirely (2026-09).** The loader previously fell back to a
  second vendor for tickers CRSP could not resolve, and spliced a vendor tail onto
  CRSP history for current constituents. Both are gone: unresolved tickers are now
  reported and excluded, so the universe is a single consistent PERMNO-keyed panel
  and the coverage gate sees the true shortfall rather than a patched-over one.
- **Dual-class tickers needed a fix.** Vendor and index files write `BRK-B`; CRSP writes `BRK` with the class in a separate `shrcls` field, so the hyphenated form matches nothing and every dual-class name was silently lost. `permno_resolution.split_share_class` now splits the suffix and matches on ticker plus share class — `BRK-B` resolves to permno 83443, `BF-B` to 29946.
- **The buy-and-hold column in the raw output is unreliable.** It reports a 2.86% CAGR, which is implausible for the period; the ETF benchmark has CRSP coverage only from late 2014 and the contaminated cache for it was quarantined. Compare against the S&P 500 total-return column instead.
- Pre-2019 constituent history remains proxy-based rather than a licensed point-in-time master.

## The universe audit that prompted the rerun

An audit of the cached CRSP universe found that **437 of 1,153 tickers (38%) resolve to more than one PERMNO**. CRSP tickers are reused across companies and collide across share classes, and this loader resolved ticker to PERMNO without a point-in-time constraint, so a single "ticker" series could splice unrelated securities together.

Two distinct failure modes, both confirmed:

- **Sequential reuse: 327 tickers (28.4%) whose segments are separated by more than a year.** `SOLV` has a **26.7-year gap** between its two securities. The cached `VOO` series spliced **Vornado Operating Co** (permno 86379, common stock, 1998-2003) onto the **Vanguard S&P 500 ETF** (permno 12305, share code 73, 2014-2024). A 200-day SMA spanning such a boundary averages two different companies' prices, so the signal is corrupted, not merely one return.
- **Simultaneous share classes: 23 tickers with duplicate dates** (TAP, BIO, MKC, STZ, LEN, CBS, CNP). These raise inside `_extract_adjusted_series`; the exception is swallowed by a bare `except Exception` logged at DEBUG only, so the ticker is **silently dropped from the universe** - invisible selection bias.

Quantified with `quantify_contamination.py`: 288,281 of 6,198,577 rows (4.7%) fail point-in-time filtering, 27,474 returns are fabricated across splices, and 2,486 of those exceed 50% in a single day. The 200-day signal itself disagrees on only 0.01% of ticker-days, but returns above 200% still reach the P&L matrix.

Reproduce both:

```bash
python audit_universe.py --cache-dir data_cache
python quantify_contamination.py
```

The strategy evaluates the holdings underlying `VOO` on a point-in-time basis and holds only constituents trading above their `200-day SMA`, allocating capital equally across active names and routing the remainder to cash when breadth collapses.

## What This Project Does

- Builds a point-in-time constituent universe using:
  - SEC-based `VOO` holdings proxy post-2019
  - public S&P 500 membership history proxy pre-2019
- Fetches price history from `CRSP/WRDS`, the single price source
- Simulates daily, weekly, semi-monthly, and monthly rebalance schedules
- Applies realistic implementation assumptions:
  - next-session open execution
  - dynamic cash rates via FRED `DGS3MO`
  - spread, impact, slippage, and regulatory-fee modeling
  - frozen input snapshots and run manifests for reproducibility
- Produces a full research report:
  - equity curves
  - drawdowns
  - rolling Sharpe
  - active breadth
  - cost diagnostics
  - schedule comparison
  - SMA parameter sweep

## Repository Layout

```text
config.py            Runtime configuration and environment loading
data_loader.py       Vendor, snapshot, and cash-rate ingestion
preprocessing.py     Universe construction, resampling, liquidity features
strategy.py          SMA and signal generation
backtest_engine.py   Constituent-level portfolio simulation
metrics.py           Performance and risk analytics
reporting.py         Tables, charts, and markdown report generation
main.py              End-to-end pipeline entrypoint
statistics_mt.py     Deflated Sharpe, Romano-Wolf stepdown, PBO via CSCV
run_pbo.py           Probability of Backtest Overfitting for the SMA sweep
data/universe/       Source universe proxy datasets
requirements.txt     Python dependencies
```

## Strategy Definition

For each eligible constituent:

1. Compute the `200-day SMA` on adjusted close.
2. Mark the name `active` when price is above its SMA.
3. On rebalance dates, allocate equally across active names.
4. If no names are active, stay in cash.
5. Execute at the next session open with modeled implementation costs.

Default reporting schedule is `semi_monthly`, with full comparisons against `daily`, `weekly`, and `monthly`.

## Realism Features

- Point-in-time constituent membership
- Snapshot-backed vendor inputs
- CRSP price sourcing, PERMNO-keyed and point-in-time
- Time-varying cash sleeve using `DGS3MO`
- Retail-implementable cost model:
  - opening-auction slippage
  - EDGE spread estimator (Ardia, Guidotti, Kroencke, JFE 2024) by default, Corwin-Schultz retained for comparison
  - participation-based impact
  - FINRA sell-side regulatory fee support
- Output manifests for deterministic reruns

## Statistical Honesty

- The SMA-length sweep and schedule comparison constitute a multiple-testing search, so the report includes the Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014) for the selected configuration and for every sweep entry, with the full sweep treated as the trial pool. A high raw Sharpe with a low deflated Sharpe means the configuration choice is not statistically distinguishable from picking the best of several noise strategies.
- The Probability of Backtest Overfitting (CSCV; Bailey, Borwein, López de Prado & Zhu 2017) is reported alongside the Deflated Sharpe over the same five-length sweep. The two answer different questions: DSR asks whether the selected Sharpe survives the search, PBO asks whether the in-sample ranking predicts the out-of-sample ranking at all.
- Trade-level profit concentration (`profit_top5pct_share`, `profit_top10pct_share`) is reported because single-stock trend following concentrates most profit in a small tail of trades; averages alone hide this dependence.
- CRSP delisting returns (`dsedelist.dlret`) are compounded into the final return of names that exit, so departures do not silently leave at their last quoted price.

## Known Limits

- Pre-2019 constituent history is still proxy-based, not a licensed S&P point-in-time master.
- Adjusted-open execution on daily data is an approximation, even with QA repair.
- Corporate-event outliers in crisis periods can still exist in constituent data and should be reviewed before live deployment.

## Setup

Create a local environment with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and provide the credentials you have available:

```bash
cp .env.example .env
```

Supported credentials:

- `WRDS_USERNAME` / `WRDS_PASSWORD` for CRSP access
- `FRED_API_KEY` for cash-rate retrieval

## Running The Backtest

From the project root:

```bash
./.venv/bin/python main.py
```

Run the unit tests (no credentials required):

```bash
./.venv/bin/python -m pytest tests -q
```

## Generated Outputs

Successful runs write artifacts to `output/`, including:

- `results_summary.csv`
- `detailed_report.md`
- `equity_curves.png`
- `drawdowns.png`
- `rolling_sharpe.png`
- `active_breadth.png`
- `cost_diagnostics.png`
- `schedule_comparison.png`
- `schedule_risk_return.png`
- `sma_sweep.png`
- `regime_comparison.png`
- `sma_sweep_returns.csv` (per-configuration daily returns; the input `run_pbo.py` needs)
- `pbo_sweep.csv`
- `run_manifest.json`

## Intended Use

This codebase is designed for research-grade backtesting and strategy evaluation, not for direct live trading deployment without a licensed point-in-time constituent master and broker-specific transaction-cost calibration.
