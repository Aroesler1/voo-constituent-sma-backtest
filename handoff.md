# Handoff

## Current goal

Stop presenting the "SMA loses to the index" result as this repository's
finding when it is also the literature's, and add the overfitting statistic the
sweep was missing.

## State (2026-09-03)

Done:

1. **"What the literature says"** paragraph in the README, one paragraph as
   asked, citing Zakamulin 2014 (*J. Asset Management* 15,
   [SSRN 2242795](https://doi.org/10.2139/ssrn.2242795)) and Zakamulin 2015
   ([SSRN 2677212](https://doi.org/10.2139/ssrn.2677212), 155 years). Both
   abstracts verified from source, not from memory.
2. **PBO via CSCV** implemented in
   `statistics_mt.probability_of_backtest_overfitting` (Bailey, Borwein,
   Lopez de Prado & Zhu 2017). 16 blocks, all C(16,8) = 12,870 symmetric
   splits, block-level sufficient statistics so no split materialises its data
   (~0.2 s for the whole enumeration). `run_pbo.py` reports it next to the
   Deflated Sharpe. `main.py` now persists the per-configuration daily return
   panel to `output/sma_sweep_returns.csv` and logs PBO to `output/pbo_sweep.csv`.
3. **Implementation-risk citation** next to the cost model, one sentence:
   arXiv [2603.20319](https://arxiv.org/abs/2603.20319) (Yin, Miki,
   Lesnichenko & Gural, 19 Mar 2026), verified quote "isolating
   transaction-cost implementation as the sole source of disagreement", with
   divergence up to 3.71% for high-turnover strategies. This strategy runs
   4.5x annual turnover, so it is in that regime.

- `.venv/bin/python -m pytest tests -q` -> **36 passed** (was 21).

## OPEN: the PBO number is not in the README yet

The brief assumed the per-configuration daily return series were "already
produced". They are not persisted anywhere. `main.py` built
`sweep_excess_returns` in memory and only ever wrote summary statistics
(`sma_sweep.csv`, `romano_wolf_sweep.csv`); `output/` is gitignored in any case.
So PBO cannot be computed from anything currently on disk.

`main.py` now writes `output/sma_sweep_returns.csv`, so **one full backtest run
produces the input and `run_pbo.py` then reports the number**. That run is the
blocker: the previous one took roughly two hours (output timestamps 11:41 ->
13:42), and `config.END_DATE` defaults to "today" while the CRSP cache ends
2026-03-18, so a rerun would either hit WRDS for the tail or fall back to cache
via the graceful-degradation path in `data_loader.fetch_crsp_daily`.

No PBO number has been invented in the meantime. The README describes the
method, cites the paper, and documents `run_pbo.py`; only the value is absent.

## Notes on reading PBO here

- Under CSCV the training and test blocks are **complementary**, so a
  configuration that got lucky in training must give that luck back in testing.
  A tied sweep therefore gives PBO near 1, not 0.5. A raw-noise panel is NOT a
  good test of this: one column wins the full sample by chance and then wins
  both halves, so its PBO swings with the seed (0.88 at seed 0, 0.17 at seed 1).
  `tests/test_pbo.py` uses a de-meaned panel that is exactly tied instead.
- With five configurations the out-of-sample rank takes five values, so PBO is
  quantised and coarse. Kept to the five lengths that exist (150/175/200/225/250)
  as instructed; widening the sweep would flatter the statistic without
  informing it.

## Next actions

1. Run `./.venv/bin/python main.py`, then `./.venv/bin/python run_pbo.py`, and
   put the PBO value in the README next to the Deflated Sharpe of 0.985.
2. Consider committing `output/sma_sweep_returns.csv` (5 columns x ~7,300 rows,
   a few hundred KB) so PBO reproduces from a clone rather than from a
   two-hour, WRDS-dependent rerun. `output/` is currently gitignored wholesale.
3. `config.END_DATE` resolves to "today" while CRSP coverage stops at
   2024-12-31 and the cache at 2026-03-18. Pinning it would make reruns
   offline and deterministic.
