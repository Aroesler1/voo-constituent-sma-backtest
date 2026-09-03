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

## PBO result (run completed 2026-09-03)

`./.venv/bin/python main.py` completed in **533.6 s** (not the ~2 h the earlier
output timestamps suggested), then `run_pbo.py`:

- **PBO = 0.613** over 7,300 daily observations, 1996-01-02 to 2024-12-31,
  16 blocks, 12,870 symmetric splits, 7,296 observations used.
- median out-of-sample rank of the in-sample winner: **3.0 of 5**
- best in-sample configuration on the full sample: `sma_150`
- annualised Sharpe by length: 150 0.5944, 175 0.5854, 250 0.5755,
  200 0.5630, 225 0.5625 -- so the pre-specified 200-day is 4th of 5,
  confirming the claim already in the README.
- Romano-Wolf reproduced exactly: **0 of 5** significant vs the benchmark.

**Read 0.613 against 0.60, not 0.50.** With five configurations "bottom half"
means rank 3 or worse, so a completely uninformative ranking scores 3/5 = 0.60.
The observed value is that number: the in-sample ranking carries no
out-of-sample information. This is now in the README next to the Deflated
Sharpe of 0.985, with the point that the two are consistent -- the search did
not manufacture the result, and the configuration choice is also worthless,
because there is no edge to select over.

The brief assumed the per-configuration daily returns were "already produced".
They were not persisted anywhere: `main.py` built them in memory and wrote only
summary statistics. `main.py` now writes `output/sma_sweep_returns.csv`.

**`output/sma_sweep_returns.csv` is NOT committed.** `output/` is gitignored
wholesale. DATA.md's stated policy is that "only derived outputs are published"
and a portfolio-level daily return panel is a long way from raw CRSP, so
committing the 872 KB file would be defensible and would let PBO reproduce from
a clone without a WRDS entitlement. Not done unilaterally -- it is a
publish-licensed-derivative decision for the repository owner.

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
