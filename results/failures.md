# Documented failures

Things that went wrong while building this, with the mechanism and the fix.
These are kept because the reasoning is more useful than the final numbers.

## 1. Tail-trimming biased the process scale β low by 2×

The first variance regression trimmed the top 2% of squared residuals "for
robustness". For a gamma process with monthly shape ≈ 0.27, most of the
variance lives in that tail; trimming it returned β ≈ 0.37 against a true
0.80, and the downstream α estimates were 2.8× high with the rate αβ still
"correct". Sensor σ was recovered exactly, which hid the problem: a checked
parameter can look fine while the one it trades off against is wrong.

**Fix:** no trimming of increments. Robustness to faulted sensors comes from
rejecting whole *units* whose variance statistic is an outlier, and from the
second fit pass that excludes channels the detector has flagged. Unit-level
rejection does not truncate the distribution; increment-level trimming does.

## 2. Increment-level OLS on squared residuals collapsed σ to its floor

With trimming removed, regressing e² on [rate·du, 1] at the increment level
was dominated by a few enormous squared residuals from faulted channels and
returned a negative intercept (σ² < 0 → floored). Aggregating to unit level
(sum of ~60 increments) reduced skewness enough for least squares to be
stable, and it made the outlier rejection in (1) possible.

## 3. Testing against the channel's own fitted rate made long accelerations invisible

The first change-point statistic compared post-onset increments to the
channel's fitted α. Because α was pooled across the channel's whole history,
a long-running 2× acceleration was absorbed into α itself, and the longer the
fault had run, the *less* detectable it became — the opposite of what any
reasonable test should do. The monotonic reversal in power vs duration was the
clue. **Fix:** estimate α from the pre-onset window only (blended with the
fleet prior), and fold its uncertainty into the test variance.

## 4. A "self-calibrated" threshold that wasn't

An earlier E2 statistic set its threshold at the 99.9th percentile of a
rolling-window score over the 90% of channels with the smallest maxima.
Calibrating on windows and flagging on channel maxima are different tests;
the result was 979 false alarms (8% of clean channels). Replaced by an exact
gamma-tail p-value with Bonferroni correction over candidate onsets, so the
false-alarm rate is a stated number (0.1% per channel), and the achieved rate
(0.43%, including the spike test) is measured in verify.py.

## 5. A spike p-value that ignored sensor noise

The single-increment spike test used the pure gamma tail. On smooth,
low-β components the noise variance is larger than the process variance per
increment, so the test produced >100 false spikes. Moment-matching a gamma to
the noise-inclusive variance fixed it.

## 6. A corruption that was undetectable by construction

The first "negative reading" corruption subtracted 0.01. Readings near zero
legitimately go slightly negative from sensor noise, so this corruption was
indistinguishable from valid data and any "detection rate" for it would have
been meaningless. The corruption was made unambiguous; `units_x10` on small
readings remains partially undetectable at ingest (0.88 caught) and this is
stated rather than tuned away.

## 7. Month-1 over-prediction of first failures

The expected-failure curve over-predicts month 1 (423 vs 327 actual) and is
well calibrated thereafter (15/18 months inside the 90% band). Units whose
last reading sits just under the estimated threshold are assigned near-certain
immediate failure, but the threshold estimate is a 97th-percentile lower
bound and the reading carries noise. This is visible, not hidden, and the
fix (a noise-aware posterior on x0) is listed under future work.

---

The entries below came from an adversarial review of a version that passed
every test it had. All of them produced a plausible answer rather than an
error, which is the class this pipeline exists to defend against.

## 8. Detecting a fault made the forecast worse than a constant

The second fit pass excluded *every* flagged channel, so all 338 came back
with shrinkage 1.0 and took the fleet-mean severity. Scored against truth,
that was actively harmful:

| class | caught (before) | missed (before) |
|---|---|---|
| accelerated | Brier 0.198 | Brier 0.156 |

The units the detector correctly identified as degrading twice as fast were
forecast *worse* than the ones it missed. Every caught unit subsequently
failed; the model gave them 0.66.

The mechanism is a category error. Class 5 (`accelerated`) is not a sensor
fault — it is real damage. Responding to "this unit is wearing out at twice
the fleet rate" by replacing its severity with the fleet average is exactly
backwards. `dropout` was the same mistake in milder form: those records are
*missing*, not wrong, and the readings that survived were clean.

**Fix:** exclude only the classes that corrupt the reading itself — bias
step, scale error, stuck. Keep dropout and accelerated channels in the fit.
After the change:

| class | caught | missed |
|---|---|---|
| accelerated | **0.000** | 0.160 |
| dropout | **0.118** | 0.309 |

`verify.py` now asserts caught-versus-missed permanently, because nothing in
the suite would otherwise notice a detector that degrades the thing it feeds.

## 9. The sampling variance used the increment count where the derivation
requires the segment count

`Sd = Σ dᵢ` telescopes within a serial to (last reading − first reading), so
the measurement noise it carries is `2σ²` per **serial segment**, not per
increment. `detect_faults.m` states that rule in a comment at the top and
applies it correctly to the post-onset window — and then `fit_gamma_process.m`
and the pre-onset window of the same file both used the increment count.

The median channel here has 57 increments and 2 segments, so the erroneous
term was ~28× too large and dominated a process term it should have been a
fraction of. It biased τ low in all eight components:

| | 64E | 52C | 75G | 23A | 27F | 41B | 13D | 19H |
|---|---|---|---|---|---|---|---|---|
| before | 0.444 | 0.444 | 0.429 | 0.415 | 0.401 | 0.387 | 0.316 | 0.362 |
| after | 0.475 | 0.491 | 0.475 | 0.459 | 0.444 | 0.415 | 0.387 | 0.415 |
| truth | 0.474 | ← same for all eight |

A systematic bias in all eight, not noise. The README previously attributed
the resulting coverage gap to part-to-part variation the model ignores — but
the simulator draws severity once per (tail, component) and never redraws it,
so there was no such variation. **A plausible explanation for a real symptom
is not the same as the cause**, and the wrong explanation stopped anyone
looking further.

## 10. Editing the numerics did not invalidate a single checkpoint

Stage keys hashed the data inputs and the Python-side params. The `.m` files
were never hashed. Changing the spike threshold in `detect_faults.m` from
`2e-5` to `0.5` — which flags essentially every channel — produced a
byte-identical `flags.csv`, because the stage was served from a checkpoint
keyed on inputs that had not changed. The run reported success and
republished the previous ledger.

The README claimed "a change anywhere upstream invalidates everything
downstream". Code is upstream.

**Fix:** hash every `.m` file into the stage key, and lift stage policy (such
as which fault classes are excluded) into `params` so it is content-addressed
too. Related: `fit1` and `fit2` declared the *same* output filenames, so each
overwrote the other's checkpoint and neither could ever be cached — both
re-ran every time and stayed correct only because the re-run order happened
to leave the right file on disk.

## 11. The test suite could not pass, and the seed hid it

`run_tests.m` seeded `rand` and `randn`. The fleet is generated with `randg`,
which has its own generator state and was never seeded — so the suite failed
about one run in three on an unchanged tree, and CI runs it behind the
README's status badge.

Seeding `randg` made it deterministic, and then it failed *every* run:
`beta within 10% (0.338 vs 0.300)`. Replicating the fit across 15 seeds
showed why. β̂ is essentially unbiased (mean 0.306 against 0.300, 2.0%) but
its sampling SD is 0.023 — **7.8% of the true value**. A ±10% band on a
single draw must fail roughly a quarter of the time no matter how correct
the estimator is.

**Fix:** assert what was actually meant. The suite now averages over
replicate fleets and compares the bias against the Monte Carlo standard error
measured from those same replicates, so the tolerance is derived rather than
chosen and the test fails only for a real bias.

## 12. Four ways a missing value became a confident number

Each of these passed every gate:

- **NaN did not survive the Python→Octave boundary.** `pandas.to_csv` writes
  NaN as an empty field and Octave's `dlmread` reads an empty field as `0`. A
  missing damage state arrived as "brand new, zero damage"; a missing
  log-severity arrived as α = 1, which is 50–250× the fleet nominal.
- **Inside the numerics, `max` and `min` drop NaN.** `max(L - x0, 1e-9)` with
  a NaN `x0` returns `1e-9` — "already at threshold, fails within days". So
  the two ends of the same boundary failed in *opposite* unsafe directions.
- **A NaN damage state was silently immortal in the availability MC.**
  `X >= L` is false for NaN, so the unit never failed and counted as mission
  capable for the whole horizon. `randg(0)` returns NaN in Octave, so
  posterior underflow was a second route in.
- **`usage_hours` had no upper fence and no finiteness check.** One row of
  `inf` or `1e12` at fleet scale moved a component's β from 0.370 to
  1,018,880 — a factor of 2.76 million — turning a component with a coin-flip
  chance of failing inside the horizon into an immortal one (P(fail) 0.4988 →
  0.0001).

And the gates themselves failed open, because `np.nan < 0` and `np.nan > 1`
are both `False`. **A check that fails open is worse than no check, because
it is also a claim.**

**Fix:** `na_rep="NaN"` on every frame crossing the boundary; explicit
finiteness guards at the top of the RUL and availability numerics rather than
clamps; a physical fence and a finiteness check on usage; and `np.isfinite`
first in every gate. The RUL gate distinguishes `+Inf` — which is a
documented "does not reach threshold inside the horizon" — from NaN, which
never means anything.

## 13. Sixteen-node quadrature was not converged

The posterior over log-severity is integrated out by Gauss–Hermite. Against a
4001-point reference the 16-node result was off by up to **0.028** in
per-unit P(fail) — larger than the calibration error of the whole model — and
convergence only sets in near 64 nodes. The worst cases were the channels
with the widest posterior, where the integrand is a sharp sigmoid in the
quadrature variable.

The fleet *total* was unaffected because the errors cancel, which is why it
went unnoticed. The per-channel number is the one an operator acts on.

---

The entries below came from a third adversarial review — this one of the
repository's evidence chain rather than its code.

## 14. The verification scored the fit the product does not ship, and could not fail

Three defects compounded. `verify.py` section 1 printed "(fit2, flagged
channels excluded)" while reading `fit_units.csv` and `fit_comp.json` —
**fit1's** outputs, with every faulted channel included (its own printed
n = 12,000 said so, since fit2 has ~11,788 rows). The headline recovery
table and the severity coverage were therefore measured on a fit nothing
downstream consumes. Second, `verify.py` contained no assertion, raise, or
exit anywhere — despite this file's own section 8 claiming the
caught-vs-missed comparison was "asserted permanently" — so CI could fail
only on a crash, never on a statistical regression. Third, two of the three
committed results artifacts (`summary_example.md`, `ledger_examples.md`)
were stale hand-copies from a superseded run, contradicting the third on
the same page.

**Fix:** sections 1–2 read the fit2 files and print the severity coverage
split by true sensor class (which also resolved the "unexplained" coverage
gap — clean channels sit at 0.873, and the faulted classes' apparent
miscoverage is a scoring artifact, since truth stores pre-fault severity);
`verify.py` now carries 26 checks with derived tolerances and exits
non-zero; CI regenerates the verification and compares it numerically
against the committed artifact; and the example artifacts are written by
`run.py --publish-examples` instead of by hand.

## 15. "Nominal 0.900" was a conditioning error

RUL interval coverage is measured on the units that failed inside the
window, but the intervals are unconditional. Conditioning on failure
truncates the interval's upper tail, so a PERFECTLY calibrated model does
not score 0.90 on this metric: per unit the attainable coverage is
(F−.05)/F for .05 < F < .95 and .90/F above, which — averaged over the
units that actually failed — gives a benchmark of **0.885** on this fleet.
The old artifact printed "0.848 (nominal 0.900)", overstating the shortfall
by 2.4× against a nominal the metric could never reach. The benchmark is
now computed in `verify.py` and the shortfall (0.037) is gated at 0.06.

## 16. The estimator was told a hidden constant, and the tests could not run on MATLAB

`availability_mc.m` drew monthly usage with a hardcoded volatility of
0.15 — the generator's hidden truth constant, copied into the estimator.
Nothing the pipeline is asked to infer may be typed in from the
simulator's source; the volatility is now estimated from each tail's own
monthly usage history and carried through `usage.csv`.

The same file (and the test suite) also called `randg('seed', ...)` —
Octave-only API, absent from base MATLAB — under a "runs on MATLAB
unchanged" claim. All gamma draws now go through `gamma_sample.m`
(Marsaglia–Tsang over randn/rand), which is portable and, more
importantly, leaves rand/randn as the only generator states in play — the
failure mode section 13 of this file documents (randg's separate state
running unseeded) is now structurally impossible.
