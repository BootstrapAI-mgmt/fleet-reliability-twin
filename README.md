# fleet-reliability-twin

Fleet-scale component deterioration modelling, remaining-useful-life (RUL)
forecasting with verified intervals, sequential anomaly detection with fault
isolation, and Monte Carlo availability — MATLAB numerics (runs on GNU Octave,
no toolboxes) under a hardened Python orchestrator. Synthetic data with hidden
ground truth; clean provenance (new code, no employer material).

Companion to `sustainment-analytics` (spares demand and allocation) and
`depot-flow-twin` (repair pipeline simulation). The organising rule is the
same across all three: **a hardened pipeline is not one that never stops; it
is one that never silently emits a wrong number.**

```
dirty inspections ─► ingest ─► fit₁ ─► detect ─► fit₂ ─► state ─► RUL ─► availability ─► ledger
                     (quarantine, (gamma   (5 fault (flagged  (thresholds,   (closed   (MC,      (evidence per
                      reconcile,   process, classes, channels  sensor-aware   form,     epistemic  unit, audited
                      refuse)      EB pool) change-   excluded) damage state)  GH quad)  +aleatory) narrative)
                                            point)
```

## Model

Damage `X` on each installed component follows a usage-scaled gamma process,
`dX ~ Gamma(α·du, β)`, observed monthly through a noisy sensor. Failure is
first passage of a threshold `L`. `β` and sensor `σ` are component
properties; severity `α` is a property of the *installation* (tail ×
component — environment and duty), pooled across every serial that has
occupied it, with a lognormal fleet prior estimated empirically. Each unit's
posterior reports how much of it is borrowed from the fleet (the shrinkage
factor), which is the evidence attribution an operator sees with the number.

RUL is the closed-form first-passage CDF of the gamma process, marginalised
over the α posterior by 16-point Gauss–Hermite quadrature. No sampling.

Fault isolation uses five structured residuals — single-increment spike,
growing-window change-point scan against the pre-onset rate, exact repeats,
missing-record fraction, and non-zero reading at a fresh install — and a
signature table that separates bias step, scale error, stuck sensor, dropout,
and genuine accelerated wear. Every threshold is a p-value against the fitted
model; nothing is hand-tuned.

## Verified results (`results/verification_output.txt`, all against hidden truth)

Fleet: 1,500 tails × 8 components × 60 months; 719k inspection rows; 480
faulted sensor channels; 7.2k corrupted records; 18-month forecast window.

| | |
|---|---|
| Ingest | 7,073 quarantined with reason codes (0.98%), 7 of 7 corruption classes at 100% except `units_x10` at 0.88 (undetectable below the physical fence by design) |
| Process scale β | all 8 within 2–15% of truth, and all biased low; sensor σ exact where identifiable |
| Severity posterior | 90% interval covers truth **0.874**; median shrinkage 0.19 |
| Thresholds | seven of eight within 1%, the eighth at 1.07%; all eight biased low |
| RUL interval coverage | **0.848** at nominal 0.90 (5,808 units that failed in-window) |
| P(fail within 18 mo) | Brier **0.096** vs 0.250 base rate; calibration by decile within 0.04 everywhere |
| First failures by month | 15 of 18 months inside the 90% band; total −0.6% |
| Fault detection | stuck 96/96 at 0 latency, dropout 91/96, bias 63/99 (rest wait for next install), false alarms **0.43%** |
| Power vs process shape | 2× acceleration: 6/9 at shape 4.5/mo → 0/18 at shape 0.2/mo |

That last row is the most useful finding. Detectability of a rate change from
monthly inspections is governed by the process shape: smooth wear is caught
within a month or two; erratic damage arriving in rare large increments
cannot be resolved per channel inside two years at a 0.1% false-alarm rate,
however the test is built. The honest product response is a tiered output —
hard flags with a stated false-alarm rate, plus a watch list (2.5× the base
rate of true faults) for human review — not a tuned classifier.

Detecting a fault also has to *improve* the forecast, and `verify.py` checks
that directly against truth. For accelerated wear the caught units now score a
Brier of **0.000** against **0.160** for the ones the detector missed; for
dropout, 0.118 against 0.309. That comparison exists because an earlier
version failed it — see [Known limitations](#known-limitations) and
`results/failures.md`.

Both test suites pass: 15 Octave known-parameter tests, 18 pytest tests of
ingest, checkpointing, retry policy, and provenance audit. The Octave suite
asserts that the estimator is *unbiased* over replicate fleets against a
Monte Carlo standard error measured from those replicates, rather than that a
single draw lands inside a chosen tolerance.

## Hardening

- Content-addressed stages: manifests record input hashes; unchanged inputs
  are skipped, a change anywhere upstream invalidates downstream, and a
  tampered or partial checkpoint is recomputed rather than trusted.
- Atomic publish: stages write to a temp dir and `os.replace` into place.
- Transient vs permanent: interpreter unavailable → retry with backoff;
  numerics error → stop (retrying a deterministic error only hides it).
- Gates: ingest refuses above 5% quarantine; detect refuses above 10%
  flagged; RUL refuses non-monotone quantiles; availability refuses out of
  [0,1]. A declared output that was not produced is a failure.
- Degradations (unmodelled component, stale reading, flagged sensor)
  propagate into every affected unit's evidence and into the narrative.
- The narrative is audited: any figure not traceable to a computed fact, or
  any omitted degradation, is a hard error.

## Layout

```
simulate/generate_fleet.py   synthetic fleet + hidden truth (only verify.py reads truth.json)
pipeline/ingest.py           quarantine, reconciliation, refusal
pipeline/orchestrator.py     stages, checkpoints, gates
pipeline/report.py           evidence ledger, audited narrative
pipeline/octave_bridge.py    file-based MATLAB/Octave invocation
matlab/fit_gamma_process.m   hierarchical fit, EB prior, shrinkage
matlab/detect_faults.m       five residuals, signature isolation
matlab/rul_quantiles.m       closed-form RUL, GH quadrature
matlab/rul_cdf.m
matlab/availability_mc.m     fleet availability / failure counts
matlab/run_stage.m           entry point
matlab/tests/run_tests.m
tests/                       pytest
verify.py                    scoring against truth
results/                     verification_output.txt, failures.md
```

## Run

```
pip install -r requirements.txt          # numpy pandas pytest; octave on PATH (or MATLAB_CMD)
python simulate/generate_fleet.py        # ~1 min
python run.py                            # ~2.5 min end to end
python verify.py
octave --eval "cd matlab/tests; run_tests"
pytest
```

## Known limitations

- Severity is pooled per installation, so a channel's estimate is shared
  across every serial that has occupied it. The residual 0.87 vs 0.90
  posterior coverage is not fully explained; an earlier version of this file
  attributed it to part-to-part variation being ignored, which was wrong —
  the simulator draws severity once per (tail, component) and never redraws
  it, so there is no such variation to ignore. The larger contributor was a
  derivation error in the sampling variance of the channel rate, now fixed
  (`results/failures.md`), which moved τ from 0.32–0.44 to 0.39–0.49 against
  a true 0.474.
- **Channels whose readings are genuinely corrupted forecast worse than a
  constant.** For stuck channels the Brier is 0.33 against a 0.25 base rate,
  and for bias-step and scale-error channels 0.21–0.23. Their damage state
  has to come from a usage model, and that model is confidently wrong rather
  than appropriately uncertain. The honest response is to widen the interval
  for these channels rather than issue a sharp probability; that is not yet
  implemented, and the numbers above are what it costs.
- Sensor σ is weakly identified when `2σ²` is much smaller than the
  per-increment process variance — which is also when it matters least.
- Current damage for RUL is the last reading, not a noise-aware posterior;
  this over-predicts month-1 failures for units near threshold (see
  `results/failures.md` §7).
- Bias steps are only detectable at the next install; a bias on a channel
  whose part is never replaced in-window is missed, and the latency
  distribution reflects that.
- The availability MC uses a fixed lognormal turnaround and spare-fill
  probability; the repair side is the subject of `depot-flow-twin`.
