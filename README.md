# fleet-reliability-twin

Fleet-scale component deterioration modelling, remaining-useful-life (RUL)
forecasting with calibrated intervals, sequential anomaly detection with
fault isolation, and a Monte Carlo availability forecast. MATLAB numerics
(run and tested on GNU Octave 8, base functions only, no toolboxes) under a
hardened Python orchestrator. Synthetic data with hidden ground truth; clean
provenance — new code, no employer material.

Companion to `sustainment-analytics` (maintenance records → hierarchical
Weibull → METRIC spares) and `depot-flow-twin` (the repair pipeline). This
repo is the *reliability digital twin* piece: what is the condition of each
installed component now, when will it fail, and can the sensor be trusted.

The organizing rule is the same across all three:
**a hardened pipeline is not one that never stops; it is one that never
silently emits a wrong number.**

## What it does

```
719k dirty inspection readings
   │
   ▼ ingest          quarantine with reason codes · reconciliation assert · refusal gate
   ▼ fit (pass 1)    hierarchical gamma-process wear model, partial pooling per channel
   ▼ detect          structured-residual sensor screening: spike / rate change / stuck / dropout / install offset
   ▼ fit (pass 2)    refit with flagged channels excluded
   ▼ state           thresholds from failure history · damage now, with a stated basis per channel
   ▼ rul             closed-form first-passage CDF, posterior marginalised by Gauss-Hermite (no MC)
   ▼ avail           Monte Carlo fleet availability, epistemic + aleatory spread
   ▼ report          evidence ledger per channel · narrative refused if any figure is untraceable
```

Every channel's forecast carries its evidence: where the damage estimate
came from (reading, stale reading projected, or usage-only because the
sensor is flagged), how much of its severity posterior is borrowed from the
fleet, how many increments it rests on, and its sensor status.

## Measured results

From `results/verification_output.txt` — the only code that reads
`truth.json` is `verify.py`. 1,500 tails × 8 components × 60 months, 18-month
truth window, 480 faulted sensor channels, 7,180 corrupted records.

| | |
|---|---|
| Ingest recall | 100% on 6 of 7 corruption kinds; 89% on 10× unit errors (a 10× error on a small reading is inside the physical range and is caught downstream as a spike residual, not at ingest) |
| Fleet parameters | β within 2–15%, sensor σ within 2–20%, nominal α within 10–20%, τ 0.32–0.44 vs 0.474, thresholds within 1% |
| RUL calibration | ECE **0.016** across 10 probability bins, 12,000 channels |
| RUL interval coverage | **0.866** at nominal 0.90 (reading-based channels); **0.52** for sensor-flagged channels forecast from usage alone — and the ledger says so |
| Cumulative failures, 18 mo | predicted 5,963 vs actual 5,808 (+2.6%, concentrated in month 1: reading noise near threshold through a convex CDF) |
| Sensor faults | stuck 96/96 at 0 latency; dropout 91/96; bias step 62/99 (remainder invisible until next install — physics, not the test); false-alarm rate **0.44%** on 11,520 clean channels |
| 2× rate acceleration | power tracks process shape: 6/9 on smooth wear (shape/mo 4.5) → 0/18 on erratic damage (shape/mo 0.2) |

That last row is the most useful finding. A persistent rate change in a
gamma process sampled monthly is detectable only when the process is smooth
enough, the pre-onset baseline is long enough, and the accelerated rate is
high relative to the fleet. The detector reports p-values against the fitted
model; the verification measures the power rather than assuming it. A
watch-list tier (uncorrected p < 0.01) surfaces the weak cases for human
review without confirming them.

## Model

Damage X on each channel (tail × component) follows a usage-scaled gamma
process: increments over usage `du` are Gamma(α·du, β). Failure at threshold
L; inspections read X + N(0, σ²). Per component, (β, σ) are fleet
properties; α is a property of the installation and its environment
(`log α ~ N(μ, τ²)`), pooled across every serial that has occupied the
channel. (β, σ) come from a unit-level moment regression with whole-unit
outlier rejection; (μ, τ) from the marginal likelihood. Each channel's
posterior on log α is the precision-weighted blend of its own data and the
fleet prior, and the shrinkage factor is reported as the evidence
attribution.

RUL uses the gamma process first-passage identity
`P(T ≤ t) = P(X(t) ≥ L − x0) = 1 − P(α u t, (L−x0)/β)` marginalised over the
α posterior by 16-point Gauss–Hermite quadrature. No sampling; 12k units in
~35 s on Octave.

Sensor screening uses five structured residuals (single-increment spike with
a noise-aware gamma tail; growing-window change-point scan tested against the
channel's *pre-onset* rate; exact repeats; trailing coverage; install
offset) and a signature table to isolate bias step / scale error / stuck /
dropout / accelerated degradation. All thresholds are p-values, not tuned
constants.

## Hardening

- Content-addressed stages: a manifest records the sha256 of every input and
  parameter; unchanged stages are skipped, any upstream change invalidates
  everything downstream, a tampered checkpoint is recomputed.
- Atomic publish: outputs are written to a temp dir and renamed into place.
- Transient vs permanent failure: an unavailable interpreter is retried
  with backoff; a numerics error is not (retrying a deterministic error only
  hides it).
- Gates: quarantine fraction, flagged-channel fraction, non-finite fleet
  parameters, non-monotone quantiles, availability outside [0,1] all stop the
  run.
- Degradations propagate: a component that cannot be modelled, a channel
  forecast from the prior only, a stale reading — all recorded and required
  to appear in the narrative.
- Provenance audit: any number in the narrative that is not traceable to a
  fact is a hard error (it caught a literal "90% band" during development).

## Layout

```
simulate/generate_fleet.py   synthetic fleet with hidden truth (only verify.py reads truth.json)
pipeline/ingest.py           quarantine, fences, reconciliation, refusal
pipeline/orchestrator.py     stages, checkpoints, gates, state construction
pipeline/octave_bridge.py    file-based MATLAB/Octave invocation
pipeline/report.py           evidence ledger, provenance-audited narrative
matlab/fit_gamma_process.m   hierarchical fit
matlab/detect_faults.m       screening and isolation
matlab/rul_quantiles.m, rul_cdf.m
matlab/availability_mc.m
matlab/run_stage.m           entry point; reads and writes only the work dir
matlab/tests/run_tests.m     6 numerics tests on known-truth simulations
tests/                       20 pytest tests: ingest, checkpointing, retry, provenance
verify.py                    scores everything against truth
results/                     verification_output.txt, summary_example.md, failures.md
```

## Run

```
pip install -r requirements.txt
python simulate/generate_fleet.py            # ~30 s, writes data/
python run.py                                # ~1 min cold, 5 s warm
python verify.py                             # scores against data/truth.json
python -m pytest tests
octave --no-gui -q --eval "addpath('matlab'); addpath('matlab/tests'); run_tests"
```

Set `MATLAB_CMD=matlab` to run the numerics under MATLAB instead of Octave.

## Known limitations

- Severity is pooled per channel; part-to-part variability (15% log-sd in
  the synthetic truth) is absorbed into the process and slightly widens the
  miss on interval coverage (0.87 vs 0.90).
- The damage estimate for a sensor-flagged channel is usage-since-install,
  which is biased high for parts installed before the record starts. Its
  forecast is worse and labelled as such; a better answer is a manual
  inspection, which is what the ledger recommends.
- Availability MC uses a single turnaround-time distribution and a flat
  spare-fill probability; the repair pipeline itself is modelled in
  `depot-flow-twin`.
- Synthetic data only. The gamma-process model, the noise model, and the
  fault classes are assumptions that real fleet data would need to test.
