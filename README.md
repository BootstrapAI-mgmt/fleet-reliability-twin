# fleet-reliability-twin

[![verification](https://github.com/BootstrapAI-mgmt/fleet-reliability-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/BootstrapAI-mgmt/fleet-reliability-twin/actions/workflows/ci.yml)

Fleet-scale component deterioration modelling, remaining-useful-life (RUL)
forecasting with intervals scored against hidden truth, whole-history anomaly
detection with fault isolation, and Monte Carlo availability — MATLAB numerics
(runs on GNU Octave, no toolboxes; gamma sampling is self-contained so the
same files run under base MATLAB) beneath a hardened Python orchestrator.
Synthetic data with hidden ground truth; clean provenance (new code, no
employer material).

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
over the α posterior by 64-point Gauss–Hermite quadrature (16 nodes were
measurably insufficient — `results/failures.md` §12). No sampling.

Fault isolation runs retrospectively over the full inspection history using
five structured residuals — single-increment spike, growing-window
change-point scan against the pre-onset rate, exact repeats, missing-record
fraction, and non-zero reading at a fresh install — and a signature table
that separates bias step, scale error, stuck sensor, dropout, and genuine
accelerated wear. Where the statistics live and where the judgement lives:
the spike and change-point tests are p-values against the fitted process
model; the spike and install tests additionally require physically large
magnitudes (4σ/5σ floors), and the stuck and dropout rules are structural
(exact repeats; missing fraction) rather than tests. The floors are chosen,
and the defence of a chosen constant is measurement — the achieved
false-alarm rate and power are scored against hidden truth below rather
than assumed.

## Verified results (`results/verification_output.txt`, all against hidden truth)

Fleet: 1,500 tails × 8 components × 60 months; 719k inspection rows; 480
faulted sensor channels; 7.2k corrupted records; 18-month forecast window.

Every number below is scored on **fit2 — the fit the product actually
ships** (reading-corrupting channels excluded); an earlier verification
scored fit1 under a fit2 header (`results/failures.md` §14).

| | |
|---|---|
| Ingest | 7,073 quarantined with reason codes (0.98%); recall 1.00 on 6 of 7 classes — **row-matched to the injected records** for the 4 classes whose corrupted row keeps its identity, by uncapped reason-code count for the 3 that destroy it (an over-count would show as >1.00) — and `units_x10` at 0.88 (undetectable below the physical fence by design) |
| Process scale β | all 8 biased low, within 1–17% of truth; sensor σ within ~2% on the smooth components, up to −20% where weakly identified (disclosed below) |
| Severity posterior | 90% interval covers truth **0.866** overall, and the artifact prints the split: **0.873 on the 11,477 clean channels**; the accelerated/scale rows (0.39–0.44) are scoring artifacts, because truth stores the pre-fault severity the channel no longer wears at |
| Thresholds | seven of eight within 1%, the eighth at 1.07%; all eight biased low |
| RUL interval coverage | **0.848 against a conditional benchmark of 0.885** — conditioning on failure-within-window truncates the interval, so a *perfectly calibrated* model scores 0.885 here, not 0.90; the 0.037 shortfall is gated at 0.06. An earlier version printed "nominal 0.900" for this metric, which was a conditioning error, not a model property |
| P(fail within 18 mo) | Brier **0.096** vs 0.250 base rate; calibration by decile within 0.04 everywhere (the worst bin's +0.039 is ~3.5 SE at n=992 — visible in the table, not smoothed away) |
| First failures by month | 15 of 18 months inside the 90% band; total −0.6% |
| Fault detection | stuck 96/96 by exact-repeat matching (the generator emits exact repeats, so this cell validates isolation logic, not field stuck-sensor power), dropout 91/96, bias 63/99 (rest wait for next install), false alarms **0.43%** |
| Power vs process shape | 2× acceleration: 7/9 at shape 4.5/mo → 0/18 at shape 0.2/mo |

That last row is the most useful finding. Detectability of a rate change from
monthly inspections is governed by the process shape: smooth wear is caught
within a month or two; erratic damage arriving in rare large increments
cannot be resolved per channel inside two years at a 0.1% false-alarm rate,
however the test is built. The honest product response is a tiered output —
hard flags with a stated false-alarm rate, plus a watch list (2.5× the base
rate of true faults) for human review — not a tuned classifier. The "onset
recovery" statistics in the artifact are retrospective estimation errors,
not real-time detection delays — the scan sees the whole history at once,
which is why some of them are negative.

Detecting a fault also has to *improve* the forecast. For accelerated wear
the caught units score a Brier of **0.000** against **0.160** for the ones
the detector missed; for dropout, 0.118 against 0.309 — and `verify.py`
**asserts** both comparisons and exits non-zero if either inverts, because
an earlier version failed exactly that comparison with nothing in place to
notice — see [Known limitations](#known-limitations) and
`results/failures.md`.

Both test suites pass: 15 Octave known-parameter checks, 18 pytest tests of
ingest, checkpointing, retry policy, and provenance audit — plus 26
asserting verification checks with derived tolerances (`verify.py` exits
non-zero on any failure). The Octave suite compares estimator bias over
replicate fleets against the Monte Carlo standard error measured from those
replicates **plus a disclosed 2%-of-truth allowance** for the small-sample
bias the fit is documented to carry; the printed message shows the whole
bar. CI regenerates the full 1,500-tail verification on every push and
compares it numerically against the committed artifact
(`tools/compare_verification.py`).

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
matlab/gamma_sample.m        portable Gamma sampler (randn/rand only; no randg)
matlab/run_stage.m           entry point
matlab/tests/run_tests.m
tools/compare_verification.py  CI's numeric diff of fresh vs committed verification
tests/                       pytest
verify.py                    scoring against truth
results/                     verification_output.txt, failures.md
```

## Run

```
pip install -r requirements.txt          # numpy pandas pytest; octave on PATH (or MATLAB_CMD)
python simulate/generate_fleet.py        # ~1 min
python run.py                            # ~2.5 min on Linux; much longer under Windows Octave
python verify.py
octave --eval "cd matlab/tests; run_tests"
pytest
```

## Known limitations

- Severity is pooled per installation, so a channel's estimate is shared
  across every serial that has occupied it. The severity-coverage shortfall
  is now decomposed in the artifact rather than guessed at: clean channels
  cover at 0.873, and the faulted classes' apparent miscoverage is a
  scoring artifact (truth stores pre-fault severity). The remaining ~0.03
  gap on clean channels is the honest residual — earlier attempts to
  explain it produced two wrong stories in a row (part-to-part variation
  that the simulator does not contain, then a variance-derivation error
  that was real but partial, moving τ to 0.39–0.48 against a true 0.474;
  `results/failures.md`).
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
  whose part is never replaced in-window is missed, and the onset-recovery
  distribution reflects that.
- The availability MC draws severity and usage **independently per unit**,
  but a real tail's components share environment and duty, so tail-down
  events are positively correlated and the MC understates the spread of
  fleet availability. Usage volatility itself is estimated from each
  tail's monthly history (an earlier version hardcoded the generator's
  hidden constant — `results/failures.md` §16). The MC also uses a fixed
  lognormal turnaround and spare-fill probability; the repair side is the
  subject of `depot-flow-twin`.

---

## Companion repositories

One of three self-contained repositories, each taking on a different problem in the same sustainment domain. They share a design stance -- synthetic data with hidden truth, estimates scored against that truth rather than asserted, and a failure log recording what went wrong and why (`results/failures.md` here; `docs/VALIDATION.md` §9 in sustainment-analytics) -- but no code, so each stands alone.

| repo | what it does | stack |
|---|---|---|
| [sustainment-analytics](https://github.com/BootstrapAI-mgmt/sustainment-analytics) | sparse censored failure records to an auditable spares buy list | MATLAB + Python |
| **fleet-reliability-twin** (this one) | gamma-process degradation, sensor-fault isolation, remaining useful life | MATLAB + Python |
| [depot-flow-twin](https://github.com/BootstrapAI-mgmt/depot-flow-twin) | Java discrete-event depot simulation, turnaround estimation and forecasting | Java 21 + Python |
