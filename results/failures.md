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
(0.44%, including the spike test) is measured in verify.py.

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

The expected-failure curve over-predicts month 1 (433 vs 327 actual) and is
well calibrated thereafter (17/18 months inside the 90% band). Units whose
last reading sits just under the estimated threshold are assigned near-certain
immediate failure, but the threshold estimate is a 97th-percentile lower
bound and the reading carries noise. This is visible, not hidden, and the
fix (a noise-aware posterior on x0) is listed under future work.
