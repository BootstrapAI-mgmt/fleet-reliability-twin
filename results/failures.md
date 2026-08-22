# Documented failures

Things that were wrong during development, how each was found, and what the
fix was. Each one was found by checking a number against truth, not by the
code raising an error — which is the argument for having truth to check
against.

## 1. Tail trimming biased β low by 2×

The first variance regression trimmed the top 2% of squared residuals "for
robustness". Monthly gamma increments at shape 0.27 carry most of their
variance in the tail; trimming it cut the estimated β to half its true
value, and the per-unit α doubled to compensate (the rate αβ is estimated
directly, so the product was right and each factor was wrong). Sensor σ was
recovered almost perfectly, which made the result look healthy.
**Fix:** no trimming of increments. Robustness to faulted sensors comes from
rejecting whole *units* whose variance statistic is outlying, and from the
second fit pass that excludes flagged channels.

## 2. Increment-level OLS collapsed σ to its floor

Removing the trim made the untrimmed increment-level regression unstable:
a handful of extreme squared residuals from scale-error channels pushed β
high and the noise intercept negative. **Fix:** aggregate to the unit level
(sums over ~60 increments are far less skewed) and regress the unit
variance statistic on `[rate·Σdu, n]`, with iterative whole-unit rejection.

## 3. Testing a rate change against the channel's own fitted rate made long accelerations invisible

The first change-point scan compared post-onset increments with the
channel's fitted α. Because α is pooled over the channel's whole history, a
long-running acceleration was absorbed into the channel's own α and the
residuals looked normal — *longer* faults were *less* detectable, which is
how it was noticed. **Fix:** estimate α from the pre-onset window only
(blended with the fleet prior) and test the post-onset sum against that.

## 4. Flagged channels disappeared from the forecast

The second fit pass excludes flagged channels, so they have no posterior
row, and `build_state` skipped them. 338 components with the *worst*
sensors got no RUL and nothing said so. The only symptom was that section 3
of the verification listed a single damage basis. **Fix:** a channel with no
posterior is forecast from the fleet prior (shrinkage 1.0), labelled
"channel excluded from fleet fit; severity is the fleet prior".

## 5. Threshold estimated 2–4% high

L was taken as the 97th percentile of each failed serial's maximum reading.
Those maxima are noisy readings clustered just under L, so the quantile sits
≈ z(0.97)·σ above L, and every RUL was slightly optimistic (cumulative
failures under-predicted, z ≈ +10). **Fix:** subtract the noise quantile.

## 6. The spike test ignored sensor noise

The single-increment p-value used the gamma tail of the process alone. On
the smooth, low-β components the noise variance exceeds the process variance
per increment and the test produced >100 false bias-step flags. **Fix:**
moment-matched gamma with variance αβ²du + 2σ².

## 7. The provenance audit rejected the narrative

Not a numerics failure — the audit refused "90% band" because the interval
level had not been registered as a fact. The fix was to register it, not to
relax the audit.

## Things that are limits, not bugs

- A bias step smaller than ~3 increment standard deviations cannot be seen
  in the increments; it shows up at the next part install as a non-zero
  offset. Detection latency equals time-to-next-install.
- Power against a 2× rate acceleration falls with process erraticness and
  with a short pre-onset baseline, and is low when the accelerated rate is
  still below fleet-typical. Reported per component in the verification.
