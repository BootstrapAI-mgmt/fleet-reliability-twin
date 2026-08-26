"""Score the pipeline against hidden truth.  Only this file reads truth.json.

This is a gate, not a printout: section 8 asserts the properties the README
claims -- reconciliation, no NaN, model beats base rate, detection does not
degrade the forecast, calibration within stated floors -- and exits non-zero
if any fails, so CI can fail on a statistical regression rather than only on
a crash. CI additionally compares this file's numeric output against the
committed artifact (tools/compare_verification.py).

Reports (to results/verification_output.txt):
  0. Run identity (platform, versions, data digests; excluded from comparison)
  1. Fleet parameter recovery (beta, sigma, alpha nominal, tau) per component
     -- scored on fit2, the fit the product ships (reading-corrupting
     channels excluded), which is the fit build_state consumes
  2. Per-unit severity posterior: 90% credible-interval coverage, split by
     the channel's true sensor-fault class
  3. Threshold estimate bias
  4. RUL: 90% interval coverage on units that failed in the forecast window,
     against the CONDITIONAL coverage benchmark (conditioning on failure
     truncates the interval's upper tail, so a perfectly calibrated model
     scores below 0.90 here -- the benchmark says what it would score),
     calibration of P(fail within H) by decile, Brier score
  5. Expected first failures by month vs actual, with Poisson-binomial band
  6. Sensor fault detection: confusion matrix, false-alarm rate, onset
     recovery, power by process shape, watch-list yield, and the
     caught-vs-missed forecast comparison
  7. Ingest: detection per corruption class, row-matched to the injected
     records wherever the corrupted row's identity survives (a reason-code
     count capped at the injected total cannot see false positives; this can)
  8. Checks (any FAIL exits non-zero)
"""
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd

work, data, out = Path("work"), Path("data"), Path("results")
out.mkdir(exist_ok=True)
tr = json.load(open(data / "truth.json")); comps = list(tr["components"]); K = len(comps)
H = tr["forecast_months"]; n_months = tr["n_months"]
L, CHK = [], []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)
def check(name, ok, detail):
    CHK.append((name, bool(ok), detail))

chan_of = lambda key: (int(key[1:5])) * K + comps.index(key[6:]) + 1

def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# 0 ---------------------------------------------------------------
p("== 0. Run identity (excluded from drift comparison) ==")
p(f"  platform {platform.platform()}")
p(f"  python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
p(f"  data digests: inspections {_digest(data / 'inspections.csv')}  truth {_digest(data / 'truth.json')}")
p(f"  fleet {tr.get('n_tails', '?')} tails x {K} components x {n_months} months; horizon {H} mo; pipeline seed 11")

# 1 ---------------------------------------------------------------
p("\n== 1. Fleet parameter recovery (fit2 -- the shipped fit; reading-corrupting channels excluded) ==")
fc = json.load(open(work / "fit2_comp.json"))
rows = []
for k, c in enumerate(comps):
    t = tr["components"][c]
    rows.append(dict(comp=c, shape_per_month=round(t["alpha_nominal"] * 45, 2),
                     beta=f"{fc[k]['beta']:.3f}/{t['beta']:.3f}", sigma=f"{fc[k]['sigma']:.3f}/{t['sensor_sd']:.3f}",
                     alpha_nom=f"{np.exp(fc[k]['mu']):.4f}/{t['alpha_nominal']:.4f}", tau=f"{fc[k]['tau']:.3f}/0.474"))
    check(f"fit2 {c}: beta finite and within 30% of truth",
          np.isfinite(fc[k]["beta"]) and abs(np.log(fc[k]["beta"] / t["beta"])) <= np.log(1.3),
          f"{fc[k]['beta']:.4f} vs {t['beta']:.4f} (sanity bound, not a calibration claim)")
p(pd.DataFrame(rows).to_string(index=False))

# 2 ---------------------------------------------------------------
U = np.loadtxt(work / "fit2_units.csv", delimiter=",")
truth_a = {chan_of(k): v for k, v in tr["alpha_per_unit"].items()}
faults = {chan_of(k): v for k, v in tr["sensor_faults"].items()}
ta = np.array([truth_a[int(c)] for c in U[:, 0]])
z = (np.log(ta) - U[:, 4]) / np.sqrt(U[:, 5])
inside = np.abs(z) < 1.645
tclass = np.array([faults.get(int(c), {}).get("cls", "none") for c in U[:, 0]])
p(f"\n== 2. Severity posterior (fit2): 90% CI coverage = {np.mean(inside):.3f} (nominal 0.900), "
  f"median shrinkage {np.median(U[:, 6]):.2f}, n = {len(U)}")
p("  by true sensor class (truth stores the PRE-FAULT severity, so an accelerated channel --")
p("  real 2x damage, correctly fitted at 2x -- is scored against the rate it no longer wears at;")
p("  its miscoverage below is a property of the scoring, not of the fit):")
for cl in sorted(set(tclass)):
    m = tclass == cl
    p(f"    {cl:12s} n={int(m.sum()):5d}  coverage {np.mean(inside[m]):.3f}")
clean_cov = float(np.mean(inside[tclass == "none"]))
check("severity coverage on clean channels in [0.85, 0.95]",
      0.85 <= clean_cov <= 0.95,
      f"{clean_cov:.3f} (sanity band around nominal 0.90; the residual gap is a disclosed limitation)")

# 3 ---------------------------------------------------------------
st = json.load(open(work / "state_ledger.json"))
p("\n== 3. Threshold estimates (est/true) ==")
p("  " + "  ".join(f"{c}:{st['thresholds'][c]:.2f}/{tr['components'][c]['threshold']:.1f}" for c in comps))

# 4 ---------------------------------------------------------------
rul = pd.read_csv(work / "rul_out.csv", header=None)
rul.columns = ["chan", "p05", "p50", "p95", "pfail"] + [f"F{m}" for m in range(1, H + 1)]
mtf = {chan_of(k): v for k, v in tr["months_to_failure_after_horizon"].items()}
rul["ttf"] = rul["chan"].astype(int).map(mtf)
failed = rul[rul.ttf > 0]
cov = np.mean((failed.p05 <= failed.ttf) & (failed.ttf <= failed.p95.replace(np.inf, 1e9)))
surv = rul[rul.ttf < 0]
# Conditioning on failure-within-horizon truncates the interval: a unit with
# P(fail within H) = F sees only the part of its distribution below H, so a
# PERFECTLY calibrated 90% interval covers (F-.05)/F of its failures when
# .05 < F < .95, 0.90/F when F >= .95, and none when F <= .05. The benchmark
# below is that quantity averaged over the units that actually failed --
# quoting "nominal 0.900" here was a conditioning error, not a model error.
def _cond_cov(F):
    if F <= 0.05: return 0.0
    if F >= 0.95: return 0.90 / F
    return (F - 0.05) / F
bench = float(np.mean([_cond_cov(f) for f in failed.pfail]))
se_cov = float(np.sqrt(bench * (1 - bench) / max(len(failed), 1)))
p(f"\n== 4. RUL ==")
p(f"  units failing within {H} mo: {len(failed)}; 90% interval coverage = {cov:.3f}")
p(f"  conditional-coverage benchmark for a perfectly calibrated model: {bench:.3f} "
  f"(se {se_cov:.4f}); shortfall {bench - cov:+.3f}")
p(f"  survivors: {len(surv)}; fraction whose p95 exceeds horizon = {np.mean(surv.p95 > H):.3f}")
y = (rul.ttf > 0).astype(float); pf = rul.pfail
brier = float(np.mean((pf - y) ** 2)); base = float(y.var())
p(f"  Brier score of P(fail within {H}) = {brier:.4f}  (all-at-base-rate would be {base:.4f})")
n_nan = int(rul[["p05", "p50", "p95", "pfail"]].isna().sum().sum())
check("no NaN in RUL output", n_nan == 0, f"{n_nan} NaN values")
check("Brier beats base rate", brier < base, f"{brier:.4f} vs {base:.4f}")
check("RUL coverage within 0.06 of the conditional benchmark", cov >= bench - 0.06,
      f"{cov:.3f} vs benchmark {bench:.3f} (regression floor around the disclosed calibration gap)")
check("survivor p95 beyond horizon for >= 95%", np.mean(surv.p95 > H) >= 0.95,
      f"{np.mean(surv.p95 > H):.3f}")
rul["bin"] = pd.cut(pf, np.linspace(0, 1, 11), include_lowest=True)
cal = rul.groupby("bin", observed=True).agg(n=("pfail", "size"), pred=("pfail", "mean"), obs=("ttf", lambda s: (s > 0).mean()))
p("  calibration by decile of predicted P(fail):")
p(cal.round(3).to_string())
ledger = json.load(open(work / "ledger.json"))["channels"]
basis = pd.Series({int(c): v["damage_basis"] for c, v in ledger.items()})
rul["basis"] = rul["chan"].astype(int).map(basis)
p("  Brier by damage basis:")
p(rul.groupby("basis").apply(lambda g: pd.Series(dict(n=len(g), brier=np.mean((g.pfail - (g.ttf > 0)) ** 2)))).round(4).to_string())

# 5 ---------------------------------------------------------------
F = rul[[f"F{m}" for m in range(1, H + 1)]].values
inc = np.diff(np.hstack([np.zeros((len(F), 1)), F]), axis=1)
exp_m = inc.sum(0); var_m = (inc * (1 - inc)).sum(0)
act = np.array([(rul.ttf == m).sum() for m in range(1, H + 1)])
lo, hi = exp_m - 1.645 * np.sqrt(var_m), exp_m + 1.645 * np.sqrt(var_m)
inside_m = int(((act >= lo) & (act <= hi)).sum())
p(f"\n== 5. First failures by month: expected vs actual (90% band) ==")
p(pd.DataFrame(dict(month=range(1, H + 1), expected=exp_m.round(1), lo=lo.round(1), hi=hi.round(1), actual=act)).to_string(index=False))
p(f"  months inside band: {inside_m}/{H}; total expected {exp_m.sum():.0f} vs actual {act.sum()} "
  f"({100 * (exp_m.sum() / act.sum() - 1):+.1f}%)")
tot_se = float(np.sqrt(var_m.sum()))
check("months inside 90% band >= 13/18", inside_m >= 13, f"{inside_m}/{H}")
check("total failures within 3.5 se (Poisson-binomial)", abs(exp_m.sum() - act.sum()) <= 3.5 * tot_se,
      f"|{exp_m.sum():.0f}-{act.sum()}| vs {3.5 * tot_se:.0f}")

# 6 ---------------------------------------------------------------
f = np.loadtxt(work / "flags.csv", delimiter=",")
cls = {0: "none", 1: "bias_step", 2: "scale_error", 3: "stuck", 4: "dropout", 5: "accelerated"}
truth = faults
df = pd.DataFrame(dict(chan=f[:, 0].astype(int), pred=[cls[int(x)] for x in f[:, 3]], onset_est=f[:, 4], watch=f[:, 8] > 0))
df["true"] = df.chan.map(lambda c: truth.get(c, {}).get("cls", "none"))
df["onset"] = df.chan.map(lambda c: truth.get(c, {}).get("onset", np.nan))
df["comp"] = [comps[(c - 1) % K] for c in df.chan]
p("\n== 6. Sensor fault detection / isolation (retrospective, whole-history scan) ==")
p(pd.crosstab(df["true"], df["pred"]).to_string())
clean = df[df.true == "none"]
fa = float(np.mean(clean.pred != "none"))
p(f"  false-alarm rate on clean channels: {fa:.4f} ({(clean.pred != 'none').sum()}/{len(clean)})")
check("false-alarm rate <= 1%", fa <= 0.01, f"{fa:.4f}")
det = df[(df.true != "none") & (df.pred != "none")]
p("  onset recovery error (months, estimated minus true; this scan is retrospective,")
p("  so negative values are possible and this is not a real-time detection delay):")
p((det.onset_est - det.onset).groupby(det["true"]).describe()[["count", "mean", "50%", "max"]].round(1).to_string())
p("  power vs process shape (flagged with any class / true), persistent faults:")
shape = {c: round(tr["components"][c]["alpha_nominal"] * 45, 2) for c in comps}
for cl in ["accelerated", "scale_error"]:
    g = df[df.true == cl].groupby("comp").apply(lambda s: f"{(s.pred != 'none').sum()}/{len(s)}")
    p(f"    {cl:12s} " + "  ".join(f"{c}({shape[c]}):{g.get(c, '0/0')}" for c in comps))
w = df[(df.pred == "none") & df.watch]
p(f"  watch list: {len(w)} channels, of which truly faulted {np.sum(w.true != 'none')} "
  f"({np.mean(w.true != 'none'):.2f} precision vs base rate {np.mean(df.true != 'none'):.3f})")

# --- detecting a fault must not make the forecast WORSE -------------
#
# It used to. Every flagged class was dropped from the refit, so a
# genuinely accelerating unit -- real damage, not a sensor fault -- had
# its severity replaced by the fleet average. Scored against truth, the
# units the detector CAUGHT came out worse than the ones it missed:
# catching a fault actively degraded the forecast. This comparison is now
# asserted in section 8 for the classes whose readings stay usable
# (accelerated, dropout); the reading-corrupting classes fall back to a
# usage model whose cost is a disclosed limitation, so they are reported
# here but not asserted.
p("  detection must not degrade the forecast:")
_b = rul.set_index(rul["chan"].astype(int))
briers = {}
for cl in ["accelerated", "dropout", "bias_step", "scale_error", "stuck"]:
    sub = df[df.true == cl]
    if sub.empty:
        continue
    rows = []; briers[cl] = {}
    for caught_flag, label in [(True, "caught"), (False, "missed")]:
        ch = sub[(sub.pred != "none") == caught_flag]["chan"].astype(int)
        g = _b.reindex(ch).dropna(subset=["pfail"])
        if len(g) == 0:
            rows.append(f"{label} n=0")
        else:
            br = float(np.mean((g.pfail - (g.ttf > 0)) ** 2))
            briers[cl][label] = (br, len(g))
            rows.append(f"{label} n={len(g):3d} brier={br:.4f}")
    p(f"    {cl:12s} " + " | ".join(rows))
p(f"    (base-rate Brier = {float(np.mean((rul.ttf > 0)) * (1 - np.mean(rul.ttf > 0))):.4f})")
for cl in ("accelerated", "dropout"):
    b = briers.get(cl, {})
    if "caught" in b and "missed" in b and min(b["caught"][1], b["missed"][1]) >= 5:
        check(f"detection does not degrade the forecast ({cl})", b["caught"][0] <= b["missed"][0],
              f"caught {b['caught'][0]:.4f} (n={b['caught'][1]}) vs missed {b['missed'][0]:.4f} (n={b['missed'][1]})")

# 7 ---------------------------------------------------------------
q = pd.read_csv(work / "quarantine.csv")
ing = json.loads((work / "ingest_summary.json").read_text())
dl = pd.DataFrame(tr["dirty_records"])
p("\n== 7. Ingest: corrupted records caught, by corruption class ==")
check("ingest reconciles", ing["rows_clean"] + ing["rows_quarantined"] == ing["rows_in"],
      f"{ing['rows_clean']}+{ing['rows_quarantined']} vs {ing['rows_in']}")
REASON = dict(bad_tail="UNKNOWN_TAIL", bad_component="UNKNOWN_COMP", month_oob="MONTH_RANGE",
              usage_neg="USAGE_NEGATIVE", neg_reading="READING_NEGATIVE", dup="DUPLICATE",
              units_x10="READING_FENCE|POINT_OUTLIER")
# Classes that leave the row's (tail, component, month) identity intact are
# matched ROW BY ROW against the injected records; a reason-code count capped
# at the injected total (the previous metric) is blind to false positives and
# to catching the wrong rows. Identity-destroying classes (bad_tail,
# bad_component, month_oob) can only be counted by reason code -- their raw
# counts are shown uncapped so an over-count would be visible.
IDENTITY_KEPT = {"usage_neg", "neg_reading", "dup", "units_x10"}
def _keys(frame):
    return {(str(t), str(c), int(m)) for t, c, m in
            zip(frame["tail"], frame["component"], pd.to_numeric(frame["month"], errors="coerce").fillna(-1))}
qk = {}
for reason_pat in set(REASON.values()):
    m = q[q.reason.str.contains(reason_pat)]
    qk[reason_pat] = _keys(m)
for kind, g in dl.groupby("kind"):
    pat = REASON[kind]
    if kind in IDENTITY_KEPT:
        n = len(_keys(g) & qk[pat])
        p(f"  {kind:14s} {n:5d}/{len(g):<5d} {n / len(g):.2f}  (row-matched)")
        floor = 0.80 if kind == "units_x10" else 0.95
        check(f"ingest recall {kind} >= {floor:.2f}", n / len(g) >= floor, f"{n}/{len(g)}")
    else:
        n = int(q.reason.str.contains(pat).sum())
        p(f"  {kind:14s} {n:5d}/{len(g):<5d} {n / len(g):.2f}  (by reason code, uncapped"
          + ("; over-count would indicate false positives)" if n > len(g) else ")"))
        check(f"ingest recall {kind} >= 0.95", n >= 0.95 * len(g), f"{n}/{len(g)}")
p("  units_x10 below the physical fence and not a point outlier (small readings x10) are "
  "undetectable at ingest by design; they surface downstream as spikes or are absorbed as noise.")

# 8 ---------------------------------------------------------------
p("\n== 8. Checks (derived tolerances and stated floors; any FAIL exits non-zero) ==")
n_fail = 0
for name, ok, detail in CHK:
    p(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    n_fail += (not ok)
p(f"  {len(CHK) - n_fail} of {len(CHK)} checks passed")
(out / "verification_output.txt").write_text("\n".join(L) + "\n")
print(f"\nwritten {out / 'verification_output.txt'}")
if n_fail:
    sys.exit(1)
