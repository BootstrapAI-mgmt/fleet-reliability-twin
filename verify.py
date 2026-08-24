"""Score the pipeline against hidden truth.  Only this file reads truth.json.

Reports (to results/verification_output.txt):
  1. Fleet parameter recovery (beta, sigma, alpha nominal, tau) per component
  2. Per-unit severity posterior: 90% credible-interval coverage
  3. Threshold estimate bias
  4. RUL: 90% interval coverage on units that failed in the forecast window,
     calibration of P(fail within H) by decile, Brier score
  5. Expected first failures by month vs actual, with Poisson-binomial band
  6. Sensor fault detection: confusion matrix, false-alarm rate, latency,
     power by process shape (accelerated / scale_error), watch-list yield
  7. Ingest: detection rate per corruption class (honest about undetectable ones)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

work, data, out = Path("work"), Path("data"), Path("results")
out.mkdir(exist_ok=True)
tr = json.load(open(data / "truth.json")); comps = list(tr["components"]); K = len(comps)
H = tr["forecast_months"]; n_months = tr["n_months"]
L = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)

chan_of = lambda key: (int(key[1:5])) * K + comps.index(key[6:]) + 1

# 1 ---------------------------------------------------------------
p("== 1. Fleet parameter recovery (fit2, flagged channels excluded) ==")
fc = json.load(open(work / "fit_comp.json"))
rows = []
for k, c in enumerate(comps):
    t = tr["components"][c]
    rows.append(dict(comp=c, shape_per_month=round(t["alpha_nominal"] * 45, 2),
                     beta=f"{fc[k]['beta']:.3f}/{t['beta']:.3f}", sigma=f"{fc[k]['sigma']:.3f}/{t['sensor_sd']:.3f}",
                     alpha_nom=f"{np.exp(fc[k]['mu']):.4f}/{t['alpha_nominal']:.4f}", tau=f"{fc[k]['tau']:.3f}/0.474"))
p(pd.DataFrame(rows).to_string(index=False))

# 2 ---------------------------------------------------------------
U = np.loadtxt(work / "fit_units.csv", delimiter=",")
truth_a = {chan_of(k): v for k, v in tr["alpha_per_unit"].items()}
ta = np.array([truth_a[int(c)] for c in U[:, 0]])
z = (np.log(ta) - U[:, 4]) / np.sqrt(U[:, 5])
p(f"\n== 2. Severity posterior: 90% CI coverage = {np.mean(np.abs(z) < 1.645):.3f} (nominal 0.900), "
  f"median shrinkage {np.median(U[:, 6]):.2f}, n = {len(U)}")

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
p(f"\n== 4. RUL ==")
p(f"  units failing within {H} mo: {len(failed)}; 90% interval coverage = {cov:.3f} (nominal 0.900)")
p(f"  survivors: {len(surv)}; fraction whose p95 exceeds horizon = {np.mean(surv.p95 > H):.3f}")
y = (rul.ttf > 0).astype(float); pf = rul.pfail
p(f"  Brier score of P(fail within {H}) = {np.mean((pf - y) ** 2):.4f}  (all-at-base-rate would be {y.var():.4f})")
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
inside = np.mean((act >= lo) & (act <= hi))
p(f"\n== 5. First failures by month: expected vs actual (90% band) ==")
p(pd.DataFrame(dict(month=range(1, H + 1), expected=exp_m.round(1), lo=lo.round(1), hi=hi.round(1), actual=act)).to_string(index=False))
p(f"  months inside band: {inside:.2f}; total expected {exp_m.sum():.0f} vs actual {act.sum()} "
  f"({100 * (exp_m.sum() / act.sum() - 1):+.1f}%)")

# 6 ---------------------------------------------------------------
f = np.loadtxt(work / "flags.csv", delimiter=",")
cls = {0: "none", 1: "bias_step", 2: "scale_error", 3: "stuck", 4: "dropout", 5: "accelerated"}
truth = {chan_of(k): v for k, v in tr["sensor_faults"].items()}
df = pd.DataFrame(dict(chan=f[:, 0].astype(int), pred=[cls[int(x)] for x in f[:, 3]], onset_est=f[:, 4], watch=f[:, 8] > 0))
df["true"] = df.chan.map(lambda c: truth.get(c, {}).get("cls", "none"))
df["onset"] = df.chan.map(lambda c: truth.get(c, {}).get("onset", np.nan))
df["comp"] = [comps[(c - 1) % K] for c in df.chan]
p("\n== 6. Sensor fault detection / isolation ==")
p(pd.crosstab(df["true"], df["pred"]).to_string())
clean = df[df.true == "none"]
p(f"  false-alarm rate on clean channels: {np.mean(clean.pred != 'none'):.4f} ({(clean.pred != 'none').sum()}/{len(clean)})")
det = df[(df.true != "none") & (df.pred != "none")]
p("  detection latency (months) where detected:")
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
# catching a fault actively degraded the forecast. Nothing in the suite
# would have surfaced that, so the comparison is a permanent check now
# rather than a one-off finding.
p("  detection must not degrade the forecast:")
_b = rul.set_index(rul["chan"].astype(int))
for cl in ["accelerated", "dropout", "bias_step", "scale_error", "stuck"]:
    sub = df[df.true == cl]
    if sub.empty:
        continue
    rows = []
    for caught_flag, label in [(True, "caught"), (False, "missed")]:
        ch = sub[(sub.pred != "none") == caught_flag]["chan"].astype(int)
        g = _b.reindex(ch).dropna(subset=["pfail"])
        if len(g) == 0:
            rows.append(f"{label} n=0")
        else:
            br = float(np.mean((g.pfail - (g.ttf > 0)) ** 2))
            rows.append(f"{label} n={len(g):3d} brier={br:.4f}")
    p(f"    {cl:12s} " + " | ".join(rows))
p(f"    (base-rate Brier = {float(np.mean((rul.ttf > 0)) * (1 - np.mean(rul.ttf > 0))):.4f};"
  f" caught should not be worse than missed)")

# 7 ---------------------------------------------------------------
q = pd.read_csv(work / "quarantine.csv")
dl = pd.DataFrame(tr["dirty_records"])
p("\n== 7. Ingest: corrupted records caught, by corruption class ==")
qkeys = set(zip(q["tail"], q["component"], q["month"]))
# corrupted rows may have had their tail/component/month altered; match on originals where unaltered
caught = {}
for kind, g in dl.groupby("kind"):
    if kind == "bad_tail": n = (q.reason.str.contains("UNKNOWN_TAIL")).sum()
    elif kind == "bad_component": n = (q.reason.str.contains("UNKNOWN_COMP")).sum()
    elif kind == "month_oob": n = (q.reason.str.contains("MONTH_RANGE")).sum()
    elif kind == "usage_neg": n = (q.reason.str.contains("USAGE_NEGATIVE")).sum()
    elif kind == "neg_reading": n = (q.reason.str.contains("READING_NEGATIVE")).sum()
    elif kind == "dup": n = (q.reason.str.contains("DUPLICATE")).sum()
    elif kind == "units_x10": n = (q.reason.str.contains("READING_FENCE|POINT_OUTLIER")).sum()
    caught[kind] = (min(n, len(g)), len(g))
for k, (n, t) in caught.items():
    p(f"  {k:14s} {n:5d}/{t:<5d} {n / t:.2f}")
p("  units_x10 below the physical fence and not a point outlier (small readings x10) are "
  "undetectable at ingest by design; they surface downstream as spikes or are absorbed as noise.")

(out / "verification_output.txt").write_text("\n".join(L) + "\n")
print(f"\nwritten {out / 'verification_output.txt'}")
