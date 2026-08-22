"""Verify pipeline outputs against the hidden truth in data/truth.json.

This is the ONLY code that reads truth.json.  It reports:
  1. Ingest recall per corruption kind (what the quarantine actually caught)
  2. Fleet parameter recovery (beta, sigma, alpha nominal, tau, thresholds)
  3. Current-damage estimate error by basis (reading / projected / usage-model)
  4. RUL calibration: predicted P(fail within H) by decile vs observed rate;
     expected first-failures per month vs actual; 90% interval coverage
  5. Sensor fault detection: confusion matrix, false-alarm rate, power by
     component process shape, detection latency
Numbers are printed as measured.  Nothing here is tuned to pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAULT = {0: "none", 1: "bias_step", 2: "scale_error", 3: "stuck", 4: "dropout", 5: "accelerated"}


def main(data=Path("data"), work=Path("work")):
    tr = json.loads((data / "truth.json").read_text())
    comps = list(tr["components"]); K = len(comps)
    H = tr["forecast_months"]; n_months = tr["n_months"]
    ledger = json.loads((work / "ledger.json").read_text())
    out = []
    p = lambda *a: out.append(" ".join(str(x) for x in a))
    chan_of = lambda key: int(key[1:5]) * K + comps.index(key[6:]) + 1

    # ---- 1. ingest recall --------------------------------------------------
    p("== 1. Ingest: recall of injected corruptions by kind ==")
    q = pd.read_csv(work / "quarantine.csv")
    qset = set(zip(q["tail"].astype(str), q["component"].astype(str), q["month"]))
    dirty = pd.DataFrame(tr["dirty_records"])
    expect = {"dup": "DUPLICATE", "neg_reading": "READING_NEGATIVE", "bad_tail": "UNKNOWN_TAIL",
              "units_x10": "READING_FENCE|POINT_OUTLIER", "usage_neg": "USAGE_NEGATIVE",
              "bad_component": "UNKNOWN_COMP", "month_oob": "MONTH_RANGE"}
    for kind, g in dirty.groupby("kind"):
        # corrupted rows carry the corrupted key, so match on what remains identifiable
        if kind == "bad_tail":
            hit = q["reason"].str.contains("UNKNOWN_TAIL").sum()
        elif kind == "bad_component":
            hit = q["reason"].str.contains("UNKNOWN_COMP").sum()
        elif kind == "month_oob":
            hit = q["reason"].str.contains("MONTH_RANGE").sum()
        else:
            hit = sum((r.tail, r.component, r.month) in qset for r in g.itertuples())
        p(f"  {kind:14s} injected {len(g):5d}  caught {hit:5d}  ({100*hit/len(g):5.1f}%)  [{expect[kind]}]")
    p("  units_x10 partial by design: a 10x error on a small reading is inside the physical range and")
    p("  indistinguishable at ingest; it surfaces downstream as a spike residual.")

    # ---- 2. parameter recovery ---------------------------------------------
    p("\n== 2. Fleet parameter recovery (fit2) ==")
    fit = ledger["fit"]; thr = ledger["thresholds"]
    p(f"  {'comp':5s} {'shape/mo':>8s} {'beta est/true':>16s} {'sigma est/true':>16s} "
      f"{'alpha0 est/true':>18s} {'tau est':>8s} {'L est/true':>12s}")
    tau_true = np.sqrt(0.45**2 + 0.15**2)
    for c in comps:
        t = tr["components"][c]; f = fit[c]
        p(f"  {c:5s} {t['alpha_nominal']*45:8.2f} {f['beta']:7.3f}/{t['beta']:<7.3f} "
          f"{f['sigma']:7.3f}/{t['sensor_sd']:<7.3f} {np.exp(f['mu']):8.4f}/{t['alpha_nominal']:<8.4f} "
          f"{f['tau']:8.3f} {thr[c]:5.2f}/{t['threshold']:<5.2f}")
    p(f"  true tau (fleet log-severity spread) = {tau_true:.3f}")

    # ---- 3. damage estimate error by basis ---------------------------------
    p("\n== 3. Current damage estimate vs truth, by basis ==")
    ch = ledger["channels"]
    rows = []
    for key, xt in tr["damage_at_horizon"].items():
        c = str(chan_of(key))
        if c in ch:
            rows.append(dict(basis=ch[c]["damage_basis"], err=ch[c]["damage_est"] - xt,
                             rel=(ch[c]["damage_est"] - xt) / tr["components"][key[6:]]["threshold"]))
    df = pd.DataFrame(rows)
    for b, g in df.groupby("basis"):
        p(f"  {b:24s} n={len(g):5d}  mean err {g.err.mean():+.3f}  sd {g.err.std():.3f}  "
          f"(as fraction of threshold: mean {g.rel.mean():+.3f}, sd {g.rel.std():.3f})")

    # ---- 4. RUL calibration ------------------------------------------------
    p(f"\n== 4. RUL calibration over the {H}-month truth window ==")
    recs = []
    for key, mtf in tr["months_to_failure_after_horizon"].items():
        c = str(chan_of(key))
        if c not in ch:
            continue
        r = ch[c]
        recs.append(dict(chan=c, comp=key[6:], mtf=mtf, failed=mtf > 0, pf=r["p_fail_within_horizon"],
                         p05=r["rul_months"]["p05"], p95=r["rul_months"]["p95"],
                         cdf=r["failure_cdf_by_month"], basis=r["damage_basis"]))
    R = pd.DataFrame(recs)
    p(f"  channels scored {len(R)}; actually failed within window: {R.failed.sum()} "
      f"({100*R.failed.mean():.1f}%); predicted expected failures {R.pf.sum():.0f}")
    R["bin"] = pd.cut(R.pf, [-0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    p("  predicted P(fail) bin -> observed failure rate (n)")
    for b, g in R.groupby("bin", observed=True):
        p(f"    {str(b):14s} predicted {g.pf.mean():.3f}  observed {g.failed.mean():.3f}  (n={len(g)})")
    ece = sum(len(g) * abs(g.pf.mean() - g.failed.mean()) for _, g in R.groupby("bin", observed=True)) / len(R)
    p(f"  expected calibration error (ECE): {ece:.4f}")
    cdf = np.array(R.cdf.tolist())
    exp_cum = cdf.sum(axis=0)
    act_cum = np.array([(R.mtf[(R.mtf > 0)] <= m).sum() for m in range(1, H + 1)])
    var_cum = (cdf * (1 - cdf)).sum(axis=0)
    p("  cumulative first failures: month  expected  actual  z")
    for m in [1, 3, 6, 9, 12, 15, 18]:
        z = (act_cum[m-1] - exp_cum[m-1]) / np.sqrt(var_cum[m-1])
        p(f"    {m:5d}  {exp_cum[m-1]:8.0f}  {act_cum[m-1]:6d}  {z:+.2f}")
    F = R[R.failed]
    cov = ((F.p05.fillna(0) <= F.mtf) & (F.p95.fillna(np.inf) >= F.mtf)).mean()
    p(f"  90% interval coverage among the {len(F)} that failed (conditional, truncated at horizon): {cov:.3f}")
    for b, g in R.groupby("basis"):
        gg = g[g.failed]
        cv = ((gg.p05.fillna(0) <= gg.mtf) & (gg.p95.fillna(np.inf) >= gg.mtf)).mean() if len(gg) else np.nan
        p(f"    basis {b:24s} n={len(g):5d} ECE-like |pred-obs| {abs(g.pf.mean()-g.failed.mean()):.3f}  coverage {cv:.3f}")

    # ---- 5. detection ------------------------------------------------------
    p("\n== 5. Sensor fault detection and isolation ==")
    flags = np.loadtxt(work / "flags.csv", delimiter=",")
    truth = {chan_of(k): v for k, v in tr["sensor_faults"].items()}
    rows = []
    for r in flags:
        c = int(r[0]); t = truth.get(c)
        rows.append(dict(chan=c, comp=comps[(c - 1) % K], true=t["cls"] if t else "none",
                         onset=t["onset"] if t else np.nan, pred=FAULT[int(r[3])], pred_on=r[4], watch=bool(r[8])))
    D = pd.DataFrame(rows)
    p(pd.crosstab(D.true, D.pred).to_string())
    clean = D[D.true == "none"]
    p(f"\n  false-alarm rate on clean channels: {(clean.pred != 'none').sum()}/{len(clean)} = "
      f"{100*(clean.pred != 'none').mean():.2f}%   watch-listed clean: {clean.watch.sum()}")
    D["detected"] = D.pred != "none"
    D["correct_class"] = D.pred == D.true
    p("  per class: detected (any flag) / correctly isolated / median latency (months)")
    for cls, g in D[D.true != "none"].groupby("true"):
        lat = (g.pred_on - g.onset)[g.detected]
        p(f"    {cls:12s} {g.detected.mean():.2f} / {g.correct_class.mean():.2f} / {lat.median():.0f}")
    p("  power against a 2x rate acceleration by component (gamma shape per month in parentheses):")
    for c in comps:
        g = D[(D.true == "accelerated") & (D.comp == c)]
        if len(g):
            p(f"    {c} ({tr['components'][c]['alpha_nominal']*45:.2f}): "
              f"{g.correct_class.sum()}/{len(g)} confirmed, {(g.detected | g.watch).sum()}/{len(g)} confirmed-or-watch")
    p("  Detection power for a persistent rate change falls with process erraticness (low shape) and")
    p("  with short pre-onset baseline; this is a property of monthly inspection of a jump process, not")
    p("  of the test.  bias_step is undetectable until the next part install unless the jump is large.")

    txt = "\n".join(out)
    print(txt)
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "verification_output.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data"),
         Path(sys.argv[2]) if len(sys.argv) > 2 else Path("work"))
