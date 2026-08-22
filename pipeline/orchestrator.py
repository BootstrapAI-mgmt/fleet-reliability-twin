"""Hardened orchestrator for the fleet reliability twin.

Stages (each a pure function of its inputs on disk):

  ingest   dirty records  -> clean_named.csv, quarantine.csv, ingest_summary.json
  encode   clean_named    -> clean.csv (numeric), config.json
  fit1     clean          -> fit_units.csv, fit_comp.json            [MATLAB]
  detect   clean, fit1    -> flags.csv, detect_calib.json            [MATLAB]
  fit2     clean, flags   -> refit with flagged channels excluded     [MATLAB]
  state    clean, fit2, flags, events -> rul_in.csv, comp_params.csv, ledger seed
  rul      rul_in         -> rul_out.csv                              [MATLAB]
  avail    state          -> avail.json                               [MATLAB]
  report   everything     -> ledger.json, summary.md

Hardening rules
  * Every stage is content-addressed: its manifest records the sha256 of its
    inputs and parameters.  A re-run with unchanged inputs is skipped; a
    change anywhere upstream invalidates everything downstream.
  * Outputs are written to a temp dir and renamed into place atomically, so
    a crash never leaves a half-written checkpoint that a later run trusts.
  * MATLAB stages distinguish transient failure (interpreter unavailable,
    timeout -> retry with backoff) from permanent failure (numerics error ->
    stop; retrying a deterministic error only hides it).
  * Gates stop the run rather than emit an unverified number.
  * Degradations (a component with too few units to model, a stale sensor)
    are recorded and propagate into the ledger and the narrative; the
    narrative is refused if it omits them.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import octave_bridge as ob
from .ingest import IngestRefused, ingest

ROOT = Path(__file__).resolve().parents[1]
FAULT_NAMES = {0: "none", 1: "bias_step", 2: "scale_error", 3: "stuck", 4: "dropout", 5: "accelerated"}


class GateFailure(RuntimeError):
    pass


@dataclass
class RunState:
    degradations: list[str] = field(default_factory=list)
    stage_log: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(inputs: list[Path], params: dict) -> str:
    h = hashlib.sha256()
    for p in inputs:
        h.update(p.name.encode()); h.update(_sha_file(p).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:16]


class Stage:
    """A content-addressed, atomically checkpointed stage."""

    def __init__(self, name: str, work: Path, inputs: list[Path], outputs: list[str],
                 params: dict, state: RunState):
        self.name, self.work, self.inputs, self.outputs = name, work, inputs, outputs
        self.params, self.state = params, state
        self.manifest = work / f"{name}.manifest.json"

    def cached(self) -> bool:
        if not self.manifest.exists():
            return False
        m = json.loads(self.manifest.read_text())
        if m.get("key") != _key(self.inputs, self.params):
            return False
        for o in self.outputs:
            p = self.work / o
            if not p.exists() or _sha_file(p) != m["outputs"].get(o):
                return False  # checkpoint tampered or partial: recompute
        return True

    def run(self, fn, retries: int = 3):
        t0 = time.time()
        if self.cached():
            self.state.stage_log.append(dict(stage=self.name, status="cached", sec=0.0))
            return
        tmp = Path(tempfile.mkdtemp(prefix=f".{self.name}.", dir=self.work))
        try:
            for inp in self.inputs:                  # stage reads only its declared inputs
                os.symlink(inp.resolve(), tmp / inp.name)
            attempt = 0
            while True:
                try:
                    fn(tmp)
                    break
                except ob.NumericsUnavailable as e:  # transient: retry with backoff
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(2 ** attempt)
                    self.state.stage_log.append(dict(stage=self.name, status="retry", reason=str(e)))
            for o in self.outputs:
                if not (tmp / o).exists():
                    raise GateFailure(f"{self.name}: declared output {o} not produced")
            for o in self.outputs:                   # atomic publish
                os.replace(tmp / o, self.work / o)
            man = dict(key=_key(self.inputs, self.params), params=self.params,
                       outputs={o: _sha_file(self.work / o) for o in self.outputs},
                       finished=time.time())
            mtmp = self.manifest.with_suffix(".tmp")
            mtmp.write_text(json.dumps(man, indent=2)); os.replace(mtmp, self.manifest)
            self.state.stage_log.append(dict(stage=self.name, status="ran", sec=round(time.time() - t0, 1)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def run_pipeline(data: Path, work: Path, cfg: dict) -> RunState:
    work.mkdir(parents=True, exist_ok=True)
    st = RunState()
    comps = cfg["components"]; K = len(comps)
    n_months = cfg["n_months"]
    cidx = {c: i + 1 for i, c in enumerate(comps)}

    # ---- ingest ------------------------------------------------------------
    def do_ingest(tmp):
        insp = pd.read_csv(tmp / "inspections.csv"); roster = pd.read_csv(tmp / "roster.csv")
        res = ingest(insp, roster, comps, n_months, cfg["max_quarantine_frac"])
        res.clean.to_csv(tmp / "clean_named.csv", index=False)
        res.quarantine.to_csv(tmp / "quarantine.csv", index=False)
        (tmp / "ingest_summary.json").write_text(json.dumps(res.summary, indent=2))
    Stage("ingest", work, [data / "inspections.csv", data / "roster.csv"],
          ["clean_named.csv", "quarantine.csv", "ingest_summary.json"],
          dict(n_months=n_months, maxq=cfg["max_quarantine_frac"]), st).run(do_ingest)
    st.facts["ingest"] = json.loads((work / "ingest_summary.json").read_text())

    # ---- encode ------------------------------------------------------------
    def do_encode(tmp):
        c = pd.read_csv(tmp / "clean_named.csv")
        c["tail"] = c["tail"].str[1:].astype(int) + 1
        c["component"] = c["component"].map(cidx)
        c[["tail", "component", "serial", "month", "usage_hours", "reading"]].to_csv(tmp / "clean.csv", index=False)
        (tmp / "config.json").write_text(json.dumps(dict(
            K=K, n_months=n_months, forecast_months=cfg["forecast_months"],
            mc_reps=cfg["mc_reps"], seed=cfg["seed"], tat=cfg["tat"])))
    Stage("encode", work, [work / "clean_named.csv"], ["clean.csv", "config.json"],
          dict(K=K, cfg=cfg), st).run(do_encode)

    # ---- fit1 / detect / fit2 --------------------------------------------
    def matlab(stage):
        def f(tmp):
            ob.run_stage(stage, tmp)
        return f
    Stage("fit1", work, [work / "clean.csv", work / "config.json"], ["fit_units.csv", "fit_comp.json"],
          dict(pass_=1), st).run(matlab("fit"))
    gate_fit(work, comps, st, "fit1")

    Stage("detect", work, [work / "clean.csv", work / "config.json", work / "fit_units.csv", work / "fit_comp.json"],
          ["flags.csv", "detect_calib.json"], {}, st).run(matlab("detect"))
    flags = np.loadtxt(work / "flags.csv", delimiter=",")
    flagged_frac = float(np.mean(flags[:, 3] > 0))
    st.facts["detect"] = dict(n_channels=int(len(flags)), n_flagged=int((flags[:, 3] > 0).sum()),
                              flagged_frac=round(flagged_frac, 4), n_watch=int(flags[:, 8].sum()),
                              by_class={FAULT_NAMES[int(k)]: int(v) for k, v in
                                        zip(*np.unique(flags[:, 3], return_counts=True))},
                              calib=json.loads((work / "detect_calib.json").read_text()))
    if flagged_frac > cfg["max_flagged_frac"]:
        raise GateFailure(f"detect: {flagged_frac:.1%} of channels flagged; either the fleet or the "
                          f"detector is broken -- refusing to forecast on it")

    def do_exclude(tmp):
        ex = (flags[:, 3] > 0).astype(int)
        np.savetxt(tmp / "exclude.csv", ex, fmt="%d")
        ob.run_stage("fit", tmp)
    Stage("fit2", work, [work / "clean.csv", work / "config.json", work / "flags.csv"],
          ["fit_units.csv", "fit_comp.json"], dict(pass_=2), st).run(do_exclude)
    gate_fit(work, comps, st, "fit2")

    # ---- state: thresholds, current damage, RUL inputs ---------------------
    def do_state(tmp):
        build_state(tmp, work, data, comps, cfg, st)
    Stage("state", work, [work / "clean.csv", work / "fit_units.csv", work / "fit_comp.json",
                          work / "flags.csv", data / "events.csv"],
          ["rul_in.csv", "comp_params.csv", "usage.csv", "avail_units.csv", "state_ledger.json"],
          dict(stale=cfg["stale_months"]), st).run(do_state)

    Stage("rul", work, [work / "rul_in.csv", work / "config.json"], ["rul_out.csv"], {}, st).run(matlab("rul"))
    rul = np.loadtxt(work / "rul_out.csv", delimiter=",")
    if not (np.all(rul[:, 1] <= rul[:, 2] + 1e-9) and np.all(rul[:, 2] <= rul[:, 3] + 1e-9)
            and np.all((rul[:, 4] >= 0) & (rul[:, 4] <= 1))):
        raise GateFailure("rul: non-monotone quantiles or pfail outside [0,1]")

    Stage("avail", work, [work / "avail_units.csv", work / "comp_params.csv", work / "usage.csv",
                          work / "config.json"], ["avail.json"], {}, st).run(matlab("avail"))
    av = json.loads((work / "avail.json").read_text())
    a = np.array(av["avail"])
    if a.min() < 0 or a.max() > 1:
        raise GateFailure("avail: availability outside [0,1]")
    st.facts["avail"] = av

    # ---- report -----------------------------------------------------------
    from .report import build_ledger, render_summary
    ledger = build_ledger(work, comps, cfg, st)
    (work / "ledger.json").write_text(json.dumps(ledger, indent=1))
    md = render_summary(ledger, st)
    (work / "summary.md").write_text(md)
    (work / "run_log.json").write_text(json.dumps(dict(stages=st.stage_log, degradations=st.degradations), indent=2))
    return st


def gate_fit(work: Path, comps, st: RunState, name: str):
    fc = json.loads((work / "fit_comp.json").read_text())
    for k, c in enumerate(comps):
        b = fc[k]["beta"]
        if b is None or not np.isfinite(b):
            st.degradations.append(f"{name}: component {c} has too few units to model; "
                                   f"no RUL will be issued for it")
        elif fc[k]["sigma"] <= 0.0011:
            st.degradations.append(f"{name}: sensor noise for {c} hit its floor; the "
                                   f"variance split is unreliable")
    st.facts[name] = {c: fc[k] for k, c in enumerate(comps)}


def build_state(tmp: Path, work: Path, data: Path, comps, cfg, st: RunState):
    K = len(comps); n_months = cfg["n_months"]; stale = cfg["stale_months"]
    D = pd.read_csv(work / "clean.csv")
    fc = json.loads((work / "fit_comp.json").read_text())
    U = np.loadtxt(work / "fit_units.csv", delimiter=",")
    flags = np.loadtxt(work / "flags.csv", delimiter=",")
    events = pd.read_csv(data / "events.csv")

    # Threshold per component: the max reading of a serial that later failed is
    # bounded above by L (plus noise); its upper quantile across failed serials
    # estimates L.  Events are the only place the pipeline learns "failed".
    failed = set(events["serial"])
    mx = D[D["serial"].isin(failed)].groupby(["component", "serial"])["reading"].max().reset_index()
    L = {}
    for k in range(1, K + 1):
        v = mx[mx["component"] == k]["reading"].values
        # the quantile is of NOISY readings clustered just under L, so it sits
        # ~z(0.97)*sigma above L; correct for that (results/failures.md #5)
        L[k] = float(np.quantile(v, 0.97) - 1.88 * fc[k - 1]["sigma"]) if len(v) >= 20 else np.nan
        if not np.isfinite(L[k]):
            st.degradations.append(f"state: no failure history for {comps[k-1]}; threshold unknown")
    comp_params = np.array([[fc[k - 1]["beta"], L[k]] for k in range(1, K + 1)], dtype=float)
    pd.DataFrame(comp_params, columns=["beta", "threshold"]).to_csv(tmp / "comp_params.csv", index=False)

    usage = D.groupby("tail")["usage_hours"].mean()
    T = int(D["tail"].max())
    ur = np.array([[t, float(usage.get(t, usage.mean()))] for t in range(1, T + 1)])
    pd.DataFrame(ur, columns=["tail", "usage_per_month"]).to_csv(tmp / "usage.csv", index=False)

    # current state per channel = latest serial's latest reading
    D = D.sort_values(["tail", "component", "month"])
    last = D.groupby(["tail", "component"]).tail(1).set_index(["tail", "component"])
    first_of_serial = D.groupby("serial")["month"].min()
    unit_by_chan = {int(r[0]): r for r in U}
    flag_by_chan = {int(r[0]): r for r in flags}
    rows, avail_rows, ledger = [], [], {}
    for (t, k), r in last.iterrows():
        chan = (t - 1) * K + k
        u = unit_by_chan.get(chan); f = flag_by_chan.get(chan)
        beta, Lk = fc[k - 1]["beta"], L[k]
        if not np.isfinite(beta) or not np.isfinite(Lk):
            continue                      # recorded as a degradation in gate_fit / above
        prior_only = False
        if u is None:
            # channel was excluded from the fleet fit (flagged sensor).  It must
            # still get a forecast -- from the fleet prior, labelled as such.
            # Dropping it silently was failure #4 in results/failures.md.
            la_mu, la_var, shrink, n_incr = fc[k - 1]["mu"], fc[k - 1]["tau"] ** 2, 1.0, 0
            prior_only = True
        else:
            la_mu, la_var, shrink, n_incr = u[4], u[5], u[6], int(u[7])
        alpha = float(np.exp(la_mu + la_var / 2))
        cls = FAULT_NAMES[int(f[3])] if f is not None else "none"
        age_months = n_months - 1 - int(r["month"])
        install = int(first_of_serial[r["serial"]])
        notes = []
        if prior_only:
            notes.append("channel excluded from fleet fit; severity is the fleet prior (100% borrowed)")
        x_read = float(r["reading"])
        if cls in ("bias_step", "scale_error", "stuck"):
            # sensor cannot be trusted: damage from usage-only model since install
            x0 = alpha * beta * ur[t - 1, 1] * (n_months - install)
            notes.append(f"sensor flagged {cls}: damage estimated from usage since install "
                         f"(month {install}), not from the reading")
            basis = "usage_model"
        elif age_months > stale:
            x0 = x_read + alpha * beta * ur[t - 1, 1] * age_months
            notes.append(f"last reading is {age_months} months old ({cls}); projected forward by model")
            basis = "stale_reading_projected"
        else:
            x0 = max(x_read, 0.0); basis = "reading"
        x0 = min(x0, 0.995 * Lk)
        rows.append([chan, x0, la_mu, la_var, beta, Lk, ur[t - 1, 1]])
        avail_rows.append([t, k, x0, la_mu, la_var])
        ledger[str(chan)] = dict(
            tail=f"T{t-1:04d}", component=comps[k - 1], serial=int(r["serial"]),
            damage_basis=basis, damage_est=round(float(x0), 4), threshold=round(Lk, 3),
            alpha_post=alpha, shrinkage=round(float(shrink), 3), n_increments=n_incr,
            sensor_status=cls, watch=bool(f[8]) if f is not None else False,
            notes=notes)
    pd.DataFrame(rows, columns=["chan", "x0", "la_mu", "la_var", "beta", "L", "usage"]).to_csv(tmp / "rul_in.csv", index=False)
    pd.DataFrame(avail_rows, columns=["tail", "comp", "x0", "la_mu", "la_var"]).to_csv(tmp / "avail_units.csv", index=False)
    (tmp / "state_ledger.json").write_text(json.dumps(dict(thresholds={comps[k-1]: L[k] for k in L},
                                                            channels=ledger)))
