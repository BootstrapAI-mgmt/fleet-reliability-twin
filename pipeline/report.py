"""Evidence ledger and provenance-enforced summary.

Every channel's forecast carries its evidence: where the current damage
estimate came from (reading / projected stale reading / usage model because
the sensor is flagged), how much of its severity posterior is borrowed from
the fleet, how many increments it rests on, and the sensor status.  That is
the explanation an operator gets with the number.

The summary prose is generated from a facts dictionary and then audited:
any number in the prose that cannot be traced to a fact is a hard error, and
the prose is refused if a recorded degradation is not mentioned.  A report
that silently omits "component X could not be modelled" is a wrong report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FAULT_NAMES = {0: "none", 1: "bias_step", 2: "scale_error", 3: "stuck", 4: "dropout", 5: "accelerated"}


class ProvenanceError(RuntimeError):
    pass


def build_ledger(work: Path, comps, cfg, st) -> dict:
    state = json.loads((work / "state_ledger.json").read_text())
    rul = pd.read_csv(work / "rul_out.csv", header=None)
    H = cfg["forecast_months"]
    rul.columns = ["chan", "p05", "p50", "p95", "pfail_H"] + [f"F{m}" for m in range(1, H + 1)]
    rul = rul.set_index(rul["chan"].astype(int).astype(str))
    channels = {}
    for chan, rec in state["channels"].items():
        q = rul.loc[chan]
        rec = dict(rec)
        rec["rul_months"] = dict(p05=_f(q["p05"]), p50=_f(q["p50"]), p95=_f(q["p95"]))
        rec["p_fail_within_horizon"] = round(float(q["pfail_H"]), 4)
        rec["failure_cdf_by_month"] = [round(float(q[f"F{m}"]), 4) for m in range(1, H + 1)]
        rec["explanation"] = explain(rec, H)
        channels[chan] = rec

    pf = np.array([c["p_fail_within_horizon"] for c in channels.values()])
    order = sorted(channels, key=lambda c: -channels[c]["p_fail_within_horizon"])
    avail = st.facts["avail"]
    ledger = dict(
        horizon_months=H,
        n_channels_forecast=len(channels),
        expected_failures_within_horizon=round(float(pf.sum()), 1),
        n_high_risk=int((pf > cfg["high_risk_pfail"]).sum()),
        high_risk_threshold=cfg["high_risk_pfail"],
        top_risk=[dict(chan=c, **{k: channels[c][k] for k in
                                   ("tail", "component", "p_fail_within_horizon", "rul_months",
                                    "damage_basis", "sensor_status", "shrinkage")})
                  for c in order[:25]],
        availability=dict(month=list(range(1, H + 1)),
                          mean=[round(r[0], 4) for r in avail["avail"]],
                          p05=[round(r[1], 4) for r in avail["avail"]],
                          p95=[round(r[2], 4) for r in avail["avail"]]),
        expected_failures_by_month_mc=[round(r[0], 1) for r in avail["fail_total"]],
        thresholds=state["thresholds"],
        ingest=st.facts["ingest"], detect=st.facts["detect"], fit=st.facts["fit2"],
        degradations=list(st.degradations),
        channels=channels,
    )
    return ledger


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 2)


def explain(rec: dict, H: int) -> str:
    q = rec["rul_months"]
    borrowed = rec["shrinkage"]
    basis = {"reading": "the latest inspection reading",
             "stale_reading_projected": "a stale reading projected forward by the wear model",
             "usage_model": "usage since install only (sensor flagged, reading untrusted)"}[rec["damage_basis"]]
    p50 = "beyond the search horizon" if q["p50"] is None else f"{q['p50']} months"
    lo = "-" if q["p05"] is None else q["p05"]
    hi = "beyond horizon" if q["p95"] is None else q["p95"]
    s = (f"{rec['tail']} {rec['component']}: damage {rec['damage_est']} of threshold "
         f"{rec['threshold']} from {basis}. Median remaining life {p50} "
         f"(90% interval {lo} to {hi}); P(fail within {H} months) = {rec['p_fail_within_horizon']}. "
         f"Severity estimate rests on {rec['n_increments']} inspection increments; "
         f"{int(round(borrowed * 100))}% of it is borrowed from the fleet prior.")
    if rec["sensor_status"] != "none":
        s += f" Sensor status: {rec['sensor_status']}."
    if rec.get("watch"):
        s += " On watch list: weak evidence of a rate change, not confirmed."
    return s


# --------------------------------------------------------------------------
def render_summary(ledger: dict, st) -> str:
    facts = {}
    def fact(name, value):
        facts[name] = value; return value

    H = fact("H", ledger["horizon_months"])
    nch = fact("nch", ledger["n_channels_forecast"])
    ef = fact("ef", ledger["expected_failures_within_horizon"])
    nhr = fact("nhr", ledger["n_high_risk"])
    thr = fact("thr", ledger["high_risk_threshold"])
    ing = ledger["ingest"]
    # "accepted" must count the rows that were accepted, not the rows that
    # arrived: quoting rows_in as accepted double-counted the quarantined
    # rows in the audited narrative's first sentence.
    rin = fact("rin", ing["rows_clean"]); rq = fact("rq", ing["rows_quarantined"])
    qf = fact("qf", round(100 * ing["quarantine_frac"], 2))
    det = ledger["detect"]
    nfl = fact("nfl", det["n_flagged"]); nw = fact("nw", det["n_watch"])
    av = ledger["availability"]
    a1 = fact("a1", round(100 * av["mean"][0], 1)); aH = fact("aH", round(100 * av["mean"][-1], 1))
    aHlo = fact("aHlo", round(100 * av["p05"][-1], 1)); aHhi = fact("aHhi", round(100 * av["p95"][-1], 1))
    band = fact("band", 90)
    by = det["by_class"]
    cls_line = ", ".join(f"{k} {fact('c_' + k, v)}" for k, v in by.items() if k != "none")

    lines = [
        "# Fleet reliability forecast",
        "",
        f"Ingest accepted {rin:,} inspection records and quarantined {rq:,} ({qf}%) with reason codes; "
        f"nothing was silently repaired.",
        f"Sensor screening flagged {nfl} channels ({cls_line}) and placed {nw} more on a watch list "
        f"with weak, unconfirmed evidence of a rate change. Only the classes that corrupt the "
        f"reading itself (bias step, scale error, stuck) are excluded from the fleet fit and have "
        f"their damage state estimated from usage; dropout channels keep the readings that did "
        f"arrive, and accelerated channels are kept in full, because accelerated wear is real "
        f"damage rather than a sensor fault.",
        "",
        f"Forecasts were issued for {nch:,} installed components over a {H}-month horizon. "
        f"The expected number of failures in that window is {ef}; {nhr} components exceed the "
        f"{thr} failure-probability threshold and are listed in the ledger with their evidence.",
        f"Monte Carlo fleet availability is {a1}% in month 1 and {aH}% in month {H} "
        f"({band}% band {aHlo}% to {aHhi}%).",
        "",
        "## Top risk",
        "",
    ]
    for i, t in enumerate(ledger["top_risk"][:10]):
        p = fact(f"tr{i}", t["p_fail_within_horizon"])
        lines.append(f"- {t['tail']} {t['component']}: P(fail) {p}, damage basis {t['damage_basis']}, "
                     f"sensor {t['sensor_status']}")
    if ledger["degradations"]:
        lines += ["", "## Degradations carried through this run", ""]
        lines += [f"- {d}" for d in ledger["degradations"]]
    md = "\n".join(lines) + "\n"
    audit(md, facts, ledger["degradations"])
    return md


def audit(md: str, facts: dict, degradations: list[str]):
    """Refuse prose containing a number not traceable to a fact."""
    allowed = set()
    for v in facts.values():
        allowed.add(f"{v}"); allowed.add(f"{v:,}" if isinstance(v, int) else f"{v}")
        if isinstance(v, float):
            allowed.add(f"{v:g}")
    # strip markdown bullets / headers, tails (T0001) and component codes (23A) before scanning
    body = re.sub(r"T\d{4}|\b\d{2}[A-H]\b|month \d+", "", md)
    for num in re.findall(r"\d[\d,]*\.?\d*", body):
        if num.rstrip(".") not in allowed and num.replace(",", "") not in allowed:
            raise ProvenanceError(f"untraceable figure in narrative: {num!r}")
    for d in degradations:
        if d not in md:
            raise ProvenanceError(f"narrative omits degradation: {d!r}")
