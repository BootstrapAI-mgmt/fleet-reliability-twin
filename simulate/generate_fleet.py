"""Synthetic fleet generator with hidden ground truth.

Generates a large fleet (tails x component types x months) whose component
degradation follows a usage-scaled gamma process.  Each component instance
accumulates damage X(t); it fails (is removed and replaced) when X crosses the
component's threshold L.  A monthly inspection records a noisy sensor reading
of X.  A fraction of sensor channels are injected with faults, and a fraction
of records are corrupted at the data-entry level.

Everything that the pipeline is later asked to infer (fleet priors, per-unit
severity, fault onsets, true remaining life) is written to truth.json and is
NEVER read by the pipeline.  Only verify.py reads it.

Gamma process:  dX over usage du ~ Gamma(shape = alpha * du, scale = beta)
  alpha: per-unit severity (fleet prior lognormal around component nominal)
  beta : component property, common across the fleet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPONENTS = {
    # code: (nominal_alpha, beta, threshold, sensor_sd)
    # alpha*beta = mean damage per flight hour; alpha*45 = gamma shape per
    # month at nominal usage.  The set deliberately spans smooth wear
    # (shape/month ~ 4, e.g. bearing wear) to erratic damage (shape/month ~ 0.2,
    # e.g. crack growth under variable load) because detectability of a rate
    # change depends strongly on it.
    "64E": (0.1000, 0.040, 6.0, 0.10),   # bearing assembly      shape/mo 4.5
    "52C": (0.0400, 0.100, 7.0, 0.12),   # actuator              shape/mo 1.8
    "75G": (0.0200, 0.250, 7.5, 0.14),   # brake assembly        shape/mo 0.9
    "23A": (0.0120, 0.400, 8.0, 0.15),   # hydraulic pump        shape/mo 0.54
    "27F": (0.0050, 0.900, 8.5, 0.18),   # avionics cooling fan  shape/mo 0.22
    "41B": (0.0045, 1.200, 9.0, 0.20),   # fuel controller       shape/mo 0.20
    "13D": (0.0030, 1.500, 9.0, 0.25),   # generator             shape/mo 0.14
    "19H": (0.0035, 1.300, 9.5, 0.22),   # environmental control shape/mo 0.16
}

FAULT_CLASSES = ["bias_step", "scale_error", "stuck", "dropout", "accelerated"]


def generate(n_tails: int, n_months: int, seed: int, fault_rate: float,
             dirty_rate: float, forecast_months: int):
    rng = np.random.default_rng(seed)
    comps = list(COMPONENTS)
    K = len(comps)

    # Per-tail usage (flight hours / month) and severity multiplier (base, harsh env)
    usage_rate = rng.lognormal(np.log(45.0), 0.35, n_tails)
    tail_env = rng.lognormal(0.0, 0.45, n_tails)  # heterogeneity in alpha

    # Per (tail, comp) severity alpha
    alpha = np.empty((n_tails, K))
    for k, c in enumerate(comps):
        a0 = COMPONENTS[c][0]
        alpha[:, k] = a0 * tail_env * rng.lognormal(0.0, 0.15, n_tails)
    beta = np.array([COMPONENTS[c][1] for c in comps])
    L = np.array([COMPONENTS[c][2] for c in comps])
    sd = np.array([COMPONENTS[c][3] for c in comps])

    total_months = n_months + forecast_months
    # Damage state, serial numbers, install month
    X = np.zeros((n_tails, K))
    # start with random prior damage so the fleet is not all new
    X[:] = rng.uniform(0, 0.5, (n_tails, K)) * L
    serial = np.arange(n_tails * K).reshape(n_tails, K)
    next_serial = n_tails * K

    # Sensor fault schedule: (tail, comp, onset_month, class, magnitude)
    n_channels = n_tails * K
    n_faults = int(fault_rate * n_channels)
    fault_idx = rng.choice(n_channels, n_faults, replace=False)
    faults = {}
    for f in fault_idx:
        t, k = divmod(int(f), K)
        onset = int(rng.integers(6, n_months - 6))
        cls = str(rng.choice(FAULT_CLASSES))
        faults[(t, k)] = dict(onset=onset, cls=cls,
                              mag=float(rng.uniform(1.0, 2.5)))

    insp_rows = []
    event_rows = []
    monthly_usage = np.zeros((n_tails, total_months))
    first_forecast_failure = np.full((n_tails, K), -1.0)  # true months-to-failure after horizon
    state_at_horizon = None

    fault_state = {}  # (t,k) -> stuck value etc.

    for m in range(total_months):
        du = usage_rate * rng.lognormal(0.0, 0.15, n_tails)
        monthly_usage[:, m] = du
        shape = alpha * du[:, None]
        # accelerated degradation (true fault, not sensor) doubles alpha after onset
        for (t, k), fl in faults.items():
            if fl["cls"] == "accelerated" and m >= fl["onset"]:
                shape[t, k] *= 2.0
        dX = rng.gamma(shape, beta[None, :])
        X = X + dX

        # Failures: crosses threshold this month
        failed = X >= L[None, :]
        for t, k in zip(*np.where(failed)):
            if m < n_months:
                event_rows.append(dict(tail=f"T{t:04d}", component=comps[k],
                                       serial=int(serial[t, k]), month=m,
                                       event="FAIL", usage_hours=float(du[t])))
            else:
                if first_forecast_failure[t, k] < 0:
                    first_forecast_failure[t, k] = m - n_months + 1
            X[t, k] = 0.0
            serial[t, k] = next_serial
            next_serial += 1
            if (t, k) in faults:  # replacing the part clears a real fault, not a sensor fault
                if faults[(t, k)]["cls"] == "accelerated":
                    faults[(t, k)]["cleared_month"] = m

        if m == n_months - 1:
            state_at_horizon = X.copy()
            serial_at_horizon = serial.copy()
            alpha_at_horizon = alpha.copy()

        if m >= n_months:
            continue  # forecast window is truth only, never observed

        # Inspection readings
        reading = X + rng.normal(0, sd[None, :], X.shape)
        missing = np.zeros_like(X, dtype=bool)
        for (t, k), fl in faults.items():
            if m < fl["onset"]:
                continue
            c = fl["cls"]
            if c == "bias_step":
                reading[t, k] += fl["mag"]
            elif c == "scale_error":
                reading[t, k] *= fl["mag"]
            elif c == "stuck":
                if (t, k) not in fault_state:
                    fault_state[(t, k)] = reading[t, k]
                reading[t, k] = fault_state[(t, k)]
            elif c == "dropout":
                if rng.random() < 0.7:
                    missing[t, k] = True
        for t in range(n_tails):
            for k in range(K):
                if missing[t, k]:
                    continue
                insp_rows.append((f"T{t:04d}", comps[k], int(serial[t, k]), m,
                                  float(du[t]), round(float(reading[t, k]), 4)))

    insp = pd.DataFrame(insp_rows, columns=["tail", "component", "serial", "month",
                                            "usage_hours", "reading"])
    events = pd.DataFrame(event_rows)

    # ---- Data-entry corruption (what ingest must catch) -------------------
    n_dirty = int(dirty_rate * len(insp))
    dirty_log = []
    idx = rng.choice(len(insp), n_dirty, replace=False)
    kinds = rng.choice(["dup", "neg_reading", "bad_tail", "units_x10",
                        "usage_neg", "bad_component", "month_oob"], n_dirty)
    insp = insp.copy()
    extra = []
    for i, kind in zip(idx, kinds):
        r = insp.iloc[i]
        dirty_log.append(dict(kind=str(kind), tail=r["tail"], component=r["component"],
                              month=int(r["month"])))
        if kind == "dup":
            extra.append(r.to_dict())
        elif kind == "neg_reading":
            insp.iat[i, 5] = -abs(insp.iat[i, 5]) - 1.5
        elif kind == "bad_tail":
            insp.iat[i, 0] = "T9999"
        elif kind == "units_x10":
            insp.iat[i, 5] = insp.iat[i, 5] * 10.0
        elif kind == "usage_neg":
            insp.iat[i, 4] = -insp.iat[i, 4]
        elif kind == "bad_component":
            insp.iat[i, 1] = "ZZZ"
        elif kind == "month_oob":
            insp.iat[i, 3] = n_months + 40
    if extra:
        insp = pd.concat([insp, pd.DataFrame(extra)], ignore_index=True)
    insp = insp.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    roster = pd.DataFrame(dict(tail=[f"T{t:04d}" for t in range(n_tails)],
                               base=rng.choice(["HILL", "TINKER", "ROBINS"], n_tails),
                               usage_rate_nominal=np.round(usage_rate, 2)))

    truth = dict(
        components={c: dict(alpha_nominal=v[0], beta=v[1], threshold=v[2], sensor_sd=v[3])
                    for c, v in COMPONENTS.items()},
        alpha_per_unit={f"T{t:04d}|{comps[k]}": float(alpha_at_horizon[t, k])
                        for t in range(n_tails) for k in range(K)},
        damage_at_horizon={f"T{t:04d}|{comps[k]}": float(state_at_horizon[t, k])
                           for t in range(n_tails) for k in range(K)},
        serial_at_horizon={f"T{t:04d}|{comps[k]}": int(serial_at_horizon[t, k])
                           for t in range(n_tails) for k in range(K)},
        months_to_failure_after_horizon={
            f"T{t:04d}|{comps[k]}": float(first_forecast_failure[t, k])
            for t in range(n_tails) for k in range(K)},
        sensor_faults={f"T{t:04d}|{comps[k]}": fl for (t, k), fl in faults.items()},
        dirty_records=dirty_log,
        n_months=n_months, forecast_months=forecast_months,
        usage_rate={f"T{t:04d}": float(usage_rate[t]) for t in range(n_tails)},
    )
    return insp, events, roster, truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tails", type=int, default=1500)
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--forecast-months", type=int, default=18)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fault-rate", type=float, default=0.04)
    p.add_argument("--dirty-rate", type=float, default=0.01)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    insp, events, roster, truth = generate(a.tails, a.months, a.seed, a.fault_rate,
                                           a.dirty_rate, a.forecast_months)
    insp.to_csv(out / "inspections.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    roster.to_csv(out / "roster.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth))
    # non-secret metadata the pipeline is allowed to read
    (out / "meta.json").write_text(json.dumps(dict(n_months=a.months, forecast_months=a.forecast_months,
                                                   components=list(COMPONENTS))))
    print(f"inspections={len(insp):,} events={len(events):,} tails={a.tails} "
          f"faulted_channels={len(truth['sensor_faults'])} dirty={len(truth['dirty_records'])}")


if __name__ == "__main__":
    main()
