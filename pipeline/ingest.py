"""Hardened ingest for inspection records.

Design rule: a record that cannot be trusted is quarantined with a reason
code, never silently dropped and never silently repaired.  The stage asserts
that rows_in == rows_clean + rows_quarantined, and refuses to proceed if the
quarantine fraction exceeds a threshold, because a high rejection rate means
the *source* is broken, and a pipeline that keeps running on a broken source
emits confident wrong numbers.

Checks, in order (a row is tagged with every reason that applies):
  SCHEMA_COLUMNS   required columns missing            -> whole file refused
  TYPE             non-numeric where numeric expected
  UNKNOWN_TAIL     tail not on the fleet roster
  UNKNOWN_COMP     component code not in catalogue
  MONTH_RANGE      month outside [0, n_months)
  USAGE_NEGATIVE   usage hours < 0
  DUPLICATE        exact duplicate (tail, component, serial, month)
  READING_NEGATIVE reading below -fence (noise can go slightly negative)
  READING_FENCE    reading above a physically impossible level
  POINT_OUTLIER    single reading disagrees with BOTH neighbours by > jump
                   while the neighbours agree with each other (units error)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["tail", "component", "serial", "month", "usage_hours", "reading"]
NUMERIC = ["serial", "month", "usage_hours", "reading"]


class IngestRefused(RuntimeError):
    pass


@dataclass
class IngestResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    summary: dict = field(default_factory=dict)


# No aircraft flies more hours in a month than the month contains.
MAX_USAGE_HOURS_PER_MONTH = 24 * 31


def ingest(insp: pd.DataFrame, roster: pd.DataFrame, components: list[str],
           n_months: int, max_quarantine_frac: float = 0.05,
           neg_fence: float = 1.0, upper_fence: float = 30.0,
           jump: float = 5.0) -> IngestResult:
    missing = [c for c in REQUIRED if c not in insp.columns]
    if missing:
        raise IngestRefused(f"SCHEMA_COLUMNS: missing {missing}")
    n_in = len(insp)
    df = insp.copy().reset_index(drop=True)
    reasons: dict[int, list[str]] = {}

    def tag(mask, code):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons.setdefault(int(i), []).append(code)

    for c in NUMERIC:
        coerced = pd.to_numeric(df[c], errors="coerce")
        tag(coerced.isna(), "TYPE")
        df[c] = coerced

    tag(~df["tail"].isin(set(roster["tail"])), "UNKNOWN_TAIL")
    tag(~df["component"].isin(set(components)), "UNKNOWN_COMP")
    tag((df["month"] < 0) | (df["month"] >= n_months), "MONTH_RANGE")
    tag(df["usage_hours"] < 0, "USAGE_NEGATIVE")
    # A negative check alone is not enough. `to_numeric` happily parses
    # "inf", "Infinity" and "1e400", and a single such row at fleet scale
    # dragged one component's beta from 0.370 to 1,018,880 -- a factor of
    # 2.76 million -- turning a component with a coin-flip chance of
    # failing inside the horizon into an immortal one (P(fail) 0.4988 ->
    # 0.0001). Every gate downstream accepted it.
    tag(~np.isfinite(df["usage_hours"]), "USAGE_NONFINITE")
    tag(df["usage_hours"] > MAX_USAGE_HOURS_PER_MONTH, "USAGE_FENCE")
    tag(df.duplicated(["tail", "component", "serial", "month"], keep="first"), "DUPLICATE")
    tag(df["reading"] < -neg_fence, "READING_NEGATIVE")
    tag(df["reading"] > upper_fence, "READING_FENCE")

    # Second pass: point outliers, evaluated only on rows that passed pass one
    bad = np.zeros(n_in, dtype=bool)
    if reasons:
        bad[list(reasons)] = True
    ok = df[~bad].sort_values(["tail", "component", "serial", "month"])
    g = ok.groupby(["tail", "component", "serial"], sort=False)["reading"]
    prev = g.shift(1); nxt = g.shift(-1)
    r = ok["reading"]
    point = ((r - prev).abs() > jump) & ((r - nxt).abs() > jump) & ((nxt - prev).abs() < jump / 2)
    point = point.fillna(False)
    tag(point.reindex(df.index, fill_value=False), "POINT_OUTLIER")

    q_idx = sorted(reasons)
    quarantine = df.loc[q_idx].copy()
    quarantine["reason"] = [";".join(reasons[i]) for i in q_idx]
    clean = df.drop(index=q_idx)
    clean = clean.astype({"serial": int, "month": int})

    # Reconciliation assertion: nothing lost, nothing invented
    assert len(clean) + len(quarantine) == n_in, "reconciliation failed"
    frac = len(quarantine) / max(n_in, 1)
    counts = pd.Series([r for v in reasons.values() for r in v]).value_counts().to_dict() if reasons else {}
    summary = dict(rows_in=n_in, rows_clean=len(clean), rows_quarantined=len(quarantine),
                   quarantine_frac=round(frac, 5), reason_counts={k: int(v) for k, v in counts.items()},
                   refused=bool(frac > max_quarantine_frac))
    if summary["refused"]:
        raise IngestRefused(f"quarantine fraction {frac:.3f} exceeds {max_quarantine_frac}: "
                            f"source is untrustworthy; {json.dumps(summary['reason_counts'])}")
    return IngestResult(clean, quarantine, summary)
