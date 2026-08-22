import numpy as np, pandas as pd, pytest
from pipeline.ingest import ingest, IngestRefused

ROSTER = pd.DataFrame(dict(tail=["T0000", "T0001"]))
COMPS = ["23A", "64E"]

def base(n=30):
    rows = []
    for t in ROSTER["tail"]:
        for c in COMPS:
            for m in range(n):
                rows.append((t, c, 7, m, 40.0, 0.1 * m))
    return pd.DataFrame(rows, columns=["tail", "component", "serial", "month", "usage_hours", "reading"])

def test_clean_passes_and_reconciles():
    df = base(); r = ingest(df, ROSTER, COMPS, 60)
    assert len(r.clean) == len(df) and len(r.quarantine) == 0
    assert r.summary["rows_in"] == r.summary["rows_clean"] + r.summary["rows_quarantined"]

@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.assign(tail=["T9999"] + list(d["tail"][1:])), "UNKNOWN_TAIL"),
    (lambda d: d.assign(component=["ZZZ"] + list(d.component[1:])), "UNKNOWN_COMP"),
    (lambda d: d.assign(month=[99] + list(d.month[1:])), "MONTH_RANGE"),
    (lambda d: d.assign(usage_hours=[-1.0] + list(d.usage_hours[1:])), "USAGE_NEGATIVE"),
    (lambda d: d.assign(reading=[-5.0] + list(d.reading[1:])), "READING_NEGATIVE"),
    (lambda d: d.assign(reading=[99.0] + list(d.reading[1:])), "READING_FENCE"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "DUPLICATE"),
    (lambda d: d.assign(reading=["x"] + list(d.reading[1:])), "TYPE"),
])
def test_each_fault_class_is_caught_with_reason(mutate, code):
    r = ingest(mutate(base()), ROSTER, COMPS, 60)
    assert len(r.quarantine) == 1 and code in r.quarantine.reason.iloc[0]

def test_point_outlier_units_error():
    d = base(); i = d.index[(d["tail"] == "T0000") & (d.component == "23A") & (d.month == 15)][0]
    d.loc[i, "reading"] *= 10
    r = ingest(d, ROSTER, COMPS, 60)
    assert "POINT_OUTLIER" in r.quarantine.reason.iloc[0]

def test_refuses_broken_source():
    d = base(); d.loc[: len(d) // 10, "tail"] = "T9999"
    with pytest.raises(IngestRefused):
        ingest(d, ROSTER, COMPS, 60, max_quarantine_frac=0.05)

def test_refuses_missing_schema():
    with pytest.raises(IngestRefused):
        ingest(base().drop(columns="reading"), ROSTER, COMPS, 60)
