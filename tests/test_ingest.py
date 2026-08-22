import numpy as np, pandas as pd, pytest
from pipeline.ingest import ingest, IngestRefused

COMPS = ["A1", "B2"]
ROSTER = pd.DataFrame(dict(tail=["T0000", "T0001"]))

def clean_frame(n=40):
    rows = []
    for t in ROSTER["tail"]:
        for c in COMPS:
            x = 0.0
            for m in range(n):
                x += 0.1
                rows.append(dict(tail=t, component=c, serial=hash((t, c)) % 1000, month=m, usage_hours=40.0, reading=round(x, 3)))
    return pd.DataFrame(rows)

def test_clean_passes_untouched():
    df = clean_frame(); r = ingest(df, ROSTER, COMPS, 40)
    assert len(r.clean) == len(df) and len(r.quarantine) == 0 and not r.summary["refused"]

def test_reconciliation_and_reason_codes():
    df = clean_frame()
    df.loc[0, "tail"] = "T9999"; df.loc[1, "component"] = "ZZ"; df.loc[2, "month"] = 99
    df.loc[3, "usage_hours"] = -1; df.loc[4, "reading"] = -5
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)           # duplicate
    r = ingest(df, ROSTER, COMPS, 40)
    assert len(r.clean) + len(r.quarantine) == len(df)
    codes = set(";".join(r.quarantine.reason).split(";"))
    assert codes == {"UNKNOWN_TAIL", "UNKNOWN_COMP", "MONTH_RANGE", "USAGE_NEGATIVE", "READING_NEGATIVE", "DUPLICATE"}

def test_point_outlier_units_error():
    df = clean_frame()
    i = df.index[(df["tail"] == "T0000") & (df.component == "A1") & (df.month == 20)][0]
    df.loc[i, "reading"] *= 10
    r = ingest(df, ROSTER, COMPS, 40)
    assert r.quarantine.reason.tolist() == ["POINT_OUTLIER"]

def test_refusal_when_source_is_broken():
    df = clean_frame(); df.loc[: len(df) // 4, "tail"] = "T9999"
    with pytest.raises(IngestRefused):
        ingest(df, ROSTER, COMPS, 40, max_quarantine_frac=0.05)

def test_schema_refusal():
    with pytest.raises(IngestRefused):
        ingest(clean_frame().drop(columns="reading"), ROSTER, COMPS, 40)

def test_type_coercion_quarantines_not_crashes():
    df = clean_frame(); df["reading"] = df["reading"].astype(object); df.loc[0, "reading"] = "n/a"
    r = ingest(df, ROSTER, COMPS, 40)
    assert r.quarantine.reason.tolist() == ["TYPE"]
