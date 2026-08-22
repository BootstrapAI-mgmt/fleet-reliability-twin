"""Run the full pipeline.  python run.py [--data data] [--work work]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.orchestrator import run_pipeline, GateFailure
from pipeline.ingest import IngestRefused

CFG = dict(
    max_quarantine_frac=0.05, max_flagged_frac=0.10, stale_months=3,
    high_risk_pfail=0.5, mc_reps=200, seed=11,
    tat=dict(mu=0.3, sigma=0.4, p_spare=0.85, backorder=2.0),  # lognormal months
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--work", default="work")
    a = ap.parse_args()
    meta = json.loads((Path(a.data) / "meta.json").read_text())   # n_months, horizon, component catalogue
    CFG.update(meta)
    t0 = time.time()
    try:
        st = run_pipeline(Path(a.data), Path(a.work), CFG)
    except (GateFailure, IngestRefused) as e:
        print(f"STOPPED: {e}"); sys.exit(2)
    for s in st.stage_log:
        print(f"  {s['stage']:8s} {s['status']:7s} {s.get('sec', '')}")
    print(f"done in {time.time()-t0:.0f}s; degradations: {len(st.degradations)}")
    print(open(Path(a.work) / "summary.md").read())
