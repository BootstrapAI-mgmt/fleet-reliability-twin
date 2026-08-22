import json, os, time
from pathlib import Path
import pytest
from pipeline.orchestrator import Stage, RunState, GateFailure
from pipeline import octave_bridge as ob
from pipeline.report import audit, ProvenanceError, explain

def test_stage_caches_and_invalidates(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); st = RunState(); calls = []
    def fn(tmp): calls.append(1); (tmp / "out.txt").write_text("x")
    s = lambda: Stage("s", tmp_path, [inp], ["out.txt"], dict(p=1), st)
    s().run(fn); s().run(fn)
    assert len(calls) == 1 and st.stage_log[-1]["status"] == "cached"
    inp.write_text("b"); s().run(fn)
    assert len(calls) == 2

def test_tampered_checkpoint_is_recomputed(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); st = RunState(); calls = []
    def fn(tmp): calls.append(1); (tmp / "out.txt").write_text("x")
    Stage("s", tmp_path, [inp], ["out.txt"], {}, st).run(fn)
    (tmp_path / "out.txt").write_text("corrupted")
    Stage("s", tmp_path, [inp], ["out.txt"], {}, st).run(fn)
    assert len(calls) == 2

def test_missing_declared_output_is_a_gate_failure(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a")
    with pytest.raises(GateFailure):
        Stage("s", tmp_path, [inp], ["out.txt"], {}, RunState()).run(lambda tmp: None)
    assert not (tmp_path / "s.manifest.json").exists()

def test_transient_failure_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    inp = tmp_path / "in.txt"; inp.write_text("a"); st = RunState(); n = {"k": 0}
    def fn(tmp):
        n["k"] += 1
        if n["k"] < 3: raise ob.NumericsUnavailable("octave busy")
        (tmp / "out.txt").write_text("x")
    Stage("s", tmp_path, [inp], ["out.txt"], {}, st).run(fn)
    assert n["k"] == 3 and sum(1 for s in st.stage_log if s["status"] == "retry") == 2

def test_permanent_failure_is_not_retried(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); n = {"k": 0}
    def fn(tmp): n["k"] += 1; raise ob.NumericsError("bad input")
    with pytest.raises(ob.NumericsError):
        Stage("s", tmp_path, [inp], ["out.txt"], {}, RunState()).run(fn)
    assert n["k"] == 1

def test_audit_rejects_untraceable_number():
    with pytest.raises(ProvenanceError):
        audit("expected failures 42 and also 17", {"ef": 42}, [])

def test_audit_rejects_omitted_degradation():
    with pytest.raises(ProvenanceError):
        audit("all good 42", {"ef": 42}, ["component 13D could not be modelled"])

def test_audit_passes_traceable():
    audit("expected failures 42 for T0001 23A in month 3", {"ef": 42}, [])

def test_explanation_states_evidence_basis():
    rec = dict(tail="T0001", component="23A", damage_est=3.2, threshold=8.0, damage_basis="usage_model",
               rul_months=dict(p05=4.0, p50=12.0, p95=None), p_fail_within_horizon=0.61, n_increments=30,
               shrinkage=0.4, sensor_status="stuck", watch=False)
    s = explain(rec, 18)
    assert "sensor flagged" in s and "40% of it is borrowed" in s and "stuck" in s
