import json, os, time
from pathlib import Path
import pytest
from pipeline.orchestrator import Stage, RunState, GateFailure
from pipeline import octave_bridge as ob

def test_stage_caches_and_invalidates(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); st = RunState(); calls = []
    def fn(tmp): calls.append(1); (tmp / "out.txt").write_text("x")
    s = Stage("s", tmp_path, [inp], ["out.txt"], {"p": 1}, st)
    s.run(fn); s.run(fn)
    assert calls == [1] and st.stage_log[-1]["status"] == "cached"
    inp.write_text("b"); Stage("s", tmp_path, [inp], ["out.txt"], {"p": 1}, st).run(fn)
    assert calls == [1, 1]
    Stage("s", tmp_path, [inp], ["out.txt"], {"p": 2}, st).run(fn)   # param change also invalidates
    assert calls == [1, 1, 1]

def test_tampered_checkpoint_is_recomputed(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); st = RunState(); calls = []
    def fn(tmp): calls.append(1); (tmp / "out.txt").write_text("x")
    Stage("s", tmp_path, [inp], ["out.txt"], {}, st).run(fn)
    (tmp_path / "out.txt").write_text("corrupted")
    Stage("s", tmp_path, [inp], ["out.txt"], {}, st).run(fn)
    assert calls == [1, 1]

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
    assert n["k"] == 3 and sum(e["status"] == "retry" for e in st.stage_log) == 2

def test_permanent_failure_is_not_retried(tmp_path):
    inp = tmp_path / "in.txt"; inp.write_text("a"); n = {"k": 0}
    def fn(tmp): n["k"] += 1; raise ob.NumericsError("bad input")
    with pytest.raises(ob.NumericsError):
        Stage("s", tmp_path, [inp], ["out.txt"], {}, RunState()).run(fn)
    assert n["k"] == 1
