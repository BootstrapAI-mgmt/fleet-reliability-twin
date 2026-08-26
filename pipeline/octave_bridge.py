"""Run a MATLAB stage under GNU Octave (or MATLAB if MATLAB_CMD is set).

The numerics never see Python objects: every call is `run_stage(stage, work)`
against a directory of CSV/JSON files, so a stage can be reproduced by hand
from its checkpoint with nothing but the work directory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MATLAB_DIR = Path(__file__).resolve().parents[1] / "matlab"


class NumericsError(RuntimeError):
    """Permanent failure inside the numerics (bad input, assertion, syntax)."""


class NumericsUnavailable(RuntimeError):
    """Transient: the interpreter could not be started (retryable)."""


def _q(p) -> str:
    """Quote a path for a single-quoted Octave/MATLAB string literal.

    Both languages escape an embedded quote by doubling it; interpolating
    the raw path meant a checkout under a directory with an apostrophe
    (O'Brien, "Josh's laptop") produced a syntax error inside the eval.
    """
    return str(p).replace("'", "''")


def run_stage(stage: str, work: Path, timeout: int = 3600) -> str:
    work = Path(work).resolve()
    code = f"addpath('{_q(MATLAB_DIR)}'); run_stage('{_q(stage)}','{_q(work)}')"
    cmd = os.environ.get("MATLAB_CMD")
    if cmd:
        argv = [cmd, "-batch", code]
    else:
        # Prefer the CLI binary: on Windows the bare `octave` launcher can
        # hang waiting for environment bootstrap that only the -cli
        # executable does itself, and a headless pipeline wants the CLI
        # regardless.
        octave = shutil.which("octave-cli") or shutil.which("octave")
        if not octave:
            raise NumericsUnavailable("neither MATLAB_CMD nor octave found on PATH")
        argv = [octave, "--no-gui", "-q", "--eval", code]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise NumericsUnavailable(str(e)) from e
    except subprocess.TimeoutExpired as e:
        raise NumericsUnavailable(f"stage {stage} timed out after {timeout}s") from e
    if p.returncode != 0:
        raise NumericsError(f"stage {stage} failed (rc={p.returncode}):\n{p.stderr[-4000:]}")
    return p.stdout
