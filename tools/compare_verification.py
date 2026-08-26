#!/usr/bin/env python3
"""Numeric drift gate: committed verification artifact vs a fresh run.

    python tools/compare_verification.py COMMITTED FRESH [--rtol 1e-6]

The run-identity section ("== 0. ...") differs between machines by design and
is dropped from both sides. Every remaining line is reduced to a skeleton
(numbers replaced by a placeholder) plus its number sequence; skeletons must
be identical and numbers must agree -- exactly for integers, within rtol for
floats. Exit 0 on match, 1 on drift. Without this comparison, "CI re-derives
the numbers" would be a claim about effort, not about agreement.
"""
import argparse
import re
import sys

_NUMBER = re.compile(r"-?(?:\d[\d,]*\.\d+(?:[eE][+-]?\d+)?|\d[\d,]*|\.\d+)")


def load(path):
    """Yield (lineno, skeleton, numbers) for every comparable line."""
    drop = False
    for i, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.rstrip("\r\n")
        if line.startswith("== "):
            drop = line.startswith("== 0.")
        if drop:
            continue
        nums = _NUMBER.findall(line)
        yield i, _NUMBER.sub("<n>", line), nums


def numbers_agree(x, y, rtol):
    x, y = x.replace(",", ""), y.replace(",", "")
    if x == y:
        return True
    is_int = all("." not in v and "e" not in v.lower() for v in (x, y))
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    if is_int:
        return fx == fy
    return abs(fx - fy) <= 1e-9 + rtol * max(abs(fx), abs(fy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("committed"); ap.add_argument("fresh")
    ap.add_argument("--rtol", type=float, default=1e-6)
    a = ap.parse_args()
    left, right = list(load(a.committed)), list(load(a.fresh))
    problems = []
    if len(left) != len(right):
        problems.append(f"comparable line counts differ: {len(left)} committed vs {len(right)} fresh")
    for (_, sk_l, nn_l), (ln_r, sk_r, nn_r) in zip(left, right):
        if sk_l != sk_r:
            problems.append(f"fresh line {ln_r}: structure changed\n  - {sk_l}\n  + {sk_r}")
        elif len(nn_l) != len(nn_r) or any(not numbers_agree(x, y, a.rtol) for x, y in zip(nn_l, nn_r)):
            problems.append(f"fresh line {ln_r}: numbers drifted\n  - {nn_l}\n  + {nn_r}")
        if len(problems) > 25:
            problems.append("(more suppressed)")
            break
    if problems:
        print("verification drift detected:")
        print("\n".join(problems))
        return 1
    print(f"no drift: {len(left)} lines agree (identity section excluded, rtol {a.rtol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
