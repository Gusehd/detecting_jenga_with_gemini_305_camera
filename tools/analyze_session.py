"""Re-run the estimator on recorded live sessions and summarise per label.

Usage: .venv/bin/python tools/analyze_session.py data/live/<session> [more sessions...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jenga6dof.estimator import BlockEstimator  # noqa: E402
from tools.live_block import fuse  # noqa: E402


def fmt(v):
    return " ".join(f"{x:+5.1f}" for x in v)


def main():
    sessions = [Path(a) for a in sys.argv[1:]]
    est = None
    groups = defaultdict(list)
    for sdir in sessions:
        cal = np.load(sdir / "intrinsics.npz")
        if est is None:
            est = BlockEstimator(cal["K"], cal["dist"], tuple(int(v) for v in cal["size"]))
        for line in open(sdir / "records.jsonl"):
            rec = json.loads(line)
            bgr = cv2.imread(str(sdir / f"{rec['frame']}.jpg"))
            depth = cv2.imread(str(sdir / f"{rec['frame']}_depth.png"),
                               cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e4
            r = est.estimate(bgr, depth, rec["n"], rec["m"])
            if r["ok"]:
                groups[(rec["n"], rec["m"], rec["label"])].append(r)
    print("label           side     n | REL dx dy dz [mm] r p y [deg]        | ABS (same)")
    for key in sorted(groups):
        rs = groups[key]
        by_side = defaultdict(list)
        for r in rs:
            by_side["+x end" if "+x" in r["faces_used"] else
                    "-x end" if "-x" in r["faces_used"] else "side"].append(r)
        for side, sub in sorted(by_side.items()):
            print(f"{str(key):16s} {side:7s} {len(sub):3d} | {fmt(fuse(sub, False)[1])} | "
                  f"{fmt(fuse(sub, True)[1])}")
        print(f"{str(key):16s} {'FUSED':7s} {len(rs):3d} | {fmt(fuse(rs, False)[1])} | "
              f"{fmt(fuse(rs, True)[1])}")


if __name__ == "__main__":
    main()
