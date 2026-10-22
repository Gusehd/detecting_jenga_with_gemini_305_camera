"""Run the block estimator on the (pristine) reference captures: deltas should be ~0
and consistent across views. Usage: .venv/bin/python tools/eval_ref.py [data/ref]"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jenga6dof.estimator import BlockEstimator  # noqa: E402

BLOCKS = [(4, 1), (4, 2), (4, 3), (5, 1), (8, 2), (9, 3), (12, 2), (15, 1), (16, 3)]


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    ref = Path(pos[0]) if pos else ROOT / "data" / "ref"
    verbose = "-v" in sys.argv
    cal = np.load(ref / "color_intrinsics.npz")
    est = BlockEstimator(cal["K"], cal["dist"], tuple(int(v) for v in cal["size"]),
                         use_edges="--no-edges" not in sys.argv)
    acc, times = defaultdict(list), []
    for f in sorted(ref.glob("snap_*.png")):
        bgr = cv2.imread(str(f))
        depth = np.load(str(f).replace(".png", "_depth.npy"))
        for n, m in BLOCKS:
            t = time.time()
            r = est.estimate(bgr, depth, n, m)
            times.append(time.time() - t)
            if not r["ok"]:
                continue
            d, a, c = r["dxyz_mm"], r["drpy_deg"], r["cam_corr"]
            acc[(n, m)].append(np.r_[d, a])
            if verbose:
                ib = r["info_block"]
                print(f"{f.stem[14:]} ({n:2d},{m}) d[mm] {d[0]:+5.1f} {d[1]:+5.1f} {d[2]:+5.1f} "
                      f"rpy[deg] {a[0]:+5.1f} {a[1]:+5.1f} {a[2]:+5.1f} | cam corr "
                      f"{np.linalg.norm(c[:3]) * 1000:4.1f}mm {np.degrees(np.linalg.norm(c[3:])):3.1f}deg"
                      f" | edge {ib.get('edge_strength_med', -1):.2f} depth {ib.get('depth_med_mm', -1):.1f}mm "
                      f"n {ib['n_depth']} {r['faces_used']}")
    print(f"\nmean time {np.mean(times) * 1000:.0f} ms/estimate")
    print("block    views | mean dx dy dz [mm] roll pitch yaw [deg] | std (same order)")
    allv = []
    for k in BLOCKS:
        if not acc[k]:
            print(f"{k}: not visible")
            continue
        v = np.array(acc[k])
        allv.append(v - v.mean(0))
        mu, sd = v.mean(0), v.std(0)
        print(f"{str(k):8s} {len(v):3d}  | " + " ".join(f"{x:+5.1f}" for x in mu)
              + " | " + " ".join(f"{x:4.1f}" for x in sd))
    allv = np.concatenate(allv)
    print("pooled view-to-view std:", " ".join(f"{x:4.1f}" for x in allv.std(0)))


if __name__ == "__main__":
    main()
