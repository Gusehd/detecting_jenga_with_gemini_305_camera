"""M1-1: solve the tag map from a calib_capture session and sanity-check the geometry.

Usage: .venv/bin/python tools/solve_tagmap.py [data/calib/<session>]
Writes config/tagmap.yaml (T_map_face for faces 0..3, map = face 0 frame).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jenga6dof.tagmap import detect, make_detector, solve_tagmap  # noqa: E402


def main():
    sessions = sorted((ROOT / "data" / "calib").glob("*/"))
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else sessions[-1]
    cfg = yaml.safe_load((ROOT / "config" / "tower.yaml").read_text())
    size = cfg["tags"]["size_mm"] / 1000
    gap = cfg["tags"]["strip_center_gap_mm"] / 1000
    cal = np.load(session / "intrinsics.npz")
    K, dist = cal["K"], cal["dist"]

    det = make_detector()
    frames = []
    for f in sorted(session.glob("f*.png")):
        d = detect(det, cv2.imread(str(f)))
        if d:
            frames.append(d)
    print(f"session {session.name}: {len(frames)} frames with tags")

    faces, info = solve_tagmap(frames, K, dist, size, gap)
    print(f"BA frames {info['n_frames']}  reproj RMS {info['rms_init_px']:.2f} -> "
          f"{info['rms_px']:.2f} px (max {info['max_px']:.2f} px)")

    print("\nface poses in map (= face 0) frame  [x right, y up, z toward front viewer]")
    for k, T in enumerate(faces):
        n = T[:3, 2]
        print(f"  face {k}: center [mm] {np.round(T[:3, 3] * 1000, 1)}  normal {np.round(n, 3)}")

    # sanity: adjacent faces ~90 deg, opposite faces ~antiparallel
    print("\nangles between face normals [deg]")
    for a in range(4):
        for b in range(a + 1, 4):
            ang = np.degrees(np.arccos(np.clip(faces[a][:3, 2] @ faces[b][:3, 2], -1, 1)))
            print(f"  face {a}-{b}: {ang:6.2f}")
    for a, b in [(0, 2), (1, 3)]:
        sep = abs((faces[b][:3, 3] - faces[a][:3, 3]) @ faces[a][:3, 2]) * 1000
        print(f"  plane separation face {a}-{b}: {sep:.1f} mm")
    # clockwise check: face 1 should be to the right (+x) of face 0 with normal ~ +x
    right_ok = faces[1][0, 3] > 0 and faces[1][0, 2] > 0.7
    left_ok = faces[3][0, 3] < 0 and faces[3][0, 2] < -0.7
    print(f"\nface 1 is on the RIGHT of front: {right_ok}   face 3 is on the LEFT: {left_ok}")

    out = {"session": session.name, "rms_px": info["rms_px"],
           "T_map_face": {k: np.round(T, 6).tolist() for k, T in enumerate(faces)}}
    (ROOT / "config" / "tagmap.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print("wrote config/tagmap.yaml")


if __name__ == "__main__":
    main()
