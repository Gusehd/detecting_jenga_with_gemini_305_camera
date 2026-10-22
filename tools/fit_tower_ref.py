"""M1-2: fit the tower height offset (and block height) from reference captures.

Horizontal pose/yaw come from the tag map (tower sits in the jig). The layer-1
bottom height z0 and the effective block height h are found by aligning the
projected layer seams with the dark seam lines in the images.

Usage: .venv/bin/python tools/fit_tower_ref.py [data/ref]
Writes T_tagmap_tower and block height into config/tower_ref.yaml and saves overlays.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jenga6dof.localize import camera_pose, load_tagmap, tower_from_faces  # noqa: E402
from jenga6dof.tagmap import detect, make_detector, to_rt  # noqa: E402
from jenga6dof.tower_model import TowerModel  # noqa: E402

MIN_TAGS = 2


def seam_map(bgr):
    """Response image highlighting thin dark horizontal-ish lines (layer seams)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    return cv2.GaussianBlur(bh, (0, 0), 1.0)


def seam_points(tm, half, k_layers, z0, h, n_samples=40):
    """3D points (tower frame) on the 4 outer faces along each layer boundary."""
    pts = []
    s = np.linspace(-half * 0.9, half * 0.9, n_samples)
    for k in k_layers:
        z = z0 + k * h
        for face in range(4):
            if face == 0:
                p = np.stack([s, np.full_like(s, -half), np.full_like(s, z)], 1)
            elif face == 1:
                p = np.stack([np.full_like(s, half), s, np.full_like(s, z)], 1)
            elif face == 2:
                p = np.stack([s, np.full_like(s, half), np.full_like(s, z)], 1)
            else:
                p = np.stack([np.full_like(s, -half), s, np.full_like(s, z)], 1)
            pts.append(np.c_[p, np.full(len(p), face)])
    return np.concatenate(pts)


def visible_faces(T_cam_tower, half):
    cam_in_tower = np.linalg.inv(T_cam_tower)[:3, 3]
    normals = {0: [0, -1, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [-1, 0, 0]}
    centers = {f: np.array(n) * half for f, n in normals.items()}
    return [f for f, n in normals.items()
            if np.dot(cam_in_tower[:2] - centers[f][:2], np.array(n[:2])) > 0.05]


def score(views, tm, half, z0, h):
    total, cnt = 0.0, 0
    k_layers = range(1, tm.num_layers)  # internal seams between layers k and k+1
    for v in views:
        P = seam_points(tm, half, k_layers, z0, h)
        P = P[np.isin(P[:, 3], v["faces"])]
        Pt = P[:, :3] @ v["T_ct"][:3, :3].T + v["T_ct"][:3, 3]
        uv, _ = cv2.projectPoints(Pt, np.zeros(3), np.zeros(3), v["K"], v["dist"])
        uv = uv.reshape(-1, 2).round().astype(int)
        H, W = v["seam"].shape
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        total += v["seam"][uv[ok, 1], uv[ok, 0]].sum()
        cnt += ok.sum()
    return total / max(cnt, 1)


def draw_overlay(bgr, T_ct, K, dist, tm, T_tower_block_fn, color):
    corners = tm.block_corners()
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for (n, m), T_tb in T_tower_block_fn().items():
        P = corners @ T_tb[:3, :3].T + T_tb[:3, 3]
        Pc = P @ T_ct[:3, :3].T + T_ct[:3, 3]
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(-1, 2).astype(int)
        for a, b in edges:
            cv2.line(bgr, tuple(uv[a]), tuple(uv[b]), color, 1, cv2.LINE_AA)


def main():
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "ref"
    cfg = yaml.safe_load((ROOT / "config" / "tower.yaml").read_text())
    size, gap = cfg["tags"]["size_mm"] / 1000, cfg["tags"]["strip_center_gap_mm"] / 1000
    cal = np.load(ref / "color_intrinsics.npz")
    K, dist = cal["K"], cal["dist"]
    faces = load_tagmap()
    tm = TowerModel.from_yaml()
    half = tm.blocks_per_layer * tm.width / 2

    det = make_detector()
    T_mt0 = tower_from_faces(faces, 0.0)
    views = []
    for f in sorted(ref.glob("snap_*.png")):
        bgr = cv2.imread(str(f))
        dets = detect(det, bgr)
        if len(dets) < MIN_TAGS:
            continue
        T_cm, rms = camera_pose(dets, faces, K, dist, size, gap)
        if T_cm is None or rms > 2.0:
            continue
        T_ct = T_cm @ T_mt0
        views.append({"name": f.stem, "bgr": bgr, "seam": seam_map(bgr), "T_ct": T_ct,
                      "K": K, "dist": dist, "faces": visible_faces(T_ct, half), "rms": rms})
    print(f"{len(views)} usable views")
    for v in views:
        print(f"  {v['name']}: tag reproj {v['rms']:.2f}px, faces {v['faces']}")

    # coarse grid search over z0 (layer-1 bottom height rel. to strip center) and h
    best = None
    for z0 in np.arange(0.0, 0.12, 0.0005):
        for h in np.arange(0.0222, 0.0238, 0.0001):
            s = score(views, tm, half, z0, h)
            if best is None or s > best[0]:
                best = (s, z0, h)
    s, z0, h = best
    # fine search
    for z0f in np.arange(z0 - 0.001, z0 + 0.001, 0.0001):
        for hf in np.arange(h - 0.0001, h + 0.0001, 0.00002):
            sf = score(views, tm, half, z0f, hf)
            if sf > best[0]:
                best = (sf, z0f, hf)
    s, z0, h = best
    print(f"\nbest: z0 = {z0 * 1000:.1f} mm above strip center, block height h = {h * 1000:.2f} mm,"
          f" tower top = {(z0 + tm.num_layers * h) * 1000:.1f} mm, score {s:.2f}")
    # robustness: score profile along z0 at best h
    prof = [(z, score(views, tm, half, z, h)) for z in np.arange(z0 - 0.015, z0 + 0.0151, 0.0025)]
    print("z0 profile:", " ".join(f"{z * 1000:.1f}:{sc:.1f}" for z, sc in prof))

    T_mt = tower_from_faces(faces, z0)
    tm_fit = TowerModel(tm.length, tm.width, h, tm.num_layers, tm.blocks_per_layer)
    out_dir = ROOT / "data" / "ref_overlay"
    out_dir.mkdir(exist_ok=True)
    for v in views:
        vis = v["bgr"].copy()
        T_ct = v["T_ct"] @ np.linalg.inv(T_mt0) @ T_mt
        draw_overlay(vis, T_ct, K, dist, tm_fit, tm_fit.all_nominal_poses, (0, 255, 0))
        cv2.imwrite(str(out_dir / f"{v['name']}.jpg"), vis)

    rv, tv = to_rt(T_mt)
    out = {"T_tagmap_tower": np.round(T_mt, 6).tolist(),
           "block_height_mm": round(float(h * 1000), 3),
           "z0_mm_above_strip_center": round(float(z0 * 1000), 2),
           "n_views": len(views)}
    (ROOT / "config" / "tower_ref.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print("wrote config/tower_ref.yaml and overlays in data/ref_overlay/")


if __name__ == "__main__":
    main()
