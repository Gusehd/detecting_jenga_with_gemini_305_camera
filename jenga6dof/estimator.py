"""Block 6DOF deviation estimation from one RGB-D frame.

Pipeline for target block (n, m):
  1. camera pose from the tag map (coarse: mm-level errors, mostly along the
     viewing direction and in tilt)
  2. local refinement of that pose against the exterior faces of the
     neighbouring layers (n-2..n+2, excluding n), depth only, pivoting about the
     target block: cancels tag-pose error and depth bias around the target
  3. fit the target block delta (in its own frame, pivot = block center) to its
     visible exterior faces (depth, point-to-plane) and their outlines (dark
     seams in the image). Roll/pitch/dz are held by tight priors (blocks rest
     flat on the layer below) but are still reported.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from .localize import camera_pose, load_tagmap
from .tagmap import detect, inv, make_detector
from .tower_model import TowerModel

ROOT = Path(__file__).resolve().parent.parent

SIGMA_DEPTH = 0.003   # m, per-point depth noise
MAX_DEPTH_PTS = 2000
MAX_EDGE_PTS = 400
# priors, order (tx, ty, tz [m], rx, ry, rz [rad]) in the pivot frame
PRIOR_CAMERA = np.array([0.006, 0.006, 0.006, np.radians(2), np.radians(2), np.radians(2)])
# block depth association (max point-to-plane dist, in-plane margin): the negative margin
# drops a 5 mm border of each face, where stereo depth bleeds into the surface behind
BLOCK_ASSIGN = ((0.025, -0.005), (0.008, -0.005), (0.005, -0.005))
# Empirical 1-sigma floors (dx, dy, dz [mm], roll, pitch, yaw [deg]) per view type, from
# hand-held tests at 25-35 cm. The formal (Jacobian) sigma is far too optimistic on its own.
STD_FLOOR_END_ONLY = np.array([1.5, 3.0, 1.5, 1.0, 1.5, 5.0])
STD_FLOOR_WITH_SIDE = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 2.0])
STD_FLOOR_RECESSED = np.array([4.0, 5.0, 2.0, 2.0, 3.0, 10.0])  # end face sunk into the tower
RECESS_MM = 2.0
# extra uncertainty of the absolute (reference-frame) estimate from the hand-held tag pose
STD_TAG_ABS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
PRIOR_BLOCK = np.array([0.04, 0.003, 0.001, np.radians(1), np.radians(1), np.radians(10)])


def se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(xi[3:]).as_matrix()
    T[:3, 3] = xi[:3]
    return T


@dataclass
class Rect:
    """Face rectangle: center c, outward normal n, in-plane axes u, v, half extents a, b."""
    c: np.ndarray
    n: np.ndarray
    u: np.ndarray
    v: np.ndarray
    a: float
    b: float

    def transformed(self, T):
        Rm = T[:3, :3]
        return Rect(Rm @ self.c + T[:3, 3], Rm @ self.n, Rm @ self.u, Rm @ self.v, self.a, self.b)

    def outline(self, step=0.002, with_tangent=False, inset=0.0):
        """Points along the 4 edges (optionally excluding `inset` at the corners)."""
        pts, tans = [], []
        for base, along, half in [(self.c + self.a * self.u, self.v, self.b),
                                  (self.c - self.a * self.u, self.v, self.b),
                                  (self.c + self.b * self.v, self.u, self.a),
                                  (self.c - self.b * self.v, self.u, self.a)]:
            s = np.arange(-half + inset, half - inset + 1e-9, step)
            pts.append(base + s[:, None] * along)
            tans.append(np.tile(along, (len(s), 1)))
        if with_tangent:
            return np.concatenate(pts), np.concatenate(tans)
        return np.concatenate(pts)


def block_faces_local(tm):
    """6 faces of a block in its local frame (x = long axis, y = width, z = up)."""
    L, W, H = tm.length / 2, tm.width / 2, tm.height / 2
    ex, ey, ez = np.eye(3)
    return {
        "+x": Rect(L * ex, ex, ey, ez, W, H), "-x": Rect(-L * ex, -ex, ey, ez, W, H),
        "+y": Rect(W * ey, ey, ex, ez, L, H), "-y": Rect(-W * ey, -ey, ex, ez, L, H),
        "+z": Rect(H * ez, ez, ex, ey, L, W), "-z": Rect(-H * ez, -ez, ex, ey, L, W),
    }


def exterior_faces(tm, n, m):
    """Names of faces of block (n, m) on the tower exterior (nominal state)."""
    # both ends are always exterior. Side faces: odd layers have block y = tower y (m=1 at -y);
    # even layers have block y = tower -x (m=1 at tower -x, i.e. block +y).
    names = ["+x", "-x"]
    if m == 1:
        names.append("-y" if n % 2 == 1 else "+y")
    if m == tm.blocks_per_layer:
        names.append("+y" if n % 2 == 1 else "-y")
    if n == tm.num_layers:
        names.append("+z")
    return names


def facing_camera(rect_cam, min_cos=0.2):
    c = rect_cam.c
    return float(np.dot(rect_cam.n, -c)) > min_cos * np.linalg.norm(c)


class FrameData:
    def __init__(self, bgr, depth, K, dist, rays):
        self.bgr, self.depth, self.K, self.dist, self.rays = bgr, depth, K, dist, rays
        self._grad = {}

    def grad(self, sigma):
        """Smoothed image gradient (gx, gy), normalised by a robust scale."""
        if sigma not in self._grad:
            g = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), sigma)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            self._grad[sigma] = (gx, gy)
        return self._grad[sigma]

    def points_in_roi(self, uv_poly, stride=2):
        """Back-project valid depth pixels inside the bbox of uv_poly -> Nx3 (camera frame)."""
        H, W = self.depth.shape
        x0, y0 = np.clip(np.floor(uv_poly.min(0)).astype(int) - 4, 0, [W - 1, H - 1])
        x1, y1 = np.clip(np.ceil(uv_poly.max(0)).astype(int) + 4, 0, [W - 1, H - 1])
        d = self.depth[y0:y1:stride, x0:x1:stride]
        r = self.rays[y0:y1:stride, x0:x1:stride]
        ok = d > 0
        z = d[ok]
        return np.c_[r[ok] * z[:, None], z]

    def project(self, P):
        uv, _ = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    @staticmethod
    def _bilinear(img, uv):
        H, W = img.shape
        x = np.clip(uv[:, 0], 0, W - 1.001)
        y = np.clip(uv[:, 1], 0, H - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)

    def edge_strength(self, uv, uv_t, sigma, scale):
        """|gradient . edge normal| at uv (normal from the projected tangent uv_t), in [0, 1]."""
        gx, gy = self.grad(sigma)
        t = uv_t - uv
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        g = np.abs(self._bilinear(gx, uv) * -t[:, 1] + self._bilinear(gy, uv) * t[:, 0])
        H, W = gx.shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return np.where(inside, np.minimum(g / scale, 1.0), 0.0)


def _associate(P, rects, max_dist, margin):
    """Nearest-rectangle index per point (-1 if none) by point-to-plane distance."""
    best = np.full(len(P), -1)
    best_d = np.full(len(P), np.inf)
    for i, r in enumerate(rects):
        q = P - r.c
        d = q @ r.n
        s, t = q @ r.u, q @ r.v
        ok = (np.abs(s) < r.a + margin) & (np.abs(t) < r.b + margin) \
            & (np.abs(d) < max_dist) & (np.abs(d) < np.abs(best_d))
        best[ok], best_d[ok] = i, d[ok]
    return best


def _subsample(a, k, rng):
    return a if len(a) <= k else a[rng.choice(len(a), k, replace=False)]


EDGE_SIGMA_RES = 0.35   # residual scale of (1 - edge strength)
EDGE_SIGMAS = (3.0, 1.2)  # image blur (px), coarse -> fine
HUBER_K = 1.5


def _huber_w(r):
    a = np.abs(r)
    return np.where(a <= HUBER_K, 1.0, np.sqrt(HUBER_K / np.maximum(a, 1e-9)))


def fit_rects(fd, rects_model, T_cam_model, pivot, prior_sigma, use_edges,
              assign_schedule=((0.015, 0.002), (0.008, 0.002), (0.006, 0.002)), seed=0):
    """Fit xi so that T_cam_model' = T_cam_model @ pivot @ exp(xi) @ pivot^-1.

    rects_model are in the model frame; only those facing the camera are used.
    Data terms are robustified by IRLS (Huber); priors are kept quadratic.
    Returns (xi, info) or (None, info).
    """
    rng = np.random.default_rng(seed)
    piv_inv = inv(pivot)

    def compose(x):
        return T_cam_model @ pivot @ se3_exp(x) @ piv_inv

    vis = [i for i, r in enumerate(rects_model) if facing_camera(r.transformed(T_cam_model))]
    if not vis:
        return None, {"reason": "no visible faces"}
    rects = [rects_model[i] for i in vis]
    outline_cam = np.concatenate([r.transformed(T_cam_model).outline(0.01) for r in rects])
    P_cam = fd.points_in_roi(fd.project(outline_cam))

    edge_pts = edge_tan = np.zeros((0, 3))
    if use_edges:
        ep, et = zip(*[r.outline(0.002, with_tangent=True, inset=0.003) for r in rects])
        edge_pts, edge_tan = np.concatenate(ep), np.concatenate(et)
        sel = _subsample(np.arange(len(edge_pts)), MAX_EDGE_PTS, rng)
        edge_pts, edge_tan = edge_pts[sel], edge_tan[sel]
        uv0 = fd.project(edge_pts @ T_cam_model[:3, :3].T + T_cam_model[:3, 3])
        lo, hi = uv0.min(0) - 40, uv0.max(0) + 40
    state = {"P": np.zeros((0, 3)), "n": np.zeros((0, 3)), "c": np.zeros((0, 3)),
             "wd": np.zeros(0), "we": np.ones(len(edge_pts)), "sigma": EDGE_SIGMAS[0],
             "scale": 1.0}

    def data_residuals(x):
        Tx = compose(x)
        Txi = inv(Tx)
        pm = state["P"] @ Txi[:3, :3].T + Txi[:3, 3]
        r_d = np.einsum("ij,ij->i", pm - state["c"], state["n"]) / SIGMA_DEPTH
        if not use_edges:
            return r_d, np.zeros(0)
        pe = edge_pts @ Tx[:3, :3].T + Tx[:3, 3]
        uv = fd.project(np.vstack([pe, pe + 0.002 * (edge_tan @ Tx[:3, :3].T)]))
        k = len(pe)
        g = fd.edge_strength(uv[:k], uv[k:], state["sigma"], state["scale"])
        return r_d, (1.0 - g) / EDGE_SIGMA_RES

    def residuals(x):
        r_d, r_e = data_residuals(x)
        return np.concatenate([state["wd"] * r_d, state["we"] * r_e, x / prior_sigma])

    xi = np.zeros(6)
    stages = [(md, mg, EDGE_SIGMAS[min(i, len(EDGE_SIGMAS) - 1)])
              for i, (md, mg) in enumerate(assign_schedule)]
    for max_dist, margin, sigma in stages:
        Ti = inv(compose(xi))
        P_model = P_cam @ Ti[:3, :3].T + Ti[:3, 3]
        idx = _associate(P_model, rects, max_dist, margin)
        keep = _subsample(np.nonzero(idx >= 0)[0], MAX_DEPTH_PTS, rng)
        state["P"] = P_cam[keep]
        state["idx"] = idx[keep]
        state["n"] = np.stack([rects[i].n for i in idx[keep]]) if len(keep) else np.zeros((0, 3))
        state["c"] = np.stack([rects[i].c for i in idx[keep]]) if len(keep) else np.zeros((0, 3))
        state["wd"] = np.ones(len(keep))
        if use_edges:
            state["sigma"] = sigma
            gx, gy = fd.grad(sigma)
            H, W = gx.shape
            x0, y0 = np.clip(lo.astype(int), 0, [W - 1, H - 1])
            x1, y1 = np.clip(hi.astype(int), 1, [W, H])
            mag = np.hypot(gx[y0:y1, x0:x1], gy[y0:y1, x0:x1])
            state["scale"] = max(1.0, float(np.percentile(mag, 97)))
            state["we"] = np.ones(len(edge_pts))
        for _ in range(2):  # IRLS
            sol = least_squares(residuals, xi, diff_step=1e-3, max_nfev=40)
            xi = sol.x
            r_d, r_e = data_residuals(xi)
            state["wd"] = _huber_w(r_d)
            if use_edges:
                state["we"] = _huber_w(r_e)

    info = {"visible": vis, "n_depth": int(len(state["P"]))}
    try:  # formal 1-sigma of xi (data + priors), from the final Jacobian
        info["xi_std"] = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)))
    except np.linalg.LinAlgError:
        info["xi_std"] = np.full(6, np.nan)
    r_d, r_e = data_residuals(xi)
    # per-face median signed point-to-plane residual (m), model frame
    info["face_resid"] = {int(vis[f]): float(np.median(r_d[state["idx"] == f]) * SIGMA_DEPTH)
                          for f in np.unique(state["idx"]) if np.sum(state["idx"] == f) >= 20}
    info["face_count"] = {int(vis[f]): int(np.sum(state["idx"] == f)) for f in np.unique(state["idx"])}
    if len(r_d):
        info["depth_med_mm"] = float(np.median(np.abs(r_d)) * SIGMA_DEPTH * 1000)
    if use_edges:
        info["edge_strength_med"] = float(np.median(1.0 - r_e * EDGE_SIGMA_RES))
    return xi, info


TOWER_FACE_NAMES = ["FRONT(id0,1)", "RIGHT(id2,3)", "BACK(id4,5)", "LEFT(id6,7)", "TOP"]


def tower_faces_showing(tm, n, m):
    """Tower faces (0..3, 4 = top) on which block (n, m) has an exterior face."""
    T = tm.nominal_pose(n, m)
    local = block_faces_local(tm)
    out = []
    for nm in exterior_faces(tm, n, m):
        nt = T[:3, :3] @ local[nm].n
        k = int(np.argmax(np.abs(nt)))
        out.append({(1, -1): 0, (0, 1): 1, (1, 1): 2, (0, -1): 3}.get((k, int(np.sign(nt[k]))), 4))
    return sorted(set(out))


class BlockEstimator:
    def __init__(self, K, dist, size_wh, use_edges=True):
        self.use_edges = use_edges
        cfg = yaml.safe_load((ROOT / "config" / "tower.yaml").read_text())
        ref = yaml.safe_load((ROOT / "config" / "tower_ref.yaml").read_text())
        self.tag_size = cfg["tags"]["size_mm"] / 1000
        self.tag_gap = cfg["tags"]["strip_center_gap_mm"] / 1000
        self.tm = TowerModel.from_yaml()
        self.tm.height = ref["block_height_mm"] / 1000
        self.T_mt = np.array(ref["T_tagmap_tower"])
        self.faces = load_tagmap()
        self.detector = make_detector()
        self.K, self.dist = K, dist
        W, H = size_wh
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
        self.rays = cv2.undistortPoints(pts, K, dist).reshape(H, W, 2)
        self.local = block_faces_local(self.tm)

    def neighbour_rects(self, n, window=2):
        """Exterior face rectangles (tower frame) of layers n-window..n+window except n."""
        rects = []
        for k in range(max(1, n - window), min(self.tm.num_layers, n + window) + 1):
            if k == n:
                continue
            for m in range(1, self.tm.blocks_per_layer + 1):
                T = self.tm.nominal_pose(k, m)
                rects += [self.local[nm].transformed(T) for nm in exterior_faces(self.tm, k, m)]
        return rects

    def camera_from_tags(self, bgr):
        dets = detect(self.detector, bgr)
        if not dets:
            return None, dets, None
        T_cm, rms = camera_pose(dets, self.faces, self.K, self.dist, self.tag_size, self.tag_gap)
        return (None if T_cm is None else T_cm @ self.T_mt), dets, rms

    def estimate(self, bgr, depth, n, m, T_ct=None):
        """T_ct: optional camera<-tower pose (e.g. from robot FK); otherwise from tags."""
        self.tm.check_index(n, m)
        out = {"n": n, "m": m, "ok": False}
        if T_ct is None:
            T_ct, dets, rms = self.camera_from_tags(bgr)
            out["tags"], out["tag_rms_px"] = sorted(dets), rms
            if T_ct is None:
                out["reason"] = "no tags visible"
                return out
        out["T_ct_tags"] = T_ct
        fd = FrameData(bgr, depth, self.K, self.dist, self.rays)
        T_tb = self.tm.nominal_pose(n, m)

        # 2. local camera refinement (depth only), pivot = target block nominal pose
        xi_c, info_c = fit_rects(fd, self.neighbour_rects(n), T_ct, T_tb, PRIOR_CAMERA,
                                 use_edges=False)
        if xi_c is None or info_c["n_depth"] < 200:
            out["reason"] = "neighbour layers not visible"
            return out
        T_ct = T_ct @ T_tb @ se3_exp(xi_c) @ inv(T_tb)
        out["cam_corr"] = xi_c
        out["info_local"] = info_c

        # 3. target block, pivot = block center (model frame = block frame)
        rects = [self.local[nm] for nm in exterior_faces(self.tm, n, m)]
        xi_b, info_b = fit_rects(fd, rects, T_ct @ T_tb, np.eye(4), PRIOR_BLOCK,
                                 use_edges=self.use_edges,
                                 assign_schedule=BLOCK_ASSIGN)
        if xi_b is None or info_b["n_depth"] < 50:
            out["reason"] = "target block faces not visible from this view"
            return out
        D = se3_exp(xi_b)
        names = exterior_faces(self.tm, n, m)
        # Offsets along face normals from depth alone: image edges of a protruding block
        # (front outline vs. where it enters the tower) can pull the fit toward zero.
        if info_b["face_resid"]:
            N = np.stack([self.local[names[k]].n for k in info_b["face_resid"]])
            r = np.array(list(info_b["face_resid"].values()))
            dt = np.linalg.lstsq(N, r, rcond=None)[0]
            Tc = np.eye(4)
            Tc[:3, 3] = dt
            D = D @ Tc
        sd = np.r_[info_b["xi_std"][:3] * 1000, np.degrees(info_b["xi_std"][3:])]
        used = [names[i] for i in info_b["visible"]]
        # end face offset along its outward normal (mm): negative = sunk into the tower
        end_offsets = [float(D[0, 3] * 1000 * (1 if nm == "+x" else -1))
                       for nm in used if nm in ("+x", "-x")]
        if any(o < -RECESS_MM for o in end_offsets) and not any(o > RECESS_MM for o in end_offsets) \
                and not any(nm in ("+y", "-y", "+z") for nm in used):
            floor = STD_FLOOR_RECESSED
        elif any(nm in ("+y", "-y", "+z") for nm in used):
            floor = STD_FLOOR_WITH_SIDE
        else:
            floor = STD_FLOOR_END_ONLY
        sd = np.maximum(sd, floor)
        # absolute delta w.r.t. the reference tower: the local correction xi_c is, up to tag
        # error, the displacement of the neighbouring layers themselves
        D_abs = se3_exp(xi_c) @ D
        out.update({
            "ok": True,
            "T_ct": T_ct,
            "T_tower_block_nominal": T_tb,
            "T_tower_block": T_tb @ D,
            "delta_block": D,
            # block frame: x = long axis, y = width, z = up
            "dxyz_mm": D[:3, 3] * 1000,
            "drpy_deg": R.from_matrix(D[:3, :3]).as_euler("xyz", degrees=True),
            "std_xyz_mm": sd[:3],
            "std_rpy_deg": sd[3:],
            "delta_abs": D_abs,
            "dxyz_abs_mm": D_abs[:3, 3] * 1000,
            "drpy_abs_deg": R.from_matrix(D_abs[:3, :3]).as_euler("xyz", degrees=True),
            "std_abs": np.hypot(sd, STD_TAG_ABS),
            "faces_used": used,
            "end_offsets_mm": end_offsets,
            "info_block": info_b,
        })
        return out
