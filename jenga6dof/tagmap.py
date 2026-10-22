"""AprilTag strip geometry, per-frame detection, and tag-map bundle adjustment.

Strip (face) frame: origin at the midpoint between the two tag centers,
+x = right along the strip, +y = up, +z = out of the paper toward the viewer.
Face k carries ids 2k (left, x = -gap/2) and 2k+1 (right, x = +gap/2).
Map frame = face 0 frame.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

NUM_FACES = 4


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)


def tag_corners_in_face(tag_id, size, gap):
    """4x3 corners (aruco order TL, TR, BR, BL) of a tag in its face frame."""
    cx = -gap / 2 if tag_id % 2 == 0 else gap / 2
    h = size / 2
    return np.array([[cx - h, h, 0], [cx + h, h, 0], [cx + h, -h, 0], [cx - h, -h, 0]])


def detect(detector, bgr):
    """Returns {tag_id: 4x2 corners} for ids belonging to the strips."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(i): c[0].astype(np.float64) for i, c in zip(ids.flatten(), corners)
            if i < 2 * NUM_FACES}


def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.ravel(rvec)).as_matrix()
    T[:3, 3] = np.ravel(tvec)
    return T


def to_rt(T):
    return R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3].copy()


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def face_pose(dets, face, K, dist, size, gap):
    """T_cam_face from the visible tags of one face (None if none visible)."""
    obj, img = [], []
    for tid in (2 * face, 2 * face + 1):
        if tid in dets:
            obj.append(tag_corners_in_face(tid, size, gap))
            img.append(dets[tid])
    if not obj:
        return None
    obj, img = np.concatenate(obj), np.concatenate(img)
    # all points are coplanar (z = 0): IPPE, then iterative refinement
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, rv, tv, useExtrinsicGuess=True,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    return to_T(rv, tv) if ok else None


def _average_T(Ts):
    rot = R.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.median(np.stack([T[:3, 3] for T in Ts]), axis=0)
    return T


def solve_tagmap(frames, K, dist, size, gap):
    """Bundle-adjust face poses in the map (face 0) frame.

    frames: list of {tag_id: 4x2 corners}. Returns (T_map_face list, info dict).
    """
    # --- init: chain pairwise relative poses through co-visible faces
    per_frame = [{f: face_pose(d, f, K, dist, size, gap) for f in range(NUM_FACES)}
                 for d in frames]
    T_map_face = {0: np.eye(4)}
    for _ in range(NUM_FACES):
        for f in range(NUM_FACES):
            if f in T_map_face:
                continue
            rels = []
            for pf in per_frame:
                for g, Tg in T_map_face.items():
                    if pf.get(g) is not None and pf.get(f) is not None:
                        rels.append(Tg @ inv(pf[g]) @ pf[f])
            if rels:
                T_map_face[f] = _average_T(rels)
    missing = [f for f in range(NUM_FACES) if f not in T_map_face]
    if missing:
        raise RuntimeError(f"faces {missing} are never co-visible with the others")

    # frames usable for BA: at least one face pose
    obs, cam_init = [], []
    for i, (d, pf) in enumerate(zip(frames, per_frame)):
        Ts = [T_map_face[f] @ inv(pf[f]) for f in range(NUM_FACES) if pf[f] is not None]
        if not Ts:
            continue
        cam_init.append(inv(_average_T(Ts)))  # T_cam_map
        obs.append(d)

    # --- bundle adjustment: params = faces 1..3 (6 each) + cameras (6 each)
    def pack():
        p = []
        for f in range(1, NUM_FACES):
            p += list(np.concatenate(to_rt(T_map_face[f])))
        for T in cam_init:
            p += list(np.concatenate(to_rt(T)))
        return np.array(p)

    def unpack(p):
        faces = [np.eye(4)] + [to_T(p[6 * k:6 * k + 3], p[6 * k + 3:6 * k + 6])
                               for k in range(NUM_FACES - 1)]
        o = 6 * (NUM_FACES - 1)
        cams = [to_T(p[o + 6 * i:o + 6 * i + 3], p[o + 6 * i + 3:o + 6 * i + 6])
                for i in range(len(obs))]
        return faces, cams

    def residuals(p):
        faces, cams = unpack(p)
        res = []
        for d, Tcm in zip(obs, cams):
            for tid, uv in d.items():
                f = tid // 2
                pts_face = tag_corners_in_face(tid, size, gap)
                T = Tcm @ faces[f]
                rv, tv = to_rt(T)
                proj, _ = cv2.projectPoints(pts_face, rv, tv, K, dist)
                res.append((proj.reshape(-1, 2) - uv).ravel())
        return np.concatenate(res)

    p0 = pack()
    r0 = residuals(p0)
    sol = least_squares(residuals, p0, loss="huber", f_scale=1.0, x_scale="jac")
    faces, cams = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    info = {
        "n_frames": len(obs),
        "rms_init_px": float(np.sqrt(np.mean(r0 ** 2))),
        "rms_px": float(np.sqrt(np.mean(r ** 2))),
        "max_px": float(np.max(np.linalg.norm(r, axis=1))),
        "T_cam_map": cams,
    }
    return faces, info
