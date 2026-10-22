"""Camera localization from the tag map, and the tag-map -> tower frame transform."""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .tagmap import NUM_FACES, tag_corners_in_face, to_T

ROOT = Path(__file__).resolve().parent.parent

# tower axes expressed in map (= face 0) axes: x_t = x_m, y_t = -z_m (away from front), z_t = y_m (up)
R_MAP_TOWER = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)


def load_tagmap(path=ROOT / "config" / "tagmap.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    return [np.array(cfg["T_map_face"][k]) for k in range(NUM_FACES)]


def tag_corners_in_map(tid, faces, size, gap):
    p = tag_corners_in_face(tid, size, gap)
    T = faces[tid // 2]
    return p @ T[:3, :3].T + T[:3, 3]


def camera_pose(dets, faces, K, dist, size, gap):
    """T_cam_map from all detected strip tags (None if no tags). Returns (T, rms_px)."""
    if not dets:
        return None, None
    obj = np.concatenate([tag_corners_in_map(t, faces, size, gap) for t in dets])
    img = np.concatenate([dets[t] for t in dets])
    # initialise from one tag (planar, robust), then refine on all corners
    t0 = max(dets, key=lambda t: cv2.contourArea(dets[t].astype(np.float32)))
    ok, rv, tv = cv2.solvePnP(tag_corners_in_map(t0, faces, size, gap), dets[t0], K, dist,
                              flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, rv, tv, useExtrinsicGuess=True,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return to_T(rv, tv), rms


def tower_from_faces(faces, z0_map):
    """T_map_tower: horizontal center/yaw from the 4 strip planes, origin height z0_map
    (map y coordinate of the layer-1 bottom face)."""
    c = np.mean([T[:3, 3] for T in faces], axis=0)
    # yaw: average the face x-axes rotated into face-0 orientation
    angs = []
    for k, T in enumerate(faces):
        xk = T[:3, 0]
        a = np.arctan2(-xk[2], xk[0]) - k * np.pi / 2  # face k is rotated by +90deg*k about y
        angs.append(np.arctan2(np.sin(a), np.cos(a)))
    yaw = float(np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs))))
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    T = np.eye(4)
    T[:3, :3] = Ry @ R_MAP_TOWER
    T[:3, 3] = [c[0], z0_map, c[2]]
    return T
