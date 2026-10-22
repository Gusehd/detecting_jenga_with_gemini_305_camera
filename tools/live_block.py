"""M2: live 6DOF deviation of block (n, m) with a hand-held Gemini 305.

Usage: .venv/bin/python tools/live_block.py [--no-ros] [--no-edges] [--no-record]
Terminal commands:
  "n m"         select a block (e.g. "5 2"); clears the history
  "label TEXT"  tag the following records (e.g. "label pull5mm"); "label" alone clears it
  "c"           clear history,  "s" save a summary line,  "q" quit
Window keys: c = clear history, s = save summary, q/ESC = quit.

Every successful estimate (frame, depth, result) is recorded to data/live/<session>/
so the data used for the numbers on screen can be re-analysed later.

Overlay: green = nominal block, red = estimated block. The fused value combines the
last N estimates per DOF with weights from each estimate's uncertainty, so views
that see a DOF well (e.g. a side face for yaw) dominate it.
"""
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jenga6dof.camera import OrbbecRGBD  # noqa: E402
from jenga6dof.estimator import TOWER_FACE_NAMES, BlockEstimator, tower_faces_showing  # noqa: E402
from jenga6dof.ros_pub import BlockPublisher  # noqa: E402

HISTORY = 15
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def stdin_reader(q):
    for line in sys.stdin:
        q.put(line.strip())


def draw_block(img, est, T_ct, T_tb, color, thick):
    P = est.tm.block_corners() @ T_tb[:3, :3].T + T_tb[:3, 3]
    Pc = P @ T_ct[:3, :3].T + T_ct[:3, 3]
    if np.any(Pc[:, 2] <= 0.01):
        return
    uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), est.K, est.dist)
    uv = uv.reshape(-1, 2).astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thick, cv2.LINE_AA)


def put_lines(img, lines, org=(10, 28), dy=26, scale=0.62):
    for k, (text, color) in enumerate(lines):
        y = org[1] + k * dy
        cv2.putText(img, text, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


STD_FLOOR = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])  # mm, deg


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def fuse(hist, absolute=True):
    """Per-DOF weighted median of (dx, dy, dz, roll, pitch, yaw) -> (4x4 delta, values).

    absolute=True: w.r.t. the reference tower; False: relative to layers n-2..n+2.
    """
    if absolute:
        X = np.array([np.r_[h["dxyz_abs_mm"], h["drpy_abs_deg"]] for h in hist])
        S = np.array([h["std_abs"] for h in hist])
    else:
        X = np.array([np.r_[h["dxyz_mm"], h["drpy_deg"]] for h in hist])
        S = np.array([np.r_[h["std_xyz_mm"], h["std_rpy_deg"]] for h in hist])
    W = 1.0 / np.maximum(np.nan_to_num(S, nan=1e3), STD_FLOOR) ** 2
    v = np.array([weighted_median(X[:, k], W[:, k]) for k in range(6)])
    D = np.eye(4)
    D[:3, :3] = R.from_euler("xyz", v[3:], degrees=True).as_matrix()
    D[:3, 3] = v[:3] / 1000
    return D, v


class Worker(threading.Thread):
    """Runs the (slow) estimator on the most recent frame."""

    def __init__(self, est, rec_dir=None):
        super().__init__(daemon=True)
        self.est, self.lock = est, threading.Lock()
        self.rec_dir, self.rec_count, self.label = rec_dir, 0, ""
        self.job, self.result, self.busy = None, None, False
        self.event = threading.Event()

    def submit(self, bgr, depth, n, m):
        with self.lock:
            self.job = (bgr, depth, n, m)
        self.event.set()

    def run(self):
        while True:
            self.event.wait()
            self.event.clear()
            with self.lock:
                job, self.job = self.job, None
            if job is None:
                continue
            self.busy = True
            t = time.time()
            try:
                r = self.est.estimate(*job)
            except Exception as e:  # keep the live view running
                r = {"ok": False, "reason": f"error: {e}", "n": job[2], "m": job[3]}
            r["ms"] = (time.time() - t) * 1000
            if r["ok"] and self.rec_dir is not None:
                self.record(job[0], job[1], r)
            self.result = r
            self.busy = False

    def record(self, bgr, depth, r):
        name = f"f{self.rec_count:05d}"
        self.rec_count += 1
        cv2.imwrite(str(self.rec_dir / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.rec_dir / f"{name}_depth.png"),
                    np.clip(depth * 1e4, 0, 65535).astype(np.uint16))  # 0.1 mm units
        rec = {"frame": name, "t": time.time(), "label": self.label, "n": r["n"], "m": r["m"],
               "dxyz_mm": r["dxyz_mm"].round(3).tolist(), "drpy_deg": r["drpy_deg"].round(3).tolist(),
               "std_xyz_mm": r["std_xyz_mm"].round(3).tolist(),
               "std_rpy_deg": r["std_rpy_deg"].round(3).tolist(),
               "dxyz_abs_mm": r["dxyz_abs_mm"].round(3).tolist(),
               "drpy_abs_deg": r["drpy_abs_deg"].round(3).tolist(),
               "std_abs": r["std_abs"].round(3).tolist(),
               "end_offsets_mm": r["end_offsets_mm"],
               "faces": r["faces_used"], "tags": r.get("tags"),
               "T_ct": np.round(r["T_ct"], 6).tolist()}
        with open(self.rec_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    use_ros = "--no-ros" not in sys.argv
    out_dir = ROOT / "data" / "live" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = None if "--no-record" in sys.argv else out_dir
    cmd_q = queue.Queue()
    threading.Thread(target=stdin_reader, args=(cmd_q,), daemon=True).start()
    pub = BlockPublisher(enabled=use_ros)
    target, hist, last_seen = None, deque(maxlen=HISTORY), None
    print('Type "n m" + Enter to select a block (e.g. "5 2"), "q" to quit.')

    with OrbbecRGBD() as cam:
        est = BlockEstimator(cam.K, cam.dist, cam.size, use_edges="--no-edges" not in sys.argv)
        np.savez(out_dir / "intrinsics.npz", K=cam.K, dist=cam.dist, size=cam.size)
        worker = Worker(est, rec_dir)
        worker.start()
        print(f"recording to {out_dir}")

        def save_summary():
            if target is None:
                return
            summary = {"t": time.time(), "label": worker.label, "n": target[0], "m": target[1],
                       "samples": len(hist)}
            if hist:
                summary["fused_abs_dxyz_mm_rpy_deg"] = fuse(hist)[1].round(2).tolist()
                summary["fused_rel_dxyz_mm_rpy_deg"] = fuse(hist, absolute=False)[1].round(2).tolist()
            with open(out_dir / "summary.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            print("summary saved:", summary)
        running = True
        while running:
            while not cmd_q.empty():
                cmd = cmd_q.get()
                if cmd in ("q", "quit", "exit"):
                    running = False
                    break
                if cmd == "c":
                    hist.clear()
                    print("history cleared")
                    continue
                if cmd == "s":
                    save_summary()
                    continue
                if cmd.startswith("label"):
                    worker.label = cmd[5:].strip()
                    print(f"label = '{worker.label}'")
                    continue
                try:
                    n, m = map(int, cmd.replace(",", " ").split())
                    est.tm.check_index(n, m)
                    target, last_seen = (n, m), None
                    hist.clear()
                    faces = [TOWER_FACE_NAMES[i] for i in tower_faces_showing(est.tm, n, m)]
                    print(f"target ({n}, {m}); visible on: {', '.join(faces)}")
                except ValueError as e:
                    print(f"? '{cmd}': {e}" if "range" in str(e) else f"? '{cmd}': type 'n m'")
            out = cam.read()
            if out is None:
                continue
            bgr, depth = out
            if target is not None and not worker.busy:
                if worker.result is not None and (worker.result["n"], worker.result["m"]) == target:
                    r = worker.result
                    if r["ok"]:
                        hist.append(r)
                    last_seen = r
                    worker.result = None
                worker.submit(bgr.copy(), depth.copy(), *target)

            vis = bgr.copy()
            lines = []
            if target is None:
                lines.append(('type "n m" in the terminal to select a block', (0, 255, 255)))
            else:
                n, m = target
                T_ct, _, _ = est.camera_from_tags(bgr)
                T_tb = est.tm.nominal_pose(n, m)
                faces = ", ".join(TOWER_FACE_NAMES[i] for i in tower_faces_showing(est.tm, n, m))
                lines.append((f"block n={n} m={m}   (visible on {faces})", (255, 255, 255)))
                if T_ct is not None:
                    if last_seen is not None and last_seen.get("ok"):
                        T_ct = last_seen["T_ct"] @ np.linalg.inv(last_seen["T_ct_tags"]) @ T_ct
                    draw_block(vis, est, T_ct, T_tb, (0, 255, 0), 1)
                if hist:
                    D, v = fuse(hist)
                    _, vr = fuse(hist, absolute=False)
                    if T_ct is not None:
                        draw_block(vis, est, T_ct, T_tb @ D, (0, 0, 255), 2)
                    lines.append((f"ABS  (vs reference, {len(hist)}):  dx {v[0]:+5.1f}  dy {v[1]:+5.1f}  "
                                  f"dz {v[2]:+5.1f} mm   roll {v[3]:+4.1f}  pitch {v[4]:+4.1f}  "
                                  f"yaw {v[5]:+4.1f} deg", (0, 255, 0)))
                    lines.append((f"REL  (vs layers {n}+-2):  dx {vr[0]:+5.1f}  dy {vr[1]:+5.1f}  "
                                  f"dz {vr[2]:+5.1f} mm   roll {vr[3]:+4.1f}  pitch {vr[4]:+4.1f}  "
                                  f"yaw {vr[5]:+4.1f} deg", (0, 220, 255)))
                    pub.publish(n, m, T_tb, D, len(hist), D_rel=fuse(hist, absolute=False)[0])
                if last_seen is not None:
                    r = last_seen
                    if r["ok"]:
                        d, a = r["dxyz_mm"], r["drpy_deg"]
                        sd, sa = r["std_xyz_mm"], r["std_rpy_deg"]
                        lines.append((f"current REL:  dx {d[0]:+5.1f}  dy {d[1]:+5.1f}  dz {d[2]:+5.1f} mm"
                                      f"  r {a[0]:+4.1f} p {a[1]:+4.1f} y {a[2]:+4.1f} deg",
                                      (200, 200, 200)))
                        lines.append((f"current 1-sigma:  dx {sd[0]:.1f} dy {sd[1]:.1f} dz {sd[2]:.1f} mm  "
                                      f"r {sa[0]:.1f} p {sa[1]:.1f} y {sa[2]:.1f} deg | faces "
                                      f"{r['faces_used']} | {r['ms']:.0f} ms", (200, 200, 200)))
                    else:
                        lines.append((f"no estimate: {r.get('reason')}", (0, 128, 255)))
                        if not hist:
                            pub.publish(n, m, T_tb, None, 0)
            lab = f"   label: {worker.label}" if worker.label else ""
            lines.append((f"keys: c = clear, s = save summary, q = quit{lab}", (160, 160, 160)))
            put_lines(vis, lines)
            cv2.imshow("jenga block 6dof", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                hist.clear()
            if key == ord("s"):
                save_summary()
    cv2.destroyAllWindows()
    pub.close()


if __name__ == "__main__":
    main()
