"""M1-1: capture frames for tag-map calibration.

Walk slowly around the tower. A frame is auto-saved when tags from two
adjacent faces are visible at once (corner views). The overlay shows how many
frames each corner (face pair) has; aim for >= 10 per corner.

Keys: space = force-save current frame, q/ESC = quit.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jenga6dof.camera import OrbbecRGBD  # noqa: E402

CORNERS = [(0, 1), (1, 2), (2, 3), (3, 0)]
TARGET_PER_CORNER = 10
MIN_INTERVAL_S = 0.7


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "calib" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    counts = {c: 0 for c in CORNERS}
    n_saved, last_save = 0, 0.0
    with OrbbecRGBD() as cam:
        np.savez(out_dir / "intrinsics.npz", K=cam.K, dist=cam.dist, size=cam.size)
        while True:
            out = cam.read()
            if out is None:
                continue
            bgr, depth = out
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            id_list = [] if ids is None else sorted(int(i) for i in ids.flatten() if i < 8)
            faces = {i // 2 for i in id_list}
            seen = [c for c in CORNERS if c[0] in faces and c[1] in faces]

            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            if (seen and now - last_save > MIN_INTERVAL_S) or key == ord(" "):
                name = f"f{n_saved:04d}"
                cv2.imwrite(str(out_dir / f"{name}.png"), bgr)
                np.save(out_dir / f"{name}_depth.npy", depth)
                for c in seen:
                    counts[c] += 1
                n_saved, last_save = n_saved + 1, now

            vis = bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            lines = [f"saved {n_saved}  tags {id_list}"]
            for c in CORNERS:
                ok = counts[c] >= TARGET_PER_CORNER
                lines.append(f"corner face{c[0]}-face{c[1]}: {counts[c]}/{TARGET_PER_CORNER}"
                             + ("  OK" if ok else ""))
            for k, text in enumerate(lines):
                color = (0, 255, 0) if k == 0 or "OK" in text else (0, 200, 255)
                cv2.putText(vis, text, (10, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 4)
                cv2.putText(vis, text, (10, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("calib capture", vis)
            if key in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    print(f"saved {n_saved} frames to {out_dir}")
    print("per corner:", counts)


if __name__ == "__main__":
    main()
