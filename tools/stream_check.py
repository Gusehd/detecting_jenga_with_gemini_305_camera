"""M0: live RGB + aligned depth view, tag detection check, and snapshot saving.

Usage: stream_check.py [out_dir]   (default: data/)
Keys: s = save snapshot to out_dir, q/ESC = quit.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jenga6dof.camera import OrbbecRGBD  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"


def main():
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    with OrbbecRGBD() as cam:
        print("color size", cam.size)
        print("K =\n", np.round(cam.K, 2))
        print("dist =", np.round(cam.dist, 4))
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(DATA / "color_intrinsics.npz", K=cam.K, dist=cam.dist, size=cam.size)
        while True:
            out = cam.read()
            if out is None:
                continue
            bgr, depth = out
            vis = bgr.copy()
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            h, w = depth.shape
            dc = depth[h // 2, w // 2]
            valid = np.count_nonzero(depth) / depth.size
            cv2.putText(vis, f"center depth {dc * 1000:.0f} mm | valid {valid * 100:.0f}% | "
                             f"tags {[] if ids is None else sorted(ids.flatten().tolist())}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.drawMarker(vis, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            dvis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.clip(depth, 0, 1.0), alpha=255.0), cv2.COLORMAP_JET)
            dvis[depth == 0] = 0
            cv2.imshow("color", vis)
            cv2.imshow("depth (0-1 m)", dvis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(DATA / f"snap_{stamp}.png"), bgr)
                np.save(DATA / f"snap_{stamp}_depth.npy", depth)
                print("saved", stamp)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
