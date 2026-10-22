"""Generate a printable A4 PDF of AprilTag 36h11 strips for the tower jig.

One strip per jig box side face (125 x 50 mm): two tags side by side.
Face k (0=front, then clockwise seen from above) carries ids 2k (left), 2k+1 (right).
Print at 100% scale (no "fit to page"), verify the 100 mm scale bar,
then measure the black square side and put it in the config.
"""
import argparse
import os
import tempfile

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

TAG_MM = 36.0          # black square side
CENTER_GAP_MM = 72.0   # distance between the two tag centers in a strip
STRIP_W_MM = 118.0
STRIP_H_MM = 46.0
FACE_NAMES = ["FRONT", "RIGHT", "BACK", "LEFT"]


def tag_png(tag_id, path, px=800):
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    cv2.imwrite(path, cv2.aruco.drawMarker(d, tag_id, px, borderBits=1))


def draw_strip(c, face, cx, cy, tmp):
    w, h, s = STRIP_W_MM * mm, STRIP_H_MM * mm, TAG_MM * mm
    c.setDash(2, 2); c.setLineWidth(0.3)
    c.rect(cx - w / 2, cy - h / 2, w, h)
    c.setDash()
    for k, dx in enumerate((-CENTER_GAP_MM / 2, CENTER_GAP_MM / 2)):
        tag_id = 2 * face + k
        png = os.path.join(tmp, f"tag_{tag_id}.png")
        tag_png(tag_id, png)
        c.drawImage(png, cx + dx * mm - s / 2, cy - s / 2, s, s)
    c.setFont("Helvetica", 9)
    c.drawCentredString(cx, cy - h / 2 - 4 * mm,
                        f"face {face} {FACE_NAMES[face]}: ids {2 * face} (left), {2 * face + 1} (right) "
                        f"| strip {STRIP_W_MM:.0f}x{STRIP_H_MM:.0f} mm, tag {TAG_MM:.0f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tags/apriltag_36h11_strips.pdf")
    args = ap.parse_args()
    W, H = A4
    c = canvas.Canvas(args.out, pagesize=A4)
    with tempfile.TemporaryDirectory() as tmp:
        for face in range(4):
            draw_strip(c, face, W / 2, H - 50 * mm - face * 60 * mm, tmp)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(15 * mm, H - 12 * mm,
                 "JENGA_6DOF AprilTag strips - PRINT AT 100% (no fit-to-page). Cut on dashed line.")
    c.setLineWidth(1)
    x0, y0 = 15 * mm, 12 * mm
    c.line(x0, y0, x0 + 100 * mm, y0)
    for k in range(11):
        c.line(x0 + k * 10 * mm, y0, x0 + k * 10 * mm, y0 + (4 if k % 5 == 0 else 2) * mm)
    c.setFont("Helvetica", 8)
    c.drawString(x0 + 102 * mm, y0, "100 mm scale check")
    c.save()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
