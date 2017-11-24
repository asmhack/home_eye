import cv2
import numpy as np
from datetime import datetime

from imutils.face_utils import rect_to_bb

HUD_COLOR = (0, 255, 0)
HUD_ROI = (0, 165, 255)
FONT_SIZE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def mark_roi(frame, rect, label=""):
    mask = np.zeros_like(frame, dtype=np.uint8)

    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(mask, (x, y), (x + w, y + h), HUD_ROI, thickness=1)
    draw_str(mask, label, (x, y - 10))

    apply_hud(frame, mask)


def apply_hud(frame, mask):
    cv2.addWeighted(frame, 1, mask, 0.5, 1, dst=frame)


def draw_str(frame, txt, target, thickness=1, shadow=False):
    x, y = target
    cv2.putText(frame, txt, (x, y), FONT, FONT_SIZE, HUD_COLOR, lineType=cv2.LINE_AA)
    if shadow:
        cv2.putText(frame, txt, (x + 1, y + 1), FONT, FONT_SIZE, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)


def draw_hud(frame, cam_name='', frame_idx=None):
    mask = np.zeros_like(frame, dtype=np.uint8)
    rows, cols = frame.shape[:2]

    margin = 40
    x1, y1 = margin, margin
    x2, y2 = cols - margin, rows - margin

    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    if cam_name:
        draw_str(mask, cam_name, (x1 - 20, y1 - 10), shadow=True)

    draw_str(mask, "{0} - {1}".format(t, frame_idx), (x1 - 20, y2 + 20), shadow=True)

    apply_hud(frame, mask)
