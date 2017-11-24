import cv2
import numpy as np
from datetime import datetime

HUD_COLOR = (0, 255, 0)
HUD_CLIGHT = (200, 255, 200)
HUD_RED = (0, 0, 255)
FONT_SIZE = 1
FONT = cv2.FONT_HERSHEY_PLAIN


class HudStatus:
    def __init__(self, blink_rate=5):
        self.status_msg = ""
        self.frame_idx = 0
        self.counter = 0
        self.blink_rate = blink_rate

    def set_status(self, msg, count=50):
        self.status_msg = msg
        self.counter = count

    def update(self):
        self.counter -= 1
        self.frame_idx += 1
        if self.counter <= 0:
            self.status_msg = ""

    def get_status(self):
        if self.frame_idx % self.blink_rate == 0:
            return ""
        else:
            return self.status_msg


def canny_filter(frame, sigma=0.33):
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(frame, lower, upper)
    return edged


def mark_rois(frame, rois, label="person detected"):
    mask = np.zeros_like(frame, dtype=np.uint8)

    for (x, y, w, h) in rois:
        cv2.rectangle(mask, (x, y), (x + w, y + h), HUD_CLIGHT, thickness=1)
        cv2.rectangle(mask, (x - 1, y - 1), (x + w + 1, y + h + 1), HUD_COLOR, thickness=1)
        cv2.rectangle(mask, (x + 1, y + 1), (x + w - 1, y + h - 1), HUD_COLOR, thickness=1)
        draw_str(mask, label, (x, y - 10))

    apply_hud(frame, mask)


def apply_hud(frame, mask):
    cv2.addWeighted(frame, 1, mask, 0.5, 1, dst=frame)

def draw_str(frame, txt, target):
    x, y = target
    cv2.putText(frame, txt, (x, y), FONT, FONT_SIZE, HUD_COLOR, lineType=cv2.LINE_AA)
    cv2.putText(frame, txt, (x+1, y+1), FONT, FONT_SIZE, (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)


def get_hud(frame, fps, idx=None):
    mask = np.zeros_like(frame, dtype=np.uint8)
    rows, cols = frame.shape[:2]

    margin = 40
    x1, y1 = margin, margin
    x2, y2 = cols - margin, rows - margin

    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    draw_str(mask, fps, (x1 + 20, y1 - 10))

    draw_str(mask, "{0} - {1}".format(t, idx), (x1 + 20, y2 + 20))

    apply_hud(frame, mask)