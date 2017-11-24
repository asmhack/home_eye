# Python 2/3 compatibility
from __future__ import print_function
import sys

import imutils

PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2

import os
import itertools as it
from contextlib import contextmanager


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f s" % ((clock()-start)))

class VideoWriter:
    def __init__(self, capture, out_file="/tmp/video.mp4", codec="X264"):
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out_file = out_file
        self.capture = capture
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.writer = cv2.VideoWriter(self.out_file, self.fourcc, fps, (w, h), True)

        self.writer.write(frame)

    def __del__(self):
        self.writer.release()


def prepare_frame(frame):
    frame = imutils.resize(frame, width=640)

    return frame