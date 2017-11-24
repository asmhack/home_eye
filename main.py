# coding: utf-8
import cv2
import threading
from time import time, sleep
from pprint import pprint
from scipy.spatial import distance
import dlib
import imutils
import numpy as np
import os
import re
from imutils.face_utils import rect_to_bb, shape_to_np, FACIAL_LANDMARKS_IDXS
from imutils.video import VideoStream, FPS
from common import clock, VideoWriter, Timer, prepare_frame
from hud import draw_hud, mark_roi

if __name__ == '__main__':

    # TODO read from args
    cam_num = 0
    detection_frequency = 4


    detector = dlib.get_frontal_face_detector()

    vs = VideoStream(usePiCamera=False).start()
    cv2.namedWindow('preview', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    frame_counter = 0
    while True:
        if frame_counter > 100000:
            frame_counter = 0

        frame = vs.read()
        frame = prepare_frame(frame)

        frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Every X frames, we will have to determine which faces
        # are present in the frame
        if (frame_counter % detection_frequency) == 0:

            faces = detector(gray, 1)

            for rect in faces:
                mark_roi(frame, rect)
                (x, y, w, h) = rect_to_bb(rect)


        draw_hud(frame, '#{}'.format(cam_num), frame_idx=frame_counter)
        cv2.imshow('preview', frame)

        if cv2.waitKey(1) == 27:
            break