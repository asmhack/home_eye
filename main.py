# coding: utf-8
import argparse

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
from face import Face
from hud import draw_hud, draw_roi

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-n", "--camera_name", help="Camera name", default='0')
    arg_parser.add_argument("-f", "--detection_frequency", help="How often detect faces on video stream", default='4')

    args = vars(arg_parser.parse_args())

    faces_on_frame = []

    detection_frequency = int(args['detection_frequency'])

    detector = dlib.get_frontal_face_detector()

    vs = VideoStream(usePiCamera=False).start()
    cv2.namedWindow('preview', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    mapped_names = {}
    frame_counter = 0
    currentFaceID = len(mapped_names)

    while True:
        if frame_counter > 100000:
            frame_counter = 0

        frame = vs.read()
        frame = prepare_frame(frame)

        frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for face in faces_on_frame:
            quality, bbox = face.update_tracker(gray)
            if quality <= 5:
                faces_on_frame.remove(face)

        # Every X frames, we will have to determine which faces
        # are present in the frame
        if (frame_counter % detection_frequency) == 0:

            faces = detector(gray, 1)

            for rect in faces:
                # match faces to faces from prev frame
                x, y, w, h = rect_to_bb(rect)

                # center point:
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matched_face = None
                for face in faces_on_frame:
                    t_x, t_y, t_w, t_h = face.get_tracker_position()

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    # check if the centerpoint of the face is within the
                    # rectangleof a tracker region. Also, the centerpoint
                    # of the tracker region must be within the region
                    # detected as a face. If both of these conditions hold
                    # we have a match
                    if ((t_x <= x_bar <= (t_x + t_w)) and
                            (t_y <= y_bar <= (t_y + t_h)) and
                            (x <= t_x_bar <= (x + w)) and
                            (y <= t_y_bar <= (y + h))):
                        matched_face = face
                        break

                if matched_face is None:
                    matched_face = Face(currentFaceID, gray, (x, y, w, h))
                    faces_on_frame.append(matched_face)
                    currentFaceID += 1

        for face in faces_on_frame:
            draw_roi(frame, face.bbox, face.name)


        draw_hud(frame, '#{}'.format(args['camera_name']), frame_idx=frame_counter)
        cv2.imshow('preview', frame)

        if cv2.waitKey(1) == 27:
            break
