# coding: utf-8
import argparse
import logging
from pprint import pprint
from time import time
import traceback

import cv2
import dlib
import sys
from imutils.face_utils import rect_to_bb
from imutils.video import VideoStream
from common import prepare_frame
from face import Face, ModifiedFaceAligner
from hud import draw_hud, draw_roi
from storage import Storage
from utils import FacesModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -  %(levelname)s: %(message)s')

if __name__ == '__main__':
    logging.info('Starting...')
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-n", "--camera_name", help="Camera name", default='0')
    arg_parser.add_argument("-f", "--detection_frequency", help="How often detect faces on video stream", default='4')

    args = vars(arg_parser.parse_args())

    detection_frequency = int(args['detection_frequency'])

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

    face_aligner = ModifiedFaceAligner(predictor)

    model = FacesModel.Instance().load()
    storage = Storage(model)

    logging.debug('Initing camera stream...')

    vs = VideoStream(usePiCamera=False).start()

    cv2.namedWindow('preview', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    faces_on_frame = []
    frame_counter = 0
    currentFaceID = storage.current_face_id

    logging.info('Processing...')
    while True:
        if frame_counter > sys.maxint - 1: # 9223372036854775807 - 1
            frame_counter = 0

        frame = vs.read()
        frame = prepare_frame(frame)

        frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for face in faces_on_frame:
            quality, bbox = face.update_tracker(gray)
            if quality <= 5:
                # todo check if we really need to save vectors and dump files
                logger.info('remove {} face "{}".'.format('recognized' if face.is_recognized else 'unrecognized', face.get_name()))
                FacesModel.Instance().save_face_visit(face.id, face.start_time, time())
                face.dump_single_face_to_fs(face.first_face, '_first_face', int(face.start_time))
                faces_on_frame.remove(face)
                # else:
                #     face.process_frame(frame, gray, frame_counter)

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
                    if face.is_it_my_face(x, y, w, h, x_bar, y_bar):
                        matched_face = face
                        break

                if matched_face is None:
                    matched_face = Face(currentFaceID, storage, facerec, frame, gray, (x, y, w, h), face_aligner)
                    faces_on_frame.append(matched_face)
                    currentFaceID += 1
                else:
                    matched_face.add_face(frame, gray)

        for face in faces_on_frame:
            draw_roi(frame, face.bbox, face.get_name())

        draw_hud(frame, '#{}'.format(args['camera_name']), frame_idx=frame_counter)
        cv2.imshow('preview', frame)

        if cv2.waitKey(1) == 27:
            break

    logging.info('Exiting...')
    cv2.destroyAllWindows()
    vs.stop()
    exit(0)
