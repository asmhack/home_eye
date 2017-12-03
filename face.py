import logging
import threading
from pprint import pprint
from time import time

import cv2
import dlib
from imutils.face_utils import shape_to_np, FACIAL_LANDMARKS_IDXS, FaceAligner
import numpy as np
import os

from utils import FacesModel

logger = logging.getLogger(__name__)


class Face(object):
    def __init__(self, id, storage, facerec, original_frame, gray, bbox, aligner, max_images_to_track=40):
        self.storage = storage
        self.facerec = facerec
        self.id = id
        self.bbox = bbox
        self.aligner = aligner
        self.max_images_to_track = max_images_to_track
        self.name = '({}) Recognizing...'.format(id)
        # face tracker
        self.tracker = dlib.correlation_tracker()
        # face vectors
        self.vectors = []  # dlib.vectors()

        self.start_time = time()

        self.tracking_quality = 0
        self.is_recognized = False
        self.face_aligned = None
        self.shape = None
        self.faces = []
        self.first_face = None
        self.detection_count_tries = 0
        self.max_detection_count_tries = 5
        self.total_images_stored = 0

        self.init_tracker(gray)
        self.add_face(original_frame, gray)

    def get_name(self):
        return self.name if self.is_recognized else str(self.id)

    def align_face(self, original_frame, gray):
        (x, y, w, h) = self.bbox
        self.face_aligned, self.shape = self.aligner.align(original_frame, gray, dlib.rectangle(x, y, x + h, y + w))
        # cv2.imshow("Aligned {}".format(self.id), self.face_aligned)
        # return cv2.cvtColor(self.face_aligned, cv2.COLOR_BGR2GRAY), self.shape

    def add_face(self, original_frame, gray):

        self.align_face(original_frame, gray)

        if self.total_images_stored <= self.max_images_to_track:

            if len(self.faces) == 0:
                self.first_face = self.face_aligned

            self.faces.append(self.face_aligned)
            # calc hash
            # face_descriptor = self.facerec.compute_face_descriptor(original_frame, self.shape, 100) # wil be faster
            face_descriptor = self.facerec.compute_face_descriptor(original_frame, self.shape)
            # add hash to current face describer
            self.vectors.append(face_descriptor)
            self.total_images_stored += 1
            if not self.is_recognized:
                if self.detection_count_tries < self.max_detection_count_tries:
                    # try to recognize face
                    # threading.Thread(target=self.recognize_face).start()
                    self.recognize_face()

                elif self.detection_count_tries == self.max_detection_count_tries:
                    # this is new person and we need to save it
                    self.storage.extend(self.vectors, self.id)
                    logger.info('SAVE 6 faces. total = {}'.format(len(self.storage.vectors)))
                    self.detection_count_tries += 1
                    self.is_recognized = True

                    FacesModel.Instance().create_new_face(self.id)

                    self.set_name(self.id)
                    self.dump()

            else:
                logger.info('just save extra faces. total = {}'.format(len(self.storage.vectors)))
                self.storage.append([self.id, face_descriptor])
                self.dump_single_face_to_fs(self.face_aligned)
                FacesModel.Instance().save_single_vector(self.id, face_descriptor)

    def dump(self):
        # threading.Thread(target=self.dump_faces_to_fs).start()
        self.dump_faces_to_fs()

    def recognize_face(self):
        logger.debug('recognize_face {}'.format(self.id))
        self.detection_count_tries += 1
        res = self.storage.match_vector(self.vectors[-1], self.id)

        if res != False:
            self.is_recognized = True
            self.id = res[0]

            self.set_name(FacesModel.Instance().get_name_by_label(self.id))

            if res[1] >= .5: # if not so similar to previous ones, then save it, even if limit was reached
                for vec in self.vectors:
                    self.storage.append([self.id, vec])
                    FacesModel.Instance().save_single_vector(self.id, vec)

                for face in self.faces:
                    self.dump_single_face_to_fs(face)

            self.total_images_stored = self.storage.get_total_images_for_label(self.id)

    def init_tracker(self, gray_full):
        x, y, w, h = self.bbox
        self.tracker.start_track(gray_full, dlib.rectangle(x, y, x + h, y + w))
        self.tracking_quality = self.tracker.update(gray_full)

    def update_tracker(self, img):
        self.tracking_quality = self.tracker.update(img)
        x, y, w, h = self.get_tracker_position()
        self.bbox = (x, y, w, h)
        return self.tracking_quality, self.bbox

    def get_tracker_position(self):
        tracked_position = self.tracker.get_position()
        x = int(tracked_position.left())
        y = int(tracked_position.top())
        w = int(tracked_position.width())
        h = int(tracked_position.height())
        return (x, y, w, h)

    def dump_vectors_to_file(self):
        FacesModel.Instance().save_multiple_vectors(self.id, self.vectors)

    def dump_faces_to_fs(self):
        if len(self.faces) > 0 and not os.path.exists('data/faces/{}'.format(self.get_name())):
            os.mkdir('data/faces/{}'.format(self.get_name()))

        logger.debug('Dumping for "{}" {} faces...'.format(self.get_name(), len(self.faces)))
        t = time()
        for i, face in enumerate(self.faces):
            self.dump_single_face_to_fs(face, i, t)

        self.dump_vectors_to_file()

    def dump_single_face_to_fs(self, face, i=None, t=time()):
        if i is None:
            i = len(self.faces)

        cv2.imwrite('data/faces/{}/{}{}.png'.format(self.get_name(), t, i), face)

    def set_name(self, name):
        self.name = name

    def is_it_my_face(self, x, y, w, h, x_bar, y_bar):
        t_x, t_y, t_w, t_h = self.get_tracker_position()

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
            return True

        return False


class ModifiedFaceAligner(FaceAligner):
    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape1 = self.predictor(gray, rect)
        shape = shape_to_np(shape1)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output, shape1
