import threading
import cv2
import dlib


class Face(object):
    def __init__(self, id, gray, bbox, max_images_to_track=500):
        self.max_images_to_track = max_images_to_track
        self.id = id
        self.bbox = bbox
        self.name = '({}) Recognizing...'.format(id)
        # face tracker
        self.tracker = dlib.correlation_tracker()
        # face vectors
        self.vectors = dlib.vectors()

        self.tracking_quality = 0
        self.is_recognized = False

        self.init_tracker(gray)

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

    def get_name(self):
        return self.name if self.is_recognized else self.id