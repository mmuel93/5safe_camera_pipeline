from image_tracking.sort import Sort
from image_tracking import Tracks


class Tracker:
    def __init__(self, img_width, img_height):
        self.mot_tracker = Sort()
        self.img_width = img_width
        self.img_height = img_height

    def track(self, detections): # TODO NUMPY AND detection.to_numpy() in main!
        #xyxy, score, id, label -> xyxy, old_id, new_id, score, label
        tracks = Tracks()
        trackers = self.mot_tracker.update(detections.to_numpy()) 
        print(trackers)
        tracks = tracks.numpy_to_tracks(trackers, self.img_width, self.img_height)
        return tracks