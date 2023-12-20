import cv2
from object_detection import find_detector_class
from image_tracking import Tracker, draw_tracks, draw_debug_tracks
from utilities import run
from ultralytics import YOLO


def start(cap, homography_fname, top_view_fname, model_name):
    model = YOLO(model_name)
    DetectorClass = find_detector_class(model_name)
    detector = DetectorClass(model=model, classes_of_interest=['car', 'person', 'bicycle'])
    _, frame = cap.read()
    height, width, c = frame.shape
    print(width, height)
    tracker = Tracker(width, height)

    while True:
        detections = detector.detect(frame)
        tracks = tracker.track(detections)
        frame = draw_tracks(frame, tracks)
        #frame = draw_debug_tracks(frame, tracks, detections)

        cv2.imshow("img", frame)
        cv2.waitKey(1)
        _, frame = cap.read()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run('conf/video.yaml', start, bufferless=False)