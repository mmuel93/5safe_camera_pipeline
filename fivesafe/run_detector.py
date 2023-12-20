import cv2
from object_detection import Detectorv8Seg, Detectorv8, draw_detections, find_detector_class
from utilities import run
from ultralytics import YOLO

def start(cap, homography_fname, top_view_fname, model_name):
    model = YOLO(model_name)
    DetectorClass = find_detector_class(model_name)
    detector = DetectorClass(model=model, classes_of_interest=['car', 'person', 'bicycle'])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read fname from RTSP Stream")
            break
        detections = detector.detect(frame)
        draw_detections(frame, detections, mask=True)
        cv2.imshow("img", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run('conf/video.yaml', start, bufferless=False)