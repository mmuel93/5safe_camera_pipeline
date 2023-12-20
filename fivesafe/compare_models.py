import os
import cv2
from ultralytics import YOLO
from object_detection import Detectorv8

def compare_models(dir_name):
    model_ft = YOLO("./finetuned.pt") 
    model = YOLO('yolov8s.pt')
    detector_ft = Detectorv8(model=model_ft, classes_of_interest=['person', 'car', 'bicycle'], conf_threshold=0.8)
    detector = Detectorv8(model=model, classes_of_interest=['person', 'car', 'bicycle'])

    fnames = os.listdir('test_images')

    for fname in fnames:
        filename = os.path.join(dir_name, fname)
        frame_ft = cv2.imread(filename)
        frame = cv2.imread(filename)
        detections = detector.detect(frame)
        detections_ft = detector_ft.detect(frame)
        for detection in detections:
            frame = detection.draw_rectangle(frame)
            frame = detection.draw_score(frame)

        for detection_ft in detections_ft:
            frame_ft = detection_ft.draw_rectangle(frame_ft, color=(0, 255, 0))
            frame_ft = detection_ft.draw_score(frame_ft, color=(0, 255, 0))

        cv2.imshow('pretrained', frame)
        cv2.imshow('finetuned', frame_ft)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    compare_models('test_images')