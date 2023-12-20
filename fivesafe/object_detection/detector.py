import numpy as np
import torch
from object_detection import Detection, Detections, Detection_w_mask
from utilities import timing
from abc import abstractmethod, ABC

class Detector(ABC): 
    """ Abstract Detector class """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, frame: np.array):
        pass

class Detectorv5(Detector):
    """ Detector with YOLOv5 """
    def __init__(
            self, 
            model=None, 
            conf_threshold=0.4, 
            classes_of_interest=['person']
    ) -> None:
        if not model:
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                'yolov5s', 
                pretrained=True
            )
        else: 
            self.model = model
        self.conf_threshold = conf_threshold
        self.classes_of_interest = classes_of_interest

    def detect(self, frame: np.array):
        """ Get Detections from model and check if valid """
        frame_detections = self.model(frame)
        detections = Detections()
        for detection in frame_detections.xyxy[0]:
            detection_candidate = Detection(
                xyxy = [
                    detection[0].item(),
                    detection[1].item(),
                    detection[2].item(),
                    detection[3].item()
                ],
                score = detection[4].item(),
                label_id = int(detection[5].item())
            )
            if detection_candidate.is_from_interest(
                self.conf_threshold, 
                self.classes_of_interest
            ):
                detections.append_measurement(detection_candidate)
        return detections

class Detectorv8(Detector):
    """ Detector with YOLOv8 """
    def __init__(
        self,
        model,
        conf_threshold=0.4,
        classes_of_interest=['person']
    ) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.classes_of_interest = classes_of_interest

    def detect(self, frame: np.array):
        """ Get Detections from model and check if valid """
        frame_detections = self.model(frame, verbose=False)
        detections = Detections()
        for detection in frame_detections[0]:#.xyxy[0]:
            detection_candidate = Detection(
                xyxy = detection.boxes.xyxy.squeeze(0).tolist(),
                score = detection.boxes.conf.item(),
                label_id = int(detection.boxes.cls.item())
            )
            if detection_candidate.is_from_interest(
                self.conf_threshold, 
                self.classes_of_interest
            ):
                detections.append_measurement(detection_candidate)
        return detections

class Detectorv8Seg(Detectorv8):
    """ Detector with YOLOv8Seg """
    def detect(self, frame: np.array):
        """ Get Detections from model and check if valid """
        frame_detections = self.model(frame, verbose=False)
        detections = Detections()
        for detection in frame_detections[0]:#.xyxy[0]:
            detection_candidate = Detection_w_mask(
                xyxy = detection.boxes.xyxy.squeeze(0).tolist(),
                score = detection.boxes.conf.item(),
                label_id = int(detection.boxes.cls.item()),
                mask = detection.masks.xy[0].astype(int)
            )
            if detection_candidate.is_from_interest(
                self.conf_threshold, 
                self.classes_of_interest
            ):
                detections.append_measurement(detection_candidate)
        return detections
