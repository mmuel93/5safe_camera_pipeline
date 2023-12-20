import json
import cv2
import numpy as np
from measurements import Measurement

class Detection(Measurement):
    def __init__(self, xyxy: tuple, label_id: int, score: int):
        self.id = None
        self.xyxy = xyxy
        self.label_id = label_id
        self.score = score

    def __repr__(self) -> str:
        return f'Detection id: {self.id}, class: {self.label()}, \
            score: {self.score}, box: {self.xywh()}'

    def is_from_interest(
            self, 
            conf_threshold: float, 
            classes_of_interest: list
        ) -> bool:
        """ check if we care for this detection """
        return (
            (self.label() in classes_of_interest) \
            & (self.score > conf_threshold)
        )

    def get_visualized_detection(self, frame: np.ndarray) -> np.ndarray:
        """ cut frame and only show detection """
        x, y, w, h = self.xywh()
        crop_img = frame[y:y+h, x:x+w]
        return crop_img
    

class Detection_w_mask(Detection):
    def __init__(self, xyxy: tuple, label_id: int, score: int, mask):
        super().__init__(xyxy, label_id, score)
        self.mask = mask

    def draw_mask(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
        offset=(0, 0),
        thickness=1
    ) -> np.ndarray:
        mask = np.array(self.mask, np.int32).reshape((-1, 1, 2))
        return cv2.polylines(
            frame, 
            [mask],
            True,
            color, 
            thickness
        )

    def __repr__(self) -> str:
        return f'Detection id: {self.id}, class: {self.label()}, \
            score: {self.score}, box: {self.xywh()}, mask: {self.mask}'
    