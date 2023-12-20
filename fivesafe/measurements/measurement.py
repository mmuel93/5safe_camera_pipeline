import cv2
import numpy as np
from abc import ABC
from measurements.labels import LABELS
from utilities import draw_rectangle

class Measurement(ABC):
    def label(self) -> str:
        return LABELS[self.label_id]

    def xywh(self) -> tuple:
        return [
            int(self.xyxy[0]),
            int(self.xyxy[1]),
            int(self.xyxy[2]) - int(self.xyxy[0]),
            int(self.xyxy[3]) - int(self.xyxy[1])
        ]

    def draw_rectangle(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
        offset=(0, 0),
        thickness=1
    ) -> np.ndarray:
        #return cv2.rectangle(
        return draw_rectangle(
            frame, 
            (int(self.xyxy[0]+offset[0]), int(self.xyxy[1]+offset[1])), 
            (int(self.xyxy[2]+offset[0]), int(self.xyxy[3]+offset[1])),
            color, 
            thickness
        )

    def draw_score(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0),
        offset=(0, 0)
    ) -> np.ndarray:
        return cv2.putText(
            frame, 
            '%.2f' % self.score,
            (int(self.xyxy[0]+offset[0]), int(self.xyxy[1]+offset[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )

    def draw_label(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0),
        offset=(0, 0)
    ) -> np.ndarray:
        return cv2.putText(
            frame, 
            self.label(),
            (int(self.xyxy[0]-10+offset[0]), int(self.xyxy[1]-6+offset[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            1,
            cv2.LINE_AA
        )

    def draw_id(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
        offset=(0, 0)
    ) -> np.ndarray:
        return cv2.putText(
            frame, 
            str(self.id),
            (int(self.xyxy[0] + offset[0]), int(self.xyxy[1]-8+offset[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )
    
    def draw_midpoint(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
    ) -> np.ndarray:
        x, y = self.get_midpoint()
        return cv2.circle(
            frame, 
            (int(x), int(y)),
            3, 
            color, 
            -1
        )

    def set_ground_contact_point(self, pt: tuple) -> None:
        self.ground_contact_point = pt

    def get_midpoint(self) -> tuple:
        x, y, w, h = self.xywh()
        return (x+0.5*w, y+h)