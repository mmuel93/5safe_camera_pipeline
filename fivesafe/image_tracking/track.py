import numpy as np
import cv2 
from measurements import Measurement

class Track(Measurement):
    def __init__(
        self, 
        xyxy: tuple, 
        label_id: int, 
        score: int, 
        detection_id: int, 
        id: int, 
        initial_rvec_est: list,
        threshold: int = 10,
        xy_world = tuple
    ) -> None:
        self.id = id
        self.xyxy = xyxy
        self.label_id = label_id
        self.score = score
        self.detection_id = detection_id
        self.threshold = threshold
        self.xy_world = None
        self.initial_rvec_est = initial_rvec_est


    def __repr__(self) -> str:
        return f'Track id: {self.id}, class: {self.label()}, \
            score: {self.score:.2f}, box: {self.xywh()}, \
            detection_id: {self.detection_id}, \
            initial_rvec_est: {self.initial_rvec_est}'
    
    def is_collision_between_bbox_and_img_border(self, img_width, img_height):
        x1, y1, x2, y2 = self.xyxy
        if self.check_collision(x1, img_width, self.threshold) \
            or self.check_collision(y1, img_height, self.threshold) \
            or self.check_collision(x2, img_width, self.threshold) \
            or self.check_collision(y2, img_height, self.threshold):
            return True
        return False 

    @staticmethod
    def check_collision(bbox_coord, img_parameter, threshold):
        if(
            bbox_coord > (0 + threshold) \
            and bbox_coord < (img_parameter - threshold)
        ):
            return False
        return True
    
    def get_dict(self):
        output = super.get_dict()
        output["detection_id"] = self.detection_id
        return output
    
    def draw_detection_id(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
        offset=(0, 0)
    ) -> np.ndarray:
        return cv2.putText(
            frame, 
            f'id: {self.id}, d_id: {self.detection_id}',
            (int(self.xyxy[0] + offset[0]), int(self.xyxy[1]+offset[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )
