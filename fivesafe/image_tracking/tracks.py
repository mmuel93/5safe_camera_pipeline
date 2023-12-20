import numpy as np
from measurements import Measurements
from image_tracking import Track


class Tracks(Measurements):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        out = ''
        for track in self:
            out += f'{track.id}: {track}\n'
        return out

    def numpy_to_tracks(self, track_list: np.ndarray, img_w, img_h):
        for track_candidate in track_list:
            x1, y1, x2, y2, track_id, detection_id, score, label_id, rvec_x, rvec_y = track_candidate
            track_candidate = Track(
                xyxy=(x1, y1, x2, y2), 
                label_id=int(label_id),
                score=score, 
                id=int(track_id), 
                detection_id=int(detection_id),
                initial_rvec_est = (rvec_x, rvec_y)
            )
            if track_candidate.is_collision_between_bbox_and_img_border(img_w, img_h):
                continue
            self.append_measurement(track_candidate)
        return self

    def append_measurement(self, measurement) -> None:
        self.append(measurement)