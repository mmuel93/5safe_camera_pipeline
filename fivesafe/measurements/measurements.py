import pickle
import numpy as np
from abc import abstractmethod

class Measurements(list):
    def __init__(self) -> None:
        super().__init__()
    
    def save_as_pickle(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_pickle(filename: str) -> np.ndarray:
        with open(filename, 'rb') as file:
            out = pickle.load(file)
        return out

    @abstractmethod
    def append_measurement(self, measurement) -> None:
        pass

    def to_numpy(self) -> np.ndarray:
        """ measurements to numpy: [x,y,x,y,score,id,label_id] """
        out = []
        for measurement in self:
            x1, y1, x2, y2 = measurement.xyxy
            out.append([
                x1, y1, x2, y2, 
                measurement.score,
                int(measurement.id),
                measurement.label_id
            ])
        if len(out) == 0:
            return np.empty((0, 7))
        return np.array(out)