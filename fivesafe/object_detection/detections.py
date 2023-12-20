from measurements import Measurements
from object_detection import Detection

class Detections(Measurements):
    def __init__(self) -> None:
        super().__init__()

    def __setitem__(self, item, value: Detection) -> None:
        """ assert that only Detection class valid """
        assert type(value) == Detection
        super().__setitem__(item, value)

    def append_measurement(self, measurement) -> None:
        measurement.id = len(self)+1
        self.append(measurement)