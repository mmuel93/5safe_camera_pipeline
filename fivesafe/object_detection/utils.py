from object_detection.detector import Detectorv5, Detectorv8, Detectorv8Seg

def draw_detections(frame, detections, mask=False):
    for detection in detections:
        frame = draw_detection(frame, detection, mask)
    return frame

def draw_detection(frame, detection, mask=False):
    if mask:
        frame = detection.draw_mask(
            frame,
            thickness=2
        )
    else:
        frame = detection.draw_rectangle(
            frame, 
            thickness=2
        )
    frame = detection.draw_label(frame)
    frame = detection.draw_id(frame)
    frame = detection.draw_score(frame)
    return frame

def draw_detection_offset(frame, detection):
    frame = detection.draw_rectangle(
        frame, 
        offset=(5, 5), 
        color=(0, 0, 0)
    )
    frame = detection.draw_id(
        frame, 
        offset=(0.5*detection.xywh()[2], 0.5*detection.xywh()[3]), 
        color=(0, 0, 0)
    )
    return frame

def find_detector_class(model_name):
    if 'yolov5' in model_name:
        DetectorClass = Detectorv5
    elif 'yolov8' in model_name:
        if 'seg' in model_name:
            DetectorClass = Detectorv8Seg
        else:
            DetectorClass = Detectorv8
    else:
        print('couldnt find detector class to model')
    return DetectorClass
