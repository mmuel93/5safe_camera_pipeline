import numpy as np
import cv2
from object_detection import Detectorv8Seg, Detectorv8, draw_detections
from utilities import run
from ultralytics import YOLO
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 10000)
#server_address = ('80.187.141.170', 31622)
print (sys.stderr, 'connecting to %s port %s' % server_address)
sock.connect(server_address)

def start(cap, homography_fname, top_view_fname, model_name):
    model = YOLO('yolov8n-seg.pt')
    detector = Detectorv8Seg(model=model, classes_of_interest=['car', 'person', 'bicycle'])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read fname from RTSP Stream")
            break
        detections = detector.detect(frame)
        msg = '\u007b"detections": [' 
        draw_detections(frame, detections, mask=True)
        for detection in detections:
            hull = cv2.convexHull(
                detection.mask.reshape(-1, 1, 2), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            hull = np.squeeze(hull)
            cv2.polylines(
                frame,
                [hull],
                True,
                (0, 255, 0),
                2
            )
            msg += f'\u007b"id": {detection.id},\
                "class": "{detection.label()}",\
                "score": {detection.score},\
                "box": {detection.xywh()},\
                "mask": {np.array2string(hull, separator=", ")}\u007d,\n'
        msg = msg[:-2]
        msg += ']\u007d'
        #json_msg = json.loads(msg)
        print(msg)
        sock.sendall(bytes(msg, encoding="utf-8"))

        cv2.imshow("img", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    run('conf/video_vup.yaml', start, bufferless=False)