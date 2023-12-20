import cv2
import numpy as np
from bev.utils import draw_vehicle_baseplate

if __name__ == "__main__":
    top_view_path = 'C:/Users/mum21730/Desktop/5_Safe/Bilder/Verkehrsuebungsplatz/Measurements_230210/Refimg/Refimg_1.JPG'
    top_view = cv2.imread(top_view_path)
    scale = 81/5
    rvec = np.array([[1], [0]])
    midpoint = np.array([[500], [500]])
    top_view = draw_vehicle_baseplate(top_view, midpoint, rvec, 4.5, 2, scale)
    cv2.namedWindow("top_view", cv2.WINDOW_NORMAL)
    cv2.imshow("top_view", top_view)
    cv2.waitKey(0)