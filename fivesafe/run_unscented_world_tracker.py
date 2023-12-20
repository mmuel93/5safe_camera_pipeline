import cv2
from object_detection import find_detector_class
from image_tracking import Tracker, draw_track
from bev import PositionEstimation, draw_world_position
from utilities import run 
from decision_module import decision_making as dm
from ultralytics import YOLO
from unscented_world_tracking import UnscentedWorldSort
import numpy as np

def start(cap, homography_fname, top_view_fname, model_name):
    # Intialize Detector, Position Estimator, CV2 Windows, and read Top View Image
    model = YOLO(model_name) 
    DetectorClass = find_detector_class(model_name)
    detector = DetectorClass(model=model, classes_of_interest=['person', 'car', 'bicycle'])

    pos_est = PositionEstimation(homography_fname, 81/5) 
    cv2.namedWindow("top_view", cv2.WINDOW_NORMAL)
    cv2.namedWindow("perspective_view", cv2.WINDOW_NORMAL)
    top_view_org = cv2.imread(top_view_fname) 

    # Read Frame, Initialize Image Tracker
    ret, frame = cap.read()
    height, width, _ = frame.shape
    tracker = Tracker(width, height)
    tv_height, tv_width, _ = top_view_org.shape

    #Initialize World Tracker
    worldtracker = WorldSort()

    # Initialize Zones from Config
    CONTOURS_INT_PATH, COLOR_INT_PATH = dm.init_contours('conf/not_intended_paths_contours.yaml')
    CONTOURS_TURNING_RIGHT, COLOR_TURN_RIGHT = dm.init_contours('conf/turning_right_zone_contours.yaml')
    COLOR_STANDARD = (79, 79, 47)

    # Draw Zones into Top View 
    dm.draw_polylines_in_top_view(top_view_org, CONTOURS_INT_PATH, color=COLOR_INT_PATH)
    dm.draw_polylines_in_top_view(top_view_org, CONTOURS_TURNING_RIGHT, color=COLOR_TURN_RIGHT)

    filenamecount = 0
    # Loop over Timesteps
    while True:
        detections = detector.detect(frame)
        tracks = tracker.track(detections)
        top_view = top_view_org.copy()

        nr_of_objects = len(tracks)

        
        # TODO Insert World Tracking Module that takes an Image Track and calculates the World Track with the Perspective Mapping; return List of world tracks to draw in next loop
        dets_world = np.empty((0, 2))
        for track in tracks:
            mask = detections[track.detection_id-1].mask
            world_position, _ = pos_est.map_entity_and_return_relevant_points(track, mask)
            world_position = (world_position[0], world_position[1])
            track.xy_world = world_position
            # DEBUGGING: Draw Raw Detection before Filtering
            top_view = draw_world_position(top_view, (int(track.xy_world[0]), int(track.xy_world[1])), track.id, (255, 0, 0))
            dets_world = np.append(dets_world, np.array([[track.xy_world[0], track.xy_world[1]]]), axis=0) 
        trjs = worldtracker.update(dets_world)
        print(len(worldtracker.trackers))
        for trj in trjs:
            # Transform Midpoint of Track to World Coordinates
            world_position = (int(trj[0]), int(trj[1]))

            # Color Code Situations
            color = COLOR_STANDARD
            if dm.is_pt_in_contours(world_position, CONTOURS_INT_PATH):
                color = COLOR_INT_PATH
            elif dm.is_pt_in_contours(world_position, CONTOURS_TURNING_RIGHT):
                color = COLOR_TURN_RIGHT
            if dm.is_crowded(nr_of_objects, 1):
                frame = dm.draw_crowded(frame, width, height)
                top_view = dm.draw_crowded(
                    top_view, 
                    tv_width, 
                    tv_height, 
                    thickness=40, 
                    font_thickness=5, 
                    font_scale=5, 
                    y_offset=200
                )

            # Draw in Perspective- and Top-View
            top_view = draw_world_position(top_view, world_position, trj[2], color)
        for track in tracks:
            frame = draw_track(frame, track, color=color, draw_detection_id=True)

        # Activate if you want to save the Visualization of the output
        #cv2.imwrite( "C:/Users/mum21730/Desktop/filter_imgs/frames/"+ "%06d.jpg" % filenamecount, frame)
        #cv2.imwrite( "C:/Users/mum21730/Desktop/filter_imgs/worldmap/"+ "%06d.jpg" % filenamecount, top_view)
        filenamecount += 1
        cv2.imshow("perspective_view", frame)
        cv2.imshow("top_view", top_view)
        cv2.waitKey(1)
        ret, frame = cap.read()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run('conf/video_vup.yaml', start, bufferless=False)