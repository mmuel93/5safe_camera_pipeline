import cv2
from object_detection import find_detector_class
from image_tracking import Tracker, draw_track
from bev import PositionEstimation, draw_world_position, draw_vehicle_baseplate
from utilities import run 
from decision_module import decision_making as dm
from ultralytics import YOLO
from world_tracking import WorldSort
from cheap_fusion import CheapFusion
import numpy as np

def start(cap, homography_fname, top_view_fname, model_name):
    # Intialize Detector, Position Estimator, CV2 Windows, and read Top View Image
    model = YOLO(model_name) 
    DetectorClass = find_detector_class(model_name)
    detector = DetectorClass(model=model, classes_of_interest=['person', 'car', 'bicycle'])
    scalefactor = 17.814
    cv2.namedWindow("top_view", cv2.WINDOW_NORMAL)
    cv2.namedWindow("perspective_view", cv2.WINDOW_NORMAL)
    top_view_org = cv2.imread(top_view_fname) 
    
    # Initialize Camera Matrices and Distortion Parameters, so if not given in conf the code will run anyway.
    mat_pv = None
    dist_pv = None


    # Read Frame, Initialize Image Tracker
    ret, frame = cap.read()
    height, width, _ = frame.shape
    tracker = Tracker(width, height)
    tv_height, tv_width, _ = top_view_org.shape

    # PositionEstimation got new Input Parameters due to local undistortion in this module. Maybe undistort Points in separate Module...
    pos_est = PositionEstimation(homography_fname, scalefactor, mat_pv, dist_pv, width, height)     # TUM 632.26/13.5


    #Initialize World Tracker; Added initialization mode for first guess of rvec. "image" will utilize Estimate from PositionEstimation Module, "infrastructure" will utilize fixed values in Function (can be found in WorldTracker)
    worldtracker_vehicles = WorldSort(dist_threshold=3, initial_rvec_estimate_mode="image")
    worldtracker_vrus = WorldSort(dist_threshold=.5, initial_rvec_estimate_mode="image")

    #Initialize cheap Fusion:   (distance for Fusion in meters, scalefactor in px/m)
    # this function 
    cheap_fuser_vrus = CheapFusion(.5, scalefactor)
    cheap_fuser_vehicles = CheapFusion(1.5, scalefactor)

    # Initialize Zones from Config
    CONTOURS_INT_PATH, COLOR_INT_PATH = dm.init_contours('conf/not_intended_paths_contours.yaml')
    CONTOURS_TURNING_RIGHT, COLOR_TURN_RIGHT = dm.init_contours('conf/turning_right_zone_contours.yaml')
    COLOR_STANDARD = (79, 79, 47)

    # Draw Zones into Top View 
    #dm.draw_polylines_in_top_view(top_view_org, CONTOURS_INT_PATH, color=COLOR_INT_PATH)
    #dm.draw_polylines_in_top_view(top_view_org, CONTOURS_TURNING_RIGHT, color=COLOR_TURN_RIGHT)

    filenamecount = 0
    # Loop over Timesteps
    while True:
        detections = detector.detect(frame)
        tracks = tracker.track(detections)
        top_view = top_view_org.copy()

        nr_of_objects = len(tracks)

        # Format of np.empty changed because of new input parameters
        dets_world_vehicles = np.empty((0, 6))
        dets_world_vrus = np.empty((0, 6))
        for track in tracks:
            mask = detections[track.detection_id-1].mask
            # psi_world is used now! Depends on intial rvec guess method of world tracker ("image")
            world_position, psi_world, gcp_img = pos_est.map_entity_and_return_relevant_points(track, mask)
            world_position = (world_position[0], world_position[1])
            track.xy_world = world_position
            # DEBUGGING: Draw Raw Detection before Filtering
            top_view = draw_world_position(top_view, (int(track.xy_world[0]), int(track.xy_world[1])), track.id, (255, 0, 0))
            if track.label_id ==2 or track.label_id==8: #TODO: Add ID for truck, bus, van and motorcycle if availabe
                # Input Format Changed!!!!!!!  track.id changed Position, psi_world is necessary for potential initial rvec estimate
                dets_world_vehicles = np.append(dets_world_vehicles, np.array([[track.xy_world[0], track.xy_world[1], track.id, psi_world[0][0], psi_world[1][0], track.label_id]]), axis=0) 
            else:
                # Input Format Changed!!!!!!!  track.id changed Position, psi_world is necessary for potential initial rvec estimate; VRUs don't have an rvec, so estimate is 0
                dets_world_vrus = np.append(dets_world_vrus, np.array([[track.xy_world[0], track.xy_world[1], track.id, 0, 0, track.label_id]]), axis=0) 
            if track.label_id:  # !=2 for cleaning cars
                try:
                    cv2.circle(frame, (int(gcp_img[0][0]), int(gcp_img[0][1])),5, (255, 255, 0), -1)
                    cv2.circle(frame, (int(gcp_img[1][0]), int(gcp_img[1][1])),5, (255, 255, 0), -1)
                except:
                    print("No GPCs in Image available")
                    continue
        trjs_vehicles = worldtracker_vehicles.update(dets_world_vehicles)
        trjs_vrus = worldtracker_vrus.update(dets_world_vrus)

        # Call Fuser and fuse objects too close together
        trjs_vrus = cheap_fuser_vrus.fuse(trjs_vrus)
        trjs_vehicles = cheap_fuser_vehicles.fuse(trjs_vehicles)

        for trj in trjs_vrus:
            # Transform Midpoint of Track to World Coordinates
            world_position = (int(trj[0]), int(trj[1]))
            #rvec_normed = (trj[3], trj[4])
            """
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
            """
            color = COLOR_STANDARD  #Delete if color coding is active
            # Draw in Perspective- and Top-View
            top_view = draw_world_position(top_view, world_position, trj[2], (255, 255, 0))

        for trj in trjs_vehicles:
            # Transform Midpoint of Track to World Coordinates
            world_position = (int(trj[0]), int(trj[1]))
            rvec_normed = (trj[3], trj[4])
            """
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
            """
            color = COLOR_STANDARD  #Delete if color coding is active
            # Draw in Perspective- and Top-View
            top_view = draw_world_position(top_view, world_position, trj[2], color)

            # Draw vehicle Baseplate on Topview and Frame if possible
            try:
                # Draw Vehicle Baseplate on Topview
                top_view, tl, tr, bl, br = draw_vehicle_baseplate(top_view, np.array([[trj[0]], [trj[1]]]), np.array([[rvec_normed[0]], [rvec_normed[1]]]), 4.5, 1.8, scalefactor, thickness=3)
                # Calculate Corner Points in Image Coordinates
                tl_img  = pos_est.transform_point_from_world_to_image(tl)
                tr_img  = pos_est.transform_point_from_world_to_image(tr)
                bl_img  = pos_est.transform_point_from_world_to_image(bl)
                br_img  = pos_est.transform_point_from_world_to_image(br)
                # Draw Baseplate to Frame
                frame = cv2.line(frame, (int(tl_img[0]), int(tl_img[1])), (int(tr_img[0]), int(tr_img[1])), color, 5)
                frame = cv2.line(frame, (int(bl_img[0]), int(bl_img[1])), (int(br_img[0]), int(br_img[1])), color, 5)
                frame = cv2.line(frame, (int(tl_img[0]), int(tl_img[1])), (int(bl_img[0]), int(bl_img[1])), color, 5)
                frame = cv2.line(frame, (int(tr_img[0]), int(tr_img[1])), (int(br_img[0]), int(br_img[1])), color, 5)
            except:
                continue

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
    run('conf/video_LUMPI.yaml', start, bufferless=False)