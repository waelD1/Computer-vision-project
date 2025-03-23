import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        
        # Camera movement threshold (minimum camera movement required to consider that the camera moved)
        self.minimum_distance = 5

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            # Size of the search window at each pyramid level (15x15 pixels)
            winSize=(15, 15),
            # Number of pyramid levels for multi-scale tracking
            maxLevel=2,
            # Stopping criteria for the iterative search algorithm
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  # Stop when either condition is met
                10,  # Maximum number of iterations
                0.03  # Minimum required accuracy (epsilon)
            ))
        

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)

        # Initialize a mask to restrict feature detection to specific regions.
        # This prevents detecting features in less relevant parts of the frame.
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1



        # Feature detection parameters are chosen to balance accuracy and efficiency:
        # - maxCorners: Limits the number of features detected to avoid excessive computation.
        # - qualityLevel: Determines the minimum quality of features (0.3 means only strong features are kept).
        # - minDistance: Ensures that detected features are at least 7 pixels apart to prevent clustering.
        # - blockSize: Defines the size of the window for computing feature quality.
        # - mask: Restricts feature detection to the defined regions.
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
            mask = mask_features
        )


    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])  
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted


    def get_camera_movement(self, frames, read_from_fragment=False, fragment_path = None):
        # Read from fragment if exists
        if read_from_fragment and fragment_path is not None and os.path.exists(fragment_path):
            with open(fragment_path, "rb") as f:
                camera_movement = pickle.load(f)

        # Initialize the camera movement to 0 for all frames
        camera_movement = [[0, 0]]*len(frames)

        # 
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            
            old_gray = frame_gray.copy()

        if fragment_path is not None:
            with open(fragment_path, "wb") as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement 
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X : {x_movement : .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Camera Movement Y : {y_movement : .2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            output_frames.append(frame)

        return output_frames



        




            
