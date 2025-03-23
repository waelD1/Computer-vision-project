from ultralytics import YOLO
import supervision as sv
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width



class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack() 


    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox'] 
                    if object == 'ball':
                        positions = get_center_of_bbox(bbox)
                    else:
                        position = get_center_of_bbox(bbox)
                    tracks[object][frame_num][track_id]['position'] = position



                



    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball position values in a sequence of bounding boxes.

        Args:
            ball_positions (list of dict): A list where each element is a dictionary of the form:
                {1: {'bbox': [x1, y1, x2, y2]}}.
                Some bounding boxes may have missing values.

        Returns:
            list of dict: A list with the same format, but with missing values interpolated.
        """

        # Extract bounding box coordinates from the input list
        # If bbox is missing, return an empty list as a default
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Convert the extracted bounding box list into a Pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values linearly based on previous and next values
        df_ball_positions = df_ball_positions.interpolate()

        # Fill any remaining missing values with the last known valid value
        df_ball_positions = df_ball_positions.bfill()

        # Convert the DataFrame back into the original dictionary format
        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20

        detections = []
        for i in range(0, len(frames), batch_size):
            # Predict the batch
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            # Append the batch detections to the list
            detections += (detections_batch)

        return detections

    
    def get_object_tracks(self, frames, read_from_fragment = False, fragment_path = None):     
        
        # If the path exist, just return the saved tracks
        if read_from_fragment and fragment_path is not None and os.path.exists(fragment_path):
            with open(fragment_path, "rb") as f:
                tracks = pickle.load(f)
            
            print("File already exist. Tracks loaded from file")
            return tracks


        detections = self.detect_frames(frames)

        tracks={
            "players": [],
            "referees" : [],
            "ball" : []
        }


        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverse = {v:k for k, v in class_names.items()}

            # Convert to Supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # We want to convert the goalkeeper to be detected as a regular player
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = class_names_inverse['player']

            # Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)


            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                # We set up a boundary box for the players
                bbox = frame_detection[0].tolist()  
                # Get the class id
                class_id = frame_detection[3]
                # Get the track ID
                track_id = frame_detection[4]

                if class_id == class_names_inverse['player']:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}


                if class_id == class_names_inverse['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox" : bbox}

            

            # For the ball, we loop over the detection without the tracks (detection_supervision instead of detection_with_tracks)
            # The reason os that the ball doesn't require identity tracking, while players and referees use detection_with_tracks for maintaining persistent identities across frames
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inverse['ball']:
                    # We put frame_num as the key because the is only one ball in the frame
                    tracks["ball"][frame_num][1] = {"bbox" : bbox}

        # Save the tracks to a file if file path is not empty
        if fragment_path is not None:
            with open(fragment_path, "wb") as f:
                pickle.dump(tracks, f)


            
        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) 
        y2_rect = (y2+ rectangle_height//2) 
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10 
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame    


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # Draw the triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Draw the border of the triangle

        return frame
    

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw the team ball control statistics on the frame.
        """

        # Draw a semi-transparent rectangle to display the team in control of the ball
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) 

        # Stores ball control information for all frames in a match
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Filter the array to count how many times each team had control of the ball up to the current frame
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        # Compute statistics for each team based on the number of frames they had control of the ball
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Display the team ball control statistics on the frame
        cv2.putText(frame, f"Team 1 Ball Control : {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control : {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 0, 0), 3)

        return frame 



    def draw_annotations(self, video_frames, tracks, team_ball_control):  
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]  
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                # The default color is blue if the team color is not available
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Draw a red triangle to the player that has the ball
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            # Referee tracker is in yellow color
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            output_video_frames.append(frame)

        return output_video_frames


     