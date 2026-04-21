import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # Define the dimensions of the football pitch 
        # 'court_width' corresponds to the width of the field
        # 'court_length' corresponds to the length of the field (here, representing 4 bands of 5.83 meters each)
        court_width = 68
        court_length = 23.32  # 4 bands of a football field, each one is 5.83 meters
        
        # Define the vertices of the trapezoid in the original image.
        # These points should be selected in the order that corresponds to the target vertices order
        # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        self.pixel_verticies = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        # Define the target vertices of the rectangle to which we want to transform the trapezoid.
        # The coordinates here map the trapezoid to a standard rectangle with dimensions based on the pitch.
        # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        self.target_verticies = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Convert the vertices to float32 for compatibility with OpenCV functions.
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        # Calculate the perspective transformation matrix using the source (pixel) vertices
        # and the destination (target) vertices. This matrix can be used to warp the image.
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)



    def transform_point(self, point):
        # Convert the point coordinates to integers and pack them into a tuple
        # This is needed for the cv2.pointPolygonTest, which expects a tuple of integer coordinates
        p = (int(point[0]), int(point[1]))
        
        # Check whether the point 'p' is inside the polygon defined by self.pixel_verticies
        # cv2.pointPolygonTest returns a value >= 0 if the point is inside or on the edge
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        # If the point is not inside the polygon, return None as the transformation should not be applied
        if not is_inside:
            return None
        
        # Reshape the point to a 3D array with shape (number_of_points, 1, 2), which is the required format for cv2.perspectiveTransform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply the perspective transformation using the transformation matrix
        # This will map the point from the original perspective to the target perspective
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        # Reshape the transformed point back to a 2D array and return it.
        return transform_point.reshape(-1, 2)
    



    def add_transformed_position_to_tracks(self, tracks):
        """
        Adds the transformed position to each track in the tracking data.

        Parameters:
        - tracks: Dictionary where each object has a list of tracked positions across frames.

        Modifies:
        - Adds 'position_transformed' key with the transformed coordinates to each track.
        """
        
        # Iterate over all objects being tracked
        for object, object_tracks in tracks.items():
            # Iterate over all frames where the object is tracked
            for frame_num, track in enumerate(object_tracks):
                # Iterate over all track IDs in the current frame
                for track_id, track_info in track.items():
                    # Extract the adjusted position
                    position = track_info['position_adjusted']
                    position = np.array(position)  # Convert to NumPy array

                    # Transform the position using the perspective transformation
                    position_transformed = self.transform_point(position)

                    # If transformation is successful, remove extra dimensions and convert to list
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    # Store the transformed position in the tracks dictionary
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
                    


