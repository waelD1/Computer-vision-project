import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():

    def __init__(self):
        # Fix the distance between the player and the ball
        # The goal is to assign the player to the ball when the player is close to the ball and not before
        self.max_player_ball_distance = 70

    
    def assign_ball_to_player(self, players, ball_bbox):


        # Get the center of the ball bounding box
        ball_position = get_center_of_bbox(ball_bbox)

        # Initialize the minimum distance to a high value
        minimum_distance = 10000
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Distance between right and left foot of the player and the ball
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            # Select the minimum distance between the right and left foot of the player and the ball
            distance = min(distance_left, distance_right)

            # Assign the player to the ball if the distance is less than the maximum distance
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
    
        return assigned_player




