from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read the video file
    video_frames = read_video("data/08fd33_4.mp4")

    # Set up the Tracker
    tracker = Tracker("models/best.pt")

    # Aplly the tracker to the video
    tracks = tracker.get_object_tracks(video_frames, read_from_fragment = True, fragment_path = 'fragments/track_fragments.pkl')

    # Interpolate ball positions to fill in missing values 
    # It allows the pointer to be present in all frames
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign team to the players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Loop over each player in each frame and assign them a team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                track['bbox'],
                                                player_id)
            # Save the team and team color in the tracks
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign the ball to a player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
     
    # Loop over each player in each frame and determine if which player has the ball
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # Save the player that has the ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Save the team that has the ball
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team']) 
        else:
            # Add the last team that had the ball 
            team_ball_control.append(team_ball_control[-1]) 


    team_ball_control = np.array(team_ball_control)
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save the output video file
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
