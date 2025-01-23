from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read the video file
    video_frames = read_video("data/08fd33_4.mp4")

    # Set up the Tracker
    tracker = Tracker("models/best.pt")

    # Aplly the tracker to the video
    tracks = tracker.get_object_tracks(video_frames, read_from_fragment = True, fragment_path = 'fragments/track_fragments.pkl')

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save the output video file
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
