import cv2


def read_video(video_path):

    """
    Reads a video from the specified file path and returns the video capture object.
    Args:
        video_path (str): The path to the video file.
    Returns:
        cv2.VideoCapture: The video capture object for the video file.
    Raises:
        Exception: If the video file cannot be opened.
    Note:
        This function reads all frames from the video and stores them in a list.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    # Create a list to store the frames
    # The script stops when there is no other frame to read
    frames = []
    while True:
        # Grabs, decodes and returns the next video frame
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames



def save_video(output_video_frames, output_video_path):
    # Define the codec using the FourCC code for XVID
    # This specifies that the video will be compressed using the XVID codec, 
    # which is a popular codec for creating .avi files
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create a VideoWriter object to save the video to a file
    # output_video_path is the file path name, 24 is the frame rate, and (output_video_frames[0].shape[1], output_video_frames[0].shape[1]) is the resolution
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()



