from moviepy.editor import VideoFileClip
import os
from PIL import Image

def extract_frames_with_interval(video_path, output_dir, interval=1):
    """
    Extract frames from a video at a configurable interval.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        interval (int): Time interval (in seconds) between frames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the video file
    clip = VideoFileClip(video_path)
    
    # Extract frames at the specified interval
    for t in range(0, int(clip.duration), interval):
        frame = clip.get_frame(t)  # Get the frame at time `t` seconds
        frame_path = os.path.join(output_dir, f"frame_{t:04d}.jpg")
        
        # Save the frame as an image
        Image.fromarray(frame).save(frame_path)

    print(f"Frames extracted every {interval} second(s) and saved in {output_dir}")


def extract_evenly_spread_frames(video_path, output_dir, num_frames):
    """
    Extract a specified number of evenly spread frames from a video.

    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        num_frames (int): Number of frames to extract evenly across the video.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video file
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Total duration of the video in seconds

    # Calculate the time interval between frames
    if num_frames > 1:
        interval = duration / (num_frames - 1)  # Spread frames evenly
    else:
        interval = duration  # If only one frame is needed, take the last frame

    # Extract frames at evenly spaced intervals
    for i in range(num_frames):
        t = min(i * interval, duration)  # Ensure `t` does not exceed video duration
        frame = clip.get_frame(t)  # Get the frame at time `t`
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")

        # Save the frame as an image
        Image.fromarray(frame).save(frame_path)

    print(f"{num_frames} frames extracted and saved in {output_dir}")