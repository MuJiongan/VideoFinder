from moviepy.editor import VideoFileClip
import os


def video_to_audio(video_path, folder_name, filename) -> str:
    
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Construct the output path
    output_path = f"{folder_name}/{filename}"
    
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Extract and save the audio as an MP3 file
    video.audio.write_audiofile(output_path, verbose=False, logger=None)
    
    # Clean up resources
    video.close()
    
    return output_path
   
