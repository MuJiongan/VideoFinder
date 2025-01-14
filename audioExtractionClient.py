from moviepy.editor import VideoFileClip


def video_to_audio(video_path, output_path) -> str:
    # Load the video file
    video = VideoFileClip(video_path)
    # Extract and save the audio as an MP3 file
    video.audio.write_audiofile(output_path)
