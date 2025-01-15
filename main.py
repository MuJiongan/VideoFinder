import os
from typing import List
from dotenv import load_dotenv
from pineconeClient import PineconeClient 
from openaiClient import OpenAIClient
from googledriveClient import GoogleDriveClient

from videoProcessingClient import extract_evenly_spread_frames
from audioExtractionClient import video_to_audio


def cleanup(video_file, frames_folder, audio_folder):
    """
    Clean up temporary files and folders after processing
    
    Args:
        video_file (str): Name of the video file to remove
        frames_folder (str): Path to the folder containing extracted frames
        audio_folder (str): Path to the folder containing audio files
    """
    # Remove video file
    if os.path.exists(f"videos/{video_file}"):
        os.remove(f"videos/{video_file}")
    
    # Remove frame images
    for file in os.listdir(frames_folder):
        file_path = os.path.join(frames_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    # Remove audio file
    audio_file = video_file.replace('.mp4', '.mp3')
    if os.path.exists(f"{audio_folder}/{audio_file}"):
        os.remove(f"{audio_folder}/{audio_file}")


if __name__ == "__main__":
    load_dotenv()
    # Initialize Pinecone, OpenAI
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  
    INDEX_NAME = "rag-documents"  # Choose an index name
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
    
    if not PINECONE_API_KEY and not OPENAI_API_KEY:
        raise ValueError("Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    
    pinecone_client = PineconeClient(PINECONE_API_KEY, INDEX_NAME)
    openai_client = OpenAIClient(OPENAI_API_KEY)
    downloader = GoogleDriveClient()
    
    

    folder_url = "https://drive.google.com/drive/folders/13yOjGoHafSHfGpmS9uYcCTUlbQ7d_R1A?usp=sharing"
    video_folder_name = "videos"
    frames_folder_name = "frames"
    audio_folder_name = "audios"
    num_frames = 10
    
    
    try:
        video_links = downloader.get_folder_file_links(folder_url)
        for i, vid in enumerate(video_links):
            
            video_file_name = str(i)+'.mp4'
            audio_file_name = str(i) + '.mp3'
            video_full_path = video_folder_name + '/' + video_file_name
            audio_full_path = audio_folder_name + '/' + audio_file_name
            try:
                downloader.download_file(vid['url'], output_path=video_folder_name, filename=video_file_name)
                video_to_audio(video_full_path, audio_folder_name, audio_file_name)
                extract_evenly_spread_frames(video_full_path, frames_folder_name, num_frames)
            except Exception as e:
                print(f"Error: {e}")
                cleanup(video_file_name, frames_folder_name, audio_folder_name)
                continue
            video_text = openai_client.get_frames_response(frames_folder_name)
            audio_text = openai_client.transcribe(audio_full_path)
            combined_text = video_text + "\n" + audio_text
            cleanup(video_file_name, frames_folder_name, audio_folder_name)
            embedding = openai_client.get_embedding(combined_text)
            pinecone_client.upload(embedding, vid)

    except ValueError as e:
        print(f"Error: {e}")
    query = "Robertson From Turkey"
    query_embedding = openai_client.get_embedding(query)
    print(pinecone_client.query_similar_videos(query_embedding, 3))