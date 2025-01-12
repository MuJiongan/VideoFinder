import os
from typing import List
from dotenv import load_dotenv
from pineconeClient import PineconeClient 
from openaiClient import OpenAIClient
from googledriveClient import GoogleDriveClient

def list_all_files(directory=".") -> List[str]:
    # Walk through all directories and files
    results = []
    for root, dirs, files in os.walk(directory):
        # Print only markdown files in current directory
        for file in files:
            if file.endswith('.md'):  # Only process markdown files
                file_path = os.path.relpath(os.path.join(root, file))
                results.append(file_path)
    return results


if __name__ == "__main__":
    # load_dotenv()
    # # Initialize Pinecone
    # PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Set your API key in environment variables
    # INDEX_NAME = "rag-documents"  # Choose an index name
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
    
    # if not PINECONE_API_KEY and not OPENAI_API_KEY:
    #     raise ValueError("Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    
    # # Initialize Pinecone 
    # pinecone_client = PineconeClient(PINECONE_API_KEY, INDEX_NAME)
    # openai_client = OpenAIClient(OPENAI_API_KEY)
    
    # # Get markdown files
    # current_directory = "."
    # markdown_files = list_all_files(current_directory)
    
    # if markdown_files:
    #     print(f"Found {len(markdown_files)} markdown files")
    #     pinecone_client.process_and_upload_files(markdown_files, openai_client.get_embedding)
    #     print("\nCompleted processing and uploading all files")
        
        
    # else:
    #     print("No markdown files found in the specified directory")
    
    # # Example query
    # query = "How to take pictures"
    # query_embedding = openai_client.get_embedding(query)
    # similar_docs = pinecone_client.query_similar_documents(query_embedding, k=2)
    # print(similar_docs)
    # # prompt = build_prompt(query, similar_docs)

    # # # print(prompt)
    # # answer = get_answer(prompt)
    # # print(answer)
    downloader = GoogleDriveClient()
    
    # Example Google Drive shareable link
    folderUrl = "https://drive.google.com/drive/folders/13yOjGoHafSHfGpmS9uYcCTUlbQ7d_R1A?usp=sharing"
    
    try:
        video_links = downloader.get_folder_file_links(folderUrl)
        for i, vid in enumerate(video_links):
            downloader.download_file(vid['url'], output_path="downloads", filename=str(i)+'.mp4')
        
    except ValueError as e:
        print(f"Error: {e}")