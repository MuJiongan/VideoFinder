from typing import List
from pinecone import Pinecone
from tqdm import tqdm
import os

class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        """Initialize Pinecone client and ensure index exists"""
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upload(self, embedding: List[float], vid: dict) -> None:
        """
        Upload embedding and video metadata to pinecone
        
        Args:
            embedding: Vector embedding of the video content
            vid: Dictionary containing video metadata (name, url, id)
        """
        vector = {
            'id': str(vid['id']),
            'values': embedding,
            'metadata': {
                'name': vid['name'],
                'url': vid['url']
            }
        }
        
        try:
            self.index.upsert(vectors=[vector])
        except Exception as e:
            print(f"Error uploading to Pinecone: {str(e)}")

    def query_similar_videos(self, query_embedding: List[float], k: int = 3) -> List[dict]:
        """Query Pinecone for most similar documents"""
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return results.matches

    
