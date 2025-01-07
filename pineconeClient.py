from typing import List
from pinecone import Pinecone
from tqdm import tqdm
import os

class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        """Initialize Pinecone client and ensure index exists"""
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def process_and_upload_files(self, files: List[str], get_embedding_func) -> None:
        """Process markdown files and upload embeddings to Pinecone"""
        batch_size = 100
        batch = []
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                content = self._get_file_content(file_path)
                embedding = get_embedding_func(content)
                
                record = {
                    'id': file_path,
                    'values': embedding,
                    'metadata': {
                        'filepath': file_path,
                        'filename': os.path.basename(file_path)
                    }
                }
                batch.append(record)
                
                if len(batch) >= batch_size:
                    self.index.upsert(vectors=batch)
                    print(f"\nUploaded batch of {len(batch)} vectors")
                    batch = []
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if batch:
            self.index.upsert(vectors=batch)
            print(f"\nUploaded final batch of {len(batch)} vectors")

    def query_similar_documents(self, query_embedding: List[float], k: int = 3) -> List[dict]:
        """Query Pinecone for most similar documents"""
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return results.matches

    @staticmethod
    def _get_file_content(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
