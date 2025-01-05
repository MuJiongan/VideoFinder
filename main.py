import os
import openai
from typing import List
from pinecone import Pinecone
from tqdm import tqdm  # for progress bar
from dotenv import load_dotenv

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

def get_file_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_embedding(text: str) -> List[float]:
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def init_pinecone(api_key: str, index_name: str):
    """Initialize Pinecone client and ensure index exists"""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

def process_and_upload_files(files: List[str], index) -> None:
    """Process markdown files and upload embeddings to Pinecone"""
    batch_size = 100  # Adjust based on your needs
    batch = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            content = get_file_content(file_path)
            embedding = get_embedding(content)
            
            # Create vector record with content in metadata
            record = {
                'id': file_path,  # Using file path as ID
                'values': embedding,
                'metadata': {
                    'filepath': file_path,
                    'filename': os.path.basename(file_path),
                    'content': content  # Add the content to metadata
                }
            }
            batch.append(record)
            
            # Upload batch when it reaches batch_size
            if len(batch) >= batch_size:
                index.upsert(vectors=batch)
                print(f"\nUploaded batch of {len(batch)} vectors")
                batch = []
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Upload any remaining vectors
    if batch:
        index.upsert(vectors=batch)
        print(f"\nUploaded final batch of {len(batch)} vectors")

def query_similar_documents(index, query_text: str, k: int = 3) -> List[dict]:
    """
    Query Pinecone for most similar documents
    Args:
        index: Pinecone index
        query_text: Text to find similar documents for
        k: Number of similar documents to return
    """
    # Get embedding for query text
    query_embedding = get_embedding(query_text)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    
    return results.matches

def build_prompt(query: str, similar_docs: List[dict]) -> str:
    docs_content = ""
    for doc in similar_docs:
        docs_content += f"Document: {doc.metadata['filename']}\nContent: {doc.metadata['content']}\n\n"
    prompt = f"You are a helpful assistant. You are given a query and a list of similar documents. You need to answer the query based on the documents. If you are not sure about the answer, you can say 'I don't know' followed by possible answers retrieved from your knowledge base. \n\n Query: {query} \n\n Similar Documents: {docs_content}"

    return prompt

def get_answer(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    load_dotenv()
    # Initialize Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Set your API key in environment variables
    INDEX_NAME = "rag-documents"  # Choose an index name
    openai.api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
    
    if not PINECONE_API_KEY:
        raise ValueError("Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    
    # Initialize Pinecone and get index
    index = init_pinecone(PINECONE_API_KEY, INDEX_NAME)
    
    # # Get markdown files
    # current_directory = "."
    # markdown_files = list_all_files(current_directory)
    
    # if markdown_files:
    #     print(f"Found {len(markdown_files)} markdown files")
    #     process_and_upload_files(markdown_files, index)
    #     print("\nCompleted processing and uploading all files")
        
        
    # else:
    #     print("No markdown files found in the specified directory")
    
    # Example query
    query = "How does naive bayes"
    similar_docs = query_similar_documents(index, query, k=2)
    print(similar_docs)
    # prompt = build_prompt(query, similar_docs)

    # # print(prompt)
    # answer = get_answer(prompt)
    # print(answer)