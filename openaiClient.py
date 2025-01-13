import openai
from typing import List

class OpenAIClient:
    def __init__(self, api_key: str):
        """Initialize OpenAI client with API key"""
        self.client = openai.OpenAI(api_key=api_key)

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a given text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text using OpenAI's Whisper model."""
        with open(audio_file, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text" 
            )
        return transcription

    def build_prompt(self, query: str, similar_docs: List[dict]) -> str:
        """Build a prompt from the query and similar documents"""
        docs_content = ""
        for doc in similar_docs:
            docs_content += f"Document: {doc.metadata['filename']}\nContent: {doc.metadata['content']}\n\n"
        
        prompt = f"""You are a helpful assistant. You are given a query and a list of similar documents. 
        You need to answer the query based on the documents. If you are not sure about the answer, 
        you can say 'I don't know' followed by possible answers retrieved from your knowledge base.

        Query: {query}

        Similar Documents: {docs_content}"""

        return prompt

    def get_answer(self, prompt: str) -> str:
        """Get completion from OpenAI's API"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
