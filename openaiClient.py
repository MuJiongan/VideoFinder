import openai
import base64
from PIL import Image
from typing import List
import os

class OpenAIClient:
    def __init__(self, api_key: str):
        """Initialize OpenAI client with API key"""
        self.client = openai.OpenAI(api_key=api_key)
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encodes an image to a Base64 string.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            str: Base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_frames_response(self, folder_path: str) -> str:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")

        # Collect all image files from the folder
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            raise ValueError("No valid image files found in the specified folder.")

        # Prepare GPT-4o messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts text from frames taken from a video. Be as detailed as possible and try to name the people involved in the scene (if there's any) and figure out what went on over time."}
        ]

        # Encode each image and add it to the messages
        for image_file in image_files:
            base64_image = self.encode_image_to_base64(image_file)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })

        # Send request to GPT-4o API
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.05,
                max_tokens=1000  # Adjust based on expected output size
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error while processing frames with GPT-4o: {e}")

    
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text using OpenAI's Whisper model."""
        with open(audio_file, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text" 
            )
        return transcription
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a given text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

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
