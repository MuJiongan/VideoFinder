import requests
from urllib.parse import urlparse, parse_qs
import os
from tqdm import tqdm

class GoogleDriveClient:
    """A class to download files from Google Drive using shareable links."""
    
    def __init__(self):
        """Initialize the downloader with necessary URLs and headers."""
        self.CHUNK_SIZE = 32768
        self.DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_folder_file_links(self, folder_url: str) -> list:
        """
        Get all file links from a shared Google Drive folder.
        
        Args:
            folder_url (str): The shared Google Drive folder URL
            
        Returns:
            list: List of dictionaries containing file information
                Each dict contains {'name': str, 'url': str, 'id': str}
                
        Raises:
            ValueError: If the URL is invalid or folder cannot be accessed
        """
        try:
            # Extract folder ID from URL
            if 'folders' not in folder_url:
                raise ValueError("Not a valid Google Drive folder URL")
            
            folder_id = folder_url.split('folders/')[1].split('?')[0]
            
            # Construct the folder view URL
            folder_view_url = f"https://drive.google.com/drive/folders/{folder_id}"
            
            # Get the folder page content
            response = requests.get(folder_view_url, headers=self.headers)
            
            if response.status_code != 200:
                raise ValueError("Could not access the folder. Make sure it's shared publicly.")
            
            # Parse the response to find file links
            import re
            # Find all file IDs
            file_ids = list(set(re.findall(r'\/d\/([^\/\\"]+)', response.text)))
            
            files = []
            for file_id in file_ids:
                # Construct file URL
                file_url = f"https://drive.google.com/file/d/{file_id}/view"
                
                # Try to get file name from the page content
                name_match = re.search(rf'{file_id}[^"]*?"([^"]+)"', response.text)
                name = name_match.group(1) if name_match else f"file_{file_id}"
                
                files.append({
                    'name': name,
                    'url': file_url,
                    'id': file_id
                })
            
            return files
            
        except Exception as e:
            raise ValueError(f"Error getting folder contents: {str(e)}")

    def extract_file_id(self, url: str) -> str:
        """
        Extract file ID from Google Drive URL.
        
        Args:
            url (str): Google Drive shareable link
            
        Returns:
            str: File ID
            
        Raises:
            ValueError: If the URL is invalid or file ID cannot be extracted
        """
        try:
            if 'drive.google.com' not in url:
                raise ValueError("Not a valid Google Drive URL")

            if '/file/d/' in url:
                # Handle links like: https://drive.google.com/file/d/{file_id}/view
                file_id = url.split('/file/d/')[1].split('/')[0]
            else:
                # Handle links like: https://drive.google.com/open?id={file_id}
                parsed = urlparse(url)
                file_id = parse_qs(parsed.query).get('id', [None])[0]

            if not file_id:
                raise ValueError("Could not extract file ID from URL")

            return file_id

        except Exception as e:
            raise ValueError(f"Error extracting file ID: {str(e)}")

    def download_file(self, url: str, output_path: str, filename: str = None) -> str:
        """
        Download a file from Google Drive using a shareable link.
        
        Args:
            url (str): Google Drive shareable link
            output_path (str): Directory to save the file
            filename (str, optional): Custom filename for the downloaded file
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            ValueError: If the URL is invalid or download fails
        """
        try:
            file_id = self.extract_file_id(url)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # First request to get the confirmation token if needed
            session = requests.Session()
            response = session.get(
                self.DOWNLOAD_URL,
                params={'id': file_id},
                headers=self.headers,
                stream=True
            )

            # Check if we need to handle the confirmation page
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    response = session.get(
                        self.DOWNLOAD_URL,
                        params={'id': file_id, 'confirm': value},
                        headers=self.headers,
                        stream=True
                    )
                    break

            # Get filename from response headers if not provided
            if not filename:
                if "Content-Disposition" in response.headers:
                    filename = response.headers["Content-Disposition"].split("filename=")[1].strip('"')
                else:
                    filename = f"downloaded_file_{file_id}"

            filepath = os.path.join(output_path, filename)
            
            # Get file size for progress bar
            file_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=file_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(self.CHUNK_SIZE):
                    size = f.write(chunk)
                    pbar.update(size)

            return filepath

        except Exception as e:
            raise ValueError(f"Error downloading file: {str(e)}")
        
    