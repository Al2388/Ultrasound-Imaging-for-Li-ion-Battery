import boto3
import os
import threading
import time
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

class CloudManager:
    def __init__(self):
        self.bucket = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION")
        self.s3 = None
        self.enabled = False

        # Try to connect to AWS
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                region_name=self.region
            )
            self.enabled = True
            print(f"[CLOUD] Connected to Bucket: {self.bucket}")
        except Exception as e:
            print(f"[CLOUD] Connection Failed: {e}")

    def upload_image_async(self, local_path, label):
        """
        Starts upload in a background thread.
        Returns the future URL immediately so the UI doesn't wait.
        """
        if not self.enabled:
            return None

        # Create a unique filename (e.g., amplitude_17000123.png)
        timestamp = int(time.time())
        cloud_filename = f"{label.lower()}_{timestamp}.png"
        
        # Start upload in background
        thread = threading.Thread(
            target=self._upload_worker, 
            args=(local_path, cloud_filename)
        )
        thread.start()

        # Return the URL where the image WILL be
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{cloud_filename}"

    def _upload_worker(self, local_path, cloud_filename):
        try:
            self.s3.upload_file(
                local_path, 
                self.bucket, 
                cloud_filename, 
                ExtraArgs={'ContentType': "image/png", 'ACL': 'public-read'}
            )
            print(f"[CLOUD] Upload Success: {cloud_filename}")
        except Exception as e:
            print(f"[CLOUD] Upload Error: {e}")