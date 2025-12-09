import boto3
import os
import threading
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load credentials from .env file
load_dotenv()

class CloudManager:
    def __init__(self):
        self.bucket = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION")
        self.s3 = None
        self.enabled = False

        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                region_name=self.region
            )
            self.enabled = True
            print(f"[CLOUD] Connected to Bucket: {self.bucket} (Secure Mode)")
        except Exception as e:
            print(f"[CLOUD] Connection Failed: {e}")

    def upload_file_async(self, local_path, cloud_name, content_type):
        """
        1. Generates a secure, temporary Presigned URL.
        2. Uploads the file privately in the background.
        3. Returns the secure URL immediately.
        """
        if not self.enabled:
            return None
        
        # 1. Generate Secure URL (Valid for 1 hour / 3600 seconds)
        try:
            secure_url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': cloud_name},
                ExpiresIn=3600
            )
        except ClientError as e:
            print(f"[CLOUD] Signing Error: {e}")
            return None

        # 2. Start Upload in Background
        thread = threading.Thread(
            target=self._upload_worker, 
            args=(local_path, cloud_name, content_type)
        )
        thread.start()

        return secure_url

    def _upload_worker(self, local_path, cloud_name, content_type):
        try:
            # Upload as PRIVATE (No public-read ACL)
            self.s3.upload_file(
                local_path, 
                self.bucket, 
                cloud_name, 
                ExtraArgs={'ContentType': content_type}
            )
            print(f"[CLOUD] Uploaded Securely: {cloud_name}")
        except Exception as e:
            print(f"[CLOUD] Upload Error: {e}")