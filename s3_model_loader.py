"""
S3 Model Loader for Render Deployment
Downloads ML models from AWS S3 bucket on startup
"""
import os
import boto3
from botocore.exceptions import ClientError

def download_model_from_s3(bucket_name: str, s3_key: str, local_path: str, aws_access_key_id: str = None, aws_secret_access_key: str = None):
    """
    Download a model file from S3 to local storage
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key (path in bucket)
        local_path: Local file path to save to
        aws_access_key_id: AWS access key (optional, can use env vars)
        aws_secret_access_key: AWS secret key (optional, can use env vars)
    """
    try:
        # Create S3 client
        if aws_access_key_id and aws_secret_access_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Use environment variables or IAM role
            s3_client = boto3.client('s3')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        print(f"📥 Downloading {s3_key} from S3 bucket {bucket_name}...")
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"✅ Downloaded to {local_path}")
        
        return True
        
    except ClientError as e:
        print(f"❌ S3 download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def setup_models_from_s3():
    """
    Download all required models from S3
    Set these environment variables in Render:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - S3_BUCKET_NAME
    - S3_CLASSIFIER_KEY (e.g., "models/pineapple_sweetness_classifier.keras")
    - S3_DETECTOR_KEY (e.g., "models/pineapple_detector.pt")
    """
    # Get environment variables
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not bucket_name:
        print("⚠️ S3_BUCKET_NAME not set, skipping S3 model download")
        return False
    
    # Define model paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "assets", "models")
    
    # Classifier model
    classifier_s3_key = os.environ.get("S3_CLASSIFIER_KEY", "models/pineapple_sweetness_classifier.keras")
    classifier_local_path = os.path.join(models_dir, "pineapple_classifier", "pineapple_sweetness_classifier.keras")
    
    # Detector model
    detector_s3_key = os.environ.get("S3_DETECTOR_KEY", "models/pineapple_detector.pt")
    detector_local_path = os.path.join(models_dir, "pineapple_detector", "pineapple_detector.pt")
    
    success = True
    
    # Download classifier
    if not download_model_from_s3(bucket_name, classifier_s3_key, classifier_local_path, aws_access_key, aws_secret_key):
        success = False
    
    # Download detector
    if not download_model_from_s3(bucket_name, detector_s3_key, detector_local_path, aws_access_key, aws_secret_key):
        success = False
    
    if success:
        print("🎉 All models downloaded successfully from S3!")
    else:
        print("⚠️ Some models failed to download from S3")
    
    return success

if __name__ == "__main__":
    setup_models_from_s3()
