# AWS S3 Setup for ML Models

This guide shows how to store your ML models in AWS S3 and load them in Render.

## 🚀 Benefits

- **Smaller Repository**: Models not in GitHub (faster clones)
- **Faster Deploys**: Render builds without large model files
- **Easy Updates**: Update models without redeploying code
- **Cost Effective**: S3 storage is ~$0.023/GB/month

## 📋 Step 1: Create AWS S3 Bucket

1. **Go to AWS Console** → S3
2. **Create Bucket**:
   - Name: `your-pineapple-models` (must be globally unique)
   - Region: `us-east-1` (recommended for Render)
   - Keep defaults for other settings
3. **Create folder structure**:
   ```
   your-pineapple-models/
   ├── models/
   │   ├── pineapple_sweetness_classifier.keras
   │   └── pineapple_detector.pt
   ```

## 📤 Step 2: Upload Your Models

### Option A: AWS Console (Web)
1. Go to your S3 bucket
2. Click "Upload" → "Add files"
3. Upload both model files to `models/` folder

### Option B: AWS CLI
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Upload models
aws s3 cp pineapple_sweetness_classifier.keras s3://your-pineapple-models/models/
aws s3 cp pineapple_detector.pt s3://your-pineapple-models/models/
```

## 🔑 Step 3: Create AWS IAM User

1. **Go to AWS Console** → IAM → Users
2. **Create User**:
   - Name: `render-pineapple-models`
   - Access type: Programmatic access
3. **Attach Policy**: Create custom policy:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject"
               ],
               "Resource": "arn:aws:s3:::your-pineapple-models/*"
           }
       ]
   }
   ```
4. **Save Credentials**: Download the Access Key ID and Secret Access Key

## ⚙️ Step 4: Configure Render Environment Variables

In your Render dashboard, add these environment variables:

```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
S3_BUCKET_NAME=your-pineapple-models
S3_CLASSIFIER_KEY=models/pineapple_sweetness_classifier.keras
S3_DETECTOR_KEY=models/pineapple_detector.pt
```

## 📁 Step 5: Remove Models from GitHub

After uploading to S3, remove model files from your repository:

```bash
# Remove model files from git (but keep locally)
git rm -r assets/models/
git commit -m "Remove model files - now using S3"
git push origin main
```

## 🧪 Step 6: Test Setup

1. **Deploy to Render** with the new configuration
2. **Check logs** for S3 download messages:
   ```
   🔄 Attempting to download models from S3...
   📥 Downloading models/pineapple_sweetness_classifier.keras from S3 bucket...
   ✅ Downloaded to /opt/render/project/src/assets/models/...
   🎉 All models downloaded successfully from S3!
   ```
3. **Test endpoints**: `/health` and `/predict`

## 💰 Cost Estimation

- **S3 Storage**: ~$0.023/GB/month
- **Data Transfer**: $0.09/GB (first 1GB free)
- **Requests**: $0.0004 per 1,000 requests

**Monthly cost for 100MB models**: ~$0.002 (basically free!)

## 🔧 Troubleshooting

### Models Not Downloading
- Check AWS credentials in Render environment variables
- Verify S3 bucket name and file paths
- Check IAM permissions

### Slow Startup
- Models download on first request (lazy loading)
- Consider pre-warming the service

### Memory Issues
- S3 models still use same memory when loaded
- Lazy loading helps with startup memory
- Consider model quantization for further reduction

## 🎯 Alternative: Use Local Models

If S3 setup fails, the app will fall back to local models (if present in the repository).

## 📚 Next Steps

1. Set up S3 bucket and upload models
2. Configure Render environment variables
3. Deploy and test
4. Remove models from GitHub repository
5. Enjoy faster deploys and smaller repo!
