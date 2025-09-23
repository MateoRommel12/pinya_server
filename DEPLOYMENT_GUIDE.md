# 🚀 Hugging Face Spaces Deployment Guide

This guide will help you deploy your Pineapple Detection app to Hugging Face Spaces.

## 📋 Prerequisites

- ✅ Hugging Face account ([huggingface.co](https://huggingface.co))
- ✅ GitHub account (recommended for easy deployment)
- ✅ Your ML models ready (already in the repository)

## 🎯 Step-by-Step Deployment

### **Step 1: Create Hugging Face Account**

1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up" 
3. **Recommended**: Sign up with your GitHub account for easy integration
4. Verify your email address

### **Step 2: Create New Space**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   ```
   Space name: pineapple-detection
   License: MIT
   SDK: Gradio
   Hardware: CPU Basic (free)
   Visibility: Public
   ```
4. Click **"Create Space"**

### **Step 3: Upload Your Files**

#### **Option A: Direct Upload (Manual)**

1. **Upload `app.py`** - Your main Gradio application
2. **Upload `requirements.txt`** - Python dependencies
3. **Upload `README.md`** - Documentation (optional)
4. **Create `models/` folder and upload**:
   - `pineapple_sweetness_classifier.keras`
   - `pineapple_detector.pt`

#### **Option B: GitHub Integration (Recommended)**

1. In your Space settings, go to **"Repository"** tab
2. Click **"Connect to GitHub"**
3. Select your repository: `MateoRommel12/pinya_server`
4. Choose **"Auto-deploy"** - HF will deploy automatically when you push to GitHub

### **Step 4: Verify File Structure**

Your Space should have this structure:
```
├── app.py                           # Main Gradio application
├── requirements.txt                 # Python dependencies  
├── README.md                       # Documentation
└── models/                         # ML models directory
    ├── pineapple_sweetness_classifier.keras
    └── pineapple_detector.pt
```

### **Step 5: Deploy & Test**

1. **Auto-deploy**: If using GitHub integration, push to GitHub and HF will deploy automatically
2. **Manual deploy**: Click "Deploy" button in your Space
3. **Wait for build**: Usually takes 2-5 minutes
4. **Test your app**: Use the web interface to upload and test images

## 🎨 Your App Features

Once deployed, your app will have:

- **🖼️ Image Upload**: Drag & drop interface
- **🔄 Auto-Analysis**: Analyzes images automatically
- **🍍 Pineapple Detection**: Uses YOLOv8 to detect pineapples
- **🍯 Sweetness Classification**: Predicts High/Medium/Low sweetness
- **📊 Confidence Scores**: Shows detection and classification confidence
- **📱 Mobile Friendly**: Works on phones and tablets

## 🔗 Sharing Your App

Once deployed, you can:

- **Share the link**: `https://huggingface.co/spaces/YOUR_USERNAME/pineapple-detection`
- **Embed in websites**: Use the embed code provided by HF
- **Share on social media**: HF provides easy sharing options

## 💰 Cost Information

**Free Tier Includes:**
- ✅ 16GB RAM (vs 512MB on Render!)
- ✅ No cold starts
- ✅ Unlimited usage
- ✅ Public sharing
- ✅ Community features

**Paid Tiers Available:**
- **CPU Upgrade**: $0.60/hour for more CPU power
- **GPU Access**: $1.05/hour for GPU acceleration
- **Private Spaces**: $9/month for private repositories

## 🛠️ Troubleshooting

### **Build Fails**
- Check `requirements.txt` for correct package names
- Verify model files are in `models/` directory
- Check Space logs for error messages

### **Models Not Loading**
- Ensure model files are in correct location
- Check file permissions
- Verify model file names match the code

### **App Not Working**
- Check the Space logs for error messages
- Test with simple images first
- Verify all dependencies are installed

## 🎉 Success!

Once deployed successfully, you'll have:

- ✅ **Fast deployment** (no memory issues like Render)
- ✅ **Beautiful UI** with Gradio
- ✅ **Easy sharing** with the community
- ✅ **No cold starts** - always running
- ✅ **16GB RAM** - plenty of memory for ML models

## 📚 Next Steps

1. **Test thoroughly** with different pineapple images
2. **Share with others** using the HF Spaces link
3. **Gather feedback** from the community
4. **Iterate and improve** your model based on feedback

## 🔗 Useful Links

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Your GitHub Repository](https://github.com/MateoRommel12/pinya_server)

---

**Need help?** Check the Hugging Face documentation or ask in their community forums!
