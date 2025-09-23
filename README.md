# Hugging Face Spaces - Pineapple Detection & Sweetness Analysis

This is a Hugging Face Spaces version of the Pineapple Detection backend using Gradio for a web interface.

## 🚀 Features

- **🍍 Pineapple Detection**: Uses YOLOv8 to detect pineapples in images
- **🍯 Sweetness Classification**: Predicts sweetness level (High/Medium/Low)
- **📊 Interactive Web Interface**: Built with Gradio
- **⚡ Real-time Analysis**: Instant results with confidence scores

## 📁 File Structure

```
├── app_hf.py              # Main Gradio application
├── requirements_hf.txt    # Python dependencies
├── README_HF.md          # This file
└── models/               # ML models directory
    ├── pineapple_sweetness_classifier.keras
    └── pineapple_detector.pt
```

## 🛠️ Setup for Hugging Face Spaces

### 1. Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: `pineapple-detection`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) or GPU (paid)

### 2. Upload Files

Upload these files to your Space:
- `app_hf.py` → Rename to `app.py`
- `requirements_hf.txt` → Rename to `requirements.txt`
- Your model files to `models/` directory

### 3. Model Files

Place your ML models in the `models/` directory:
```
models/
├── pineapple_sweetness_classifier.keras
└── pineapple_detector.pt
```

### 4. Deploy

The Space will automatically build and deploy! 🎉

## 🎯 Usage

1. **Upload Image**: Click on the image upload area
2. **Auto-Analysis**: The app automatically analyzes the image
3. **View Results**: See detection and sweetness analysis results
4. **Try Different Images**: Upload more images to test

## 📊 Output Format

The app provides:
- **Detection Status**: Whether pineapple was detected
- **Confidence Score**: Detection confidence percentage
- **Sweetness Prediction**: High/Medium/Low classification
- **Probability Breakdown**: Confidence for each sweetness level

## 🔧 Customization

### Environment Variables

You can set these in your Space settings:
- `MODEL_PATH`: Path to sweetness classifier model
- `DETECTOR_PATH`: Path to detector model directory
- `CLASS_ORDER`: Custom class order (e.g., "High,Medium,Low")

### Model Thresholds

Modify detection thresholds in `app_hf.py`:
```python
detection_result = detector_model.detect(
    image_data, 
    detection_threshold=0.5,    # Minimum confidence for detection
    confidence_threshold=0.3    # Minimum confidence for bounding boxes
)
```

## 💰 Cost Comparison

| Platform | Free Tier | RAM | Cold Starts | ML Focus |
|----------|-----------|-----|-------------|----------|
| **Hugging Face** | ✅ | 16GB | ❌ No | ✅ Yes |
| Render | ✅ | 512MB | ✅ Yes | ❌ No |
| Railway | ✅ | 1GB | ❌ No | ❌ No |

## 🚀 Advantages of Hugging Face Spaces

1. **🎯 ML-Optimized**: Built specifically for ML applications
2. **💾 More Memory**: 16GB RAM vs 512MB on Render
3. **⚡ No Cold Starts**: Always running on free tier
4. **🔗 Easy Sharing**: Built-in community features
5. **📦 Model Hub**: Can use models from Hugging Face Hub
6. **🆓 Generous Free Tier**: More resources than other platforms

## 🔄 Migration from Render

To migrate from Render to Hugging Face Spaces:

1. **Copy Models**: Upload your `.keras` and `.pt` files
2. **Use Gradio Interface**: Replace FastAPI with Gradio
3. **Update Dependencies**: Use `requirements_hf.txt`
4. **Deploy**: Create Space and upload files

## 📚 Next Steps

1. Create Hugging Face Space
2. Upload model files
3. Deploy the application
4. Share with community!

## 🆘 Troubleshooting

### Models Not Loading
- Check file paths in `models/` directory
- Verify model file names match the code
- Check Space logs for error messages

### Memory Issues
- Use lazy loading (already implemented)
- Consider model quantization
- Upgrade to GPU tier if needed

### Slow Performance
- Models load on first request
- Consider pre-warming the Space
- Use smaller model variants
