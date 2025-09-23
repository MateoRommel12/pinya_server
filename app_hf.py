# Hugging Face Spaces - Pineapple Detection + Sweetness Analysis
import os
import json
import uuid
import datetime
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "models/pineapple_sweetness_classifier.keras")
)
DEFAULT_DETECTOR_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "models/pineapple_detector.pt")
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", DEFAULT_DETECTOR_PATH)

# Set class order if needed: e.g. CLASS_ORDER="High,Medium,Low"
CLASS_ORDER_ENV = os.environ.get("CLASS_ORDER")
CLASS_NAMES = [c.strip() for c in CLASS_ORDER_ENV.split(",")] if CLASS_ORDER_ENV else ["High", "Low", "Medium"]

# ---------------- Keras Sweetness Classifier ----------------
class KerasClassifier:
    def __init__(self, model_path: str, class_names: List[str]):
        self.model_path = model_path
        self.class_names = class_names
        self.model = None
        self.input_size = (224, 224)
        print(f"🍯 Keras classifier initialized for: {class_names}")
        print(f"📁 Model path: {model_path}")

    def _load_model(self):
        """Load the Keras model."""
        if self.model is not None:
            return
        
        try:
            print(f"🔄 Loading Keras model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Keras model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Keras model: {e}")
            raise RuntimeError(f"Failed to load .keras model from {self.model_path}: {e}")

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Predict sweetness from image data."""
        self._load_model()
        
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_data))
            image = image.convert('RGB')
            image = image.resize(self.input_size)
            
            # Convert to array and normalize
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            probs = self.model.predict(img_array, verbose=0)[0]
            best_idx = np.argmax(probs)
            
            return {
                "prediction": self.class_names[best_idx],
                "confidence": float(probs[best_idx]),
                "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
            }
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "error": str(e)
            }

# ---------------- YOLOv8 Pineapple Detector (.pt file) ----------------
class UltralyticsYOLODetector:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # Look for .pt file in the directory
        pt_files = [
            os.path.join(model_dir, "pineapple_detector.pt"),
            os.path.join(model_dir, "best.pt"),
            os.path.join(model_dir, "model.pt")
        ]
        
        self.pt_path = None
        for path in pt_files:
            if os.path.exists(path):
                self.pt_path = path
                print(f"📁 Found .pt model: {path}")
                break
        
        if not self.pt_path:
            raise RuntimeError(f"No .pt model found in: {model_dir}. Looking for pineapple_detector.pt, best.pt, or model.pt")
        
        self.model = None
        self.input_size = (640, 640)
        self.class_names = ["Ripe", "Semi_Ripe", "Un_Ripe"]
        print(f"🔄 YOLOv8 .pt model will be loaded on first prediction request")
        print(f"🍍 Classes: {self.class_names}")
    
    def _load_model(self):
        """Load the YOLOv8 .pt model using ultralytics."""
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            import torch
            
            print(f"🔄 Loading YOLOv8 model from: {self.pt_path}")
            
            try:
                print("🔧 Attempting to load with weights_only=False...")
                
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules.conv.Conv',
                    'ultralytics.nn.modules.block.C2f',
                    'ultralytics.nn.modules.head.Detect'
                ])
                
                self.model = YOLO(self.pt_path)
                print(f"✅ YOLOv8 model loaded successfully!")
                
            except Exception as load_error:
                print(f"⚠️ Standard loading failed: {load_error}")
                print("🔧 Trying alternative loading method...")
                
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                try:
                    self.model = YOLO(self.pt_path)
                    print(f"✅ YOLOv8 model loaded with patched torch.load!")
                finally:
                    torch.load = original_load
            
            print("🧪 Testing model with dummy inference...")
            from PIL import Image
            
            dummy_img = Image.new('RGB', (640, 640), color='white')
            
            results = self.model(dummy_img, conf=0.1, verbose=False)
            print(f"✅ Model test successful - ready for detection!")
            
        except ImportError as e:
            print(f"❌ ultralytics package not found: {e}")
            raise RuntimeError("ultralytics package required. Install with: pip install ultralytics")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 model: {e}")
            raise RuntimeError(f"Failed to load .pt model from {self.pt_path}: {e}")

    def detect(self, image_data: bytes, detection_threshold: float = 0.5, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Detect pineapples in image data."""
        self._load_model()
        
        try:
            # Load image
            image = Image.open(BytesIO(image_data))
            image = image.convert('RGB')
            
            # Run detection
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            detections = []
            all_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(confidence),
                            "class": self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
                        }
                        
                        all_detections.append(detection)
                        
                        # Filter by detection threshold
                        if confidence >= detection_threshold:
                            detections.append(detection)
            
            # Determine if pineapple is detected
            is_pineapple = len(detections) > 0
            max_confidence = max([d["confidence"] for d in all_detections]) if all_detections else 0.0
            
            return {
                "is_pineapple": bool(is_pineapple),
                "confidence": float(max_confidence),
                "threshold": float(detection_threshold),
                "confidence_threshold": float(confidence_threshold),
                "detections": detections,
                "total_detections": int(len(detections)),
                "all_detections": all_detections,
                "debug_info": {
                    "model_type": "ultralytics_yolov8",
                    "model_path": self.pt_path,
                    "total_found": len(all_detections),
                    "high_confidence": len(detections)
                }
            }
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return {
                "is_pineapple": False,
                "confidence": 0.0,
                "threshold": float(detection_threshold),
                "confidence_threshold": float(confidence_threshold),
                "detections": [],
                "total_detections": 0,
                "all_detections": [],
                "debug_info": {"error": str(e)}
            }

# Initialize models with lazy loading for memory optimization
detector = None
classifier = None
print("🔄 Models will be loaded on first prediction request to save memory")

def get_detector():
    """Lazy load detector model"""
    global detector
    if detector is None:
        try:
            detector = UltralyticsYOLODetector(DETECTOR_PATH)
            print(f"✅ Pineapple detector loaded from: {DETECTOR_PATH}")
        except Exception as e:
            print(f"❌ Failed to load detector: {e}")
            return None
    return detector

def get_classifier():
    """Lazy load classifier model"""
    global classifier
    if classifier is None:
        try:
            classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)
            print(f"✅ Sweetness classifier loaded from: {MODEL_PATH}")
            print(f"🍯 Classes: {CLASS_NAMES}")
        except Exception as e:
            print(f"❌ Failed to load classifier: {e}")
            return None
    return classifier

# ---------------- Gradio Interface ----------------
def predict_pineapple(image):
    """Main prediction function for Gradio interface."""
    if image is None:
        return "Please upload an image", None
    
    try:
        # Convert PIL image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()
        
        # Get models
        detector_model = get_detector()
        classifier_model = get_classifier()
        
        if detector_model is None or classifier_model is None:
            return "❌ Failed to load ML models", None
        
        # Step 1: Detect if image contains pineapple
        detection_result = detector_model.detect(image_data, detection_threshold=0.5, confidence_threshold=0.3)
        
        result_text = f"🍍 **Pineapple Detection:**\n"
        result_text += f"- Detected: {'✅ Yes' if detection_result['is_pineapple'] else '❌ No'}\n"
        result_text += f"- Confidence: {detection_result['confidence']:.2%}\n"
        result_text += f"- Detections: {detection_result['total_detections']}\n\n"
        
        if detection_result['is_pineapple']:
            # Step 2: Analyze sweetness
            sweetness_result = classifier_model.predict(image_data)
            
            result_text += f"🍯 **Sweetness Analysis:**\n"
            result_text += f"- Prediction: {sweetness_result['prediction']}\n"
            result_text += f"- Confidence: {sweetness_result['confidence']:.2%}\n\n"
            
            result_text += f"📊 **Probabilities:**\n"
            for class_name, prob in sweetness_result['probabilities'].items():
                result_text += f"- {class_name}: {prob:.2%}\n"
        else:
            result_text += "🍯 **Sweetness Analysis:** Skipped (no pineapple detected)"
        
        return result_text, image
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Pineapple Detection & Sweetness Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🍍 Pineapple Detection & Sweetness Analysis
    
    Upload an image to detect pineapples and analyze their sweetness level!
    
    **Features:**
    - 🎯 **Detection**: Uses YOLOv8 to detect pineapples
    - 🍯 **Sweetness Classification**: Predicts sweetness level (High/Medium/Low)
    - 📊 **Confidence Scores**: Shows detection and classification confidence
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Pineapple Image",
                type="pil",
                height=400
            )
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Markdown(label="Analysis Results")
            output_image = gr.Image(label="Processed Image", height=400)
    
    # Event handlers
    predict_btn.click(
        fn=predict_pineapple,
        inputs=[input_image],
        outputs=[output_text, output_image]
    )
    
    # Auto-predict on image upload
    input_image.change(
        fn=predict_pineapple,
        inputs=[input_image],
        outputs=[output_text, output_image]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
