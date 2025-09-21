# Capstone/server/app.py - Sweetness Analysis Only (Detection handled by React Native)
import os
import json
import uuid
import datetime
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import tensorflow as tf

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "assets/models/pineapple_classifier/pineapple_sweetness_classifier.keras")
)
DEFAULT_DETECTOR_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "assets/models/pineapple_detector")
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", DEFAULT_DETECTOR_PATH)

# Set class order if needed: e.g. CLASS_ORDER="High,Medium,Low"
CLASS_ORDER_ENV = os.environ.get("CLASS_ORDER")
CLASS_NAMES = [c.strip() for c in CLASS_ORDER_ENV.split(",")] if CLASS_ORDER_ENV else ["High", "Low", "Medium"]

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- DB (SQLite) ----------------
SQLALCHEMY_DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "predictions.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    image_path = Column(String(512), nullable=True)
    is_pineapple = Column(Boolean, nullable=False)
    detection_confidence = Column(Float, nullable=True)
    bounding_boxes = Column(String(1024), nullable=True)  # JSON string of detected boxes
    prediction = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Sweetness Classifier ----------------
class KerasClassifier:
    def __init__(self, model_path: str, class_names: List[str], input_size=(224, 224)):
        if not (os.path.exists(model_path) or os.path.isdir(model_path)):
            raise RuntimeError(f"Model file not found at: {model_path}")
        try:
            # Try tf.keras first (disable safe mode to relax deserialization constraints)
            self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e1:
            # Fallback to standalone keras (Keras 3) if present
            try:
                import keras  # type: ignore
                self.model = keras.models.load_model(model_path, compile=False, safe_mode=False)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model at {model_path}.\n"
                    f"tf.keras error: {e1}\n"
                    f"keras error: {e2}"
                )
            
        self.class_names = class_names
        self.input_size = input_size

    def preprocess(self, img_bytes: bytes) -> np.ndarray:
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize(self.input_size)
        arr = np.array(img).astype("float32") / 255.0  # matches notebook preprocessing
        return np.expand_dims(arr, axis=0)

    def predict(self, img_bytes: bytes) -> Dict[str, Any]:
        x = self.preprocess(img_bytes)
        logits = self.model.predict(x, verbose=0)
        logits = np.squeeze(logits)

        if logits.ndim == 0:
            raise ValueError("Model output is scalar; expected class probabilities/logits.")

        probs = logits
        if not np.allclose(np.sum(probs), 1.0, atol=1e-3):
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)

        if len(probs) != len(self.class_names):
            raise ValueError(f"Model output size {len(probs)} != class mapping size {len(self.class_names)}.")

        best_idx = int(np.argmax(probs))
        return {
            "prediction": self.class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
        }

# ---------------- TensorFlow.js YOLOv8 Pineapple Detector ----------------
class UltralyticsYOLODetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Check for TensorFlow Lite model first (more reliable for YOLOv8)
        tflite_paths = [
            os.path.join(model_path, "best_saved_model", "best_float32.tflite"),
            os.path.join(model_path, "best_float32.tflite"),
            os.path.join(model_path, "model.tflite")
        ]
        
        self.tflite_path = None
        for path in tflite_paths:
            if os.path.exists(path):
                self.tflite_path = path
                print(f"📁 TensorFlow Lite model found: {path}")
                break
        
        # Fallback to SavedModel
        if not self.tflite_path:
            savedmodel_paths = [
                os.path.join(model_path, "best_saved_model"),
                os.path.join(model_path, "saved_model"),
                model_path
            ]
            
            self.savedmodel_path = None
            for path in savedmodel_paths:
                if os.path.exists(os.path.join(path, "saved_model.pb")):
                    self.savedmodel_path = path
                    print(f"📁 SavedModel found: {path}")
                    break
        
        # Fallback to TensorFlow.js format
        if not self.savedmodel_path:
            self.model_json_path = os.path.join(model_path, "model.json")
            if os.path.exists(self.model_json_path):
                weight_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
                if weight_files:
                    print(f"📁 TensorFlow.js model found: {self.model_json_path}")
                    print(f"📦 Weight files: {weight_files}")
                else:
                    raise RuntimeError(f"No .bin weight files found in: {model_path}")
            else:
                raise RuntimeError(f"No model found at: {model_path}. Looking for SavedModel or TensorFlow.js format.")
        
        self.model = None
        self.input_size = (640, 640)  # Match SavedModel signature: 640x640
        # Updated for your trained 3-class model
        self.class_names = ["Ripe", "Semi_Ripe", "Un_Ripe"]
        print(f"🔄 YOLOv8 model will be loaded on first prediction request")
        print(f"🍍 Classes: {self.class_names}")
    
    def _load_model(self):
        """Load the TensorFlow.js model."""
        if self.model is not None:
            return
        
        try:
            # Try loading SavedModel first (if found during initialization)
            if self.savedmodel_path:
                print(f"🔄 Attempting to load SavedModel from: {self.savedmodel_path}")
                self.model = tf.saved_model.load(self.savedmodel_path)
                print(f"✅ SavedModel YOLOv8 model loaded from: {self.savedmodel_path}")
                
                # Test the model with a dummy inference
                print("🧪 Testing model with dummy image...")
                dummy_input = np.random.random((1, 640, 640, 3)).astype(np.float32)  # Match SavedModel signature: 640x640, channels-last
                test_output = self.model(dummy_input)
                print(f"✅ Model test successful - output shape: {test_output.shape}")
                print(f"✅ Real YOLOv8 model is ready for inference!")
            else:
                # Fallback to TensorFlow.js loading (will likely fail but try anyway)
                raise Exception("No SavedModel available, trying TensorFlow.js fallback")
            
        except Exception as e:
            print(f"❌ SavedModel loading failed: {e}")
            print(f"📁 Tried to load from: {self.savedmodel_path if hasattr(self, 'savedmodel_path') and self.savedmodel_path else 'No SavedModel path found'}")
            
            # Fallback: try loading as a Keras model
            try:
                print(f"⚠️ Trying TensorFlow.js fallback...")
                if hasattr(self, 'model_json_path') and os.path.exists(self.model_json_path):
                    import json
                    
                    # Load model architecture
                    with open(self.model_json_path, 'r') as f:
                        model_config = json.load(f)
                    
                    print(f"⚠️ TensorFlow.js model detected. Consider converting to TensorFlow SavedModel format.")
                    print(f"⚠️ Falling back to simulated detection for now.")
                    
                    # Create a dummy model for testing
                    self.model = "tfjs_model_placeholder"
                else:
                    print(f"❌ No TensorFlow.js model found either")
                    raise RuntimeError(f"No valid model found in {self.model_path}")
                
            except Exception as e2:
                print(f"❌ All model loading attempts failed!")
                print(f"SavedModel error: {e}")
                print(f"TensorFlow.js error: {e2}")
                raise RuntimeError(f"Failed to load any model format: SavedModel({e}), TensorFlow.js({e2})")
    
    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess image for YOLOv8 inference - match training preprocessing."""
        print(f"📸 Original image size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # YOLOv8 preprocessing: letterbox resize (maintain aspect ratio)
        target_size = self.input_size[0]  # 640
        
        # Calculate scaling factor and padding
        img_w, img_h = img.size
        scale = min(target_size / img_w, target_size / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # Resize with aspect ratio preserved
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"📸 Resized to: {img_resized.size}")
        
        # Create canvas and paste resized image (letterbox)
        canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))  # Gray padding
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        print(f"📸 Letterboxed with padding at ({paste_x}, {paste_y})")
        
        # Convert to numpy array  
        img_array = np.array(canvas).astype(np.float32)
        
        # Try YOLOv8 normalization: keep 0-255 range (some models expect this)
        # img_array = img_array / 255.0  # Comment out normalization to test
        
        # Keep HWC format (channels last) for TensorFlow SavedModel
        img_batch = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        
        print(f"📸 Final preprocessed shape: {img_batch.shape}")
        print(f"📸 Value range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        
        return img_batch
    
    def _postprocess_predictions(self, predictions: np.ndarray, confidence_threshold: float = 0.25) -> List[Dict]:
        """Postprocess YOLOv8 predictions."""
        detections = []
        
        # YOLOv8 output format: [batch, num_detections, 6]
        # 6 values: [x_center, y_center, width, height, conf_ripen, conf_unripen]
        
        if predictions.ndim == 3:
            batch_predictions = predictions[0]  # Remove batch dimension
        else:
            batch_predictions = predictions
        
        for detection in batch_predictions:
            if len(detection) >= 6:
                x_center, y_center, width, height, conf_ripen, conf_unripen = detection[:6]
                
                # Get the class with higher confidence
                if conf_ripen > conf_unripen:
                    class_name = "ripen"
                    confidence = float(conf_ripen)
                    class_id = 0
                else:
                    class_name = "unripen" 
                    confidence = float(conf_unripen)
                    class_id = 1
                
                # Only keep detections above threshold
                if confidence >= confidence_threshold:
                    # Convert from center format to corner format
                    x1 = float(x_center - width / 2)
                    y1 = float(y_center - height / 2)
                    x2 = float(x_center + width / 2)
                    y2 = float(y_center + height / 2)
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_name,
                        "class_id": class_id
                    })
        
        return detections
    
    def detect(self, img_bytes: bytes) -> Dict[str, Any]:
        # Load model on first use
        self._load_model()
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        
        # Set thresholds
        confidence_threshold = 0.25  # 25% confidence threshold
        detection_threshold = 0.70   # 70% detection threshold (stricter to reduce false positives)
        
        print(f"🔍 Running TensorFlow.js YOLOv8 detection with conf={confidence_threshold}")
        print(f"🎯 Detection threshold: {detection_threshold}")
        
        detections = []
        max_confidence = 0.0
        is_pineapple = False
        all_detections = []
        
        try:
            if self.model == "tfjs_model_placeholder":
                # Simulated detection for TensorFlow.js models
                print("⚠️ Using simulated detection - please convert model to TensorFlow SavedModel format")
                
                # Simulate some detections based on image analysis
                img_array = np.array(img)
                
                # Simple heuristic: look for pineapple-like colors and patterns
                # This is a fallback until proper TensorFlow.js model loading is implemented
                
                # More restrictive pineapple detection to reduce false positives
                # Convert bytes to PIL Image and then to HSV for better color analysis
                pil_img = Image.open(BytesIO(img_bytes))
                img_hsv = np.array(pil_img.convert('HSV'))
                
                # Look for pineapple-specific color patterns in HSV space (more reliable)
                # Pineapple yellow: Hue 40-60, Saturation 100-255, Value 150-255
                # Pineapple brown/texture: Hue 10-30, Saturation 50-200, Value 80-180
                yellow_mask = (
                    (img_hsv[:,:,0] >= 40) & (img_hsv[:,:,0] <= 60) &  # Yellow hue range
                    (img_hsv[:,:,1] >= 100) & (img_hsv[:,:,1] <= 255) &  # High saturation
                    (img_hsv[:,:,2] >= 150) & (img_hsv[:,:,2] <= 255)   # Bright value
                )
                
                brown_mask = (
                    (img_hsv[:,:,0] >= 10) & (img_hsv[:,:,0] <= 30) &   # Brown hue range
                    (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,1] <= 200) &  # Medium saturation
                    (img_hsv[:,:,2] >= 80) & (img_hsv[:,:,2] <= 180)    # Medium-dark value
                )
                
                # Combine masks for pineapple-like regions
                pineapple_mask = yellow_mask | brown_mask
                pineapple_pixels = np.sum(pineapple_mask)
                total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
                pineapple_ratio = pineapple_pixels / total_pixels
                
                # Much more restrictive thresholds to reduce false positives
                # Require at least 25% pineapple-like pixels AND additional validation
                min_pineapple_ratio = 0.25
                
                print(f"🔍 Simulated detection analysis:")
                print(f"  Yellow pixels: {np.sum(yellow_mask)}")
                print(f"  Brown pixels: {np.sum(brown_mask)}")
                print(f"  Total pineapple-like pixels: {pineapple_pixels}")
                print(f"  Pineapple ratio: {pineapple_ratio:.3f}")
                print(f"  Required ratio: {min_pineapple_ratio}")
                
                # Additional shape/size validation
                height, width = img_array.shape[:2]
                aspect_ratio = max(height, width) / min(height, width)
                is_reasonable_size = 200 <= min(height, width) <= 2000  # Reasonable image size
                is_reasonable_aspect = aspect_ratio <= 3.0  # Not too elongated
                
                print(f"  Image size: {height}x{width}")
                print(f"  Aspect ratio: {aspect_ratio:.2f}")
                print(f"  Size check: {is_reasonable_size}")
                print(f"  Aspect check: {is_reasonable_aspect}")
                
                if (pineapple_ratio >= min_pineapple_ratio and 
                    is_reasonable_size and 
                    is_reasonable_aspect):
                    
                    # Calculate confidence based on how much it looks like a pineapple
                    confidence = min(0.75, pineapple_ratio * 2.5)  # Cap at 75% for simulated detection
                    
                    # Determine ripeness based on yellow vs brown ratio
                    yellow_ratio = np.sum(yellow_mask) / total_pixels
                    brown_ratio = np.sum(brown_mask) / total_pixels
                    
                    if yellow_ratio > brown_ratio * 2:  # Mostly yellow
                        class_name, class_id = "Ripe", 0
                    elif yellow_ratio > brown_ratio * 0.5:  # Mixed colors
                        class_name, class_id = "Semi_Ripe", 1
                    else:  # Mostly brown/green
                        class_name, class_id = "Un_Ripe", 2
                    
                    print(f"  ✅ Potential pineapple detected: {class_name} (conf: {confidence:.3f})")
                    
                    detections.append({
                        "bbox": [100.0, 100.0, 500.0, 500.0],  # Simulated bounding box
                        "confidence": confidence,
                        "class": class_name,
                        "class_id": class_id
                    })
                    max_confidence = confidence
                    is_pineapple = confidence >= detection_threshold
                else:
                    print(f"  ❌ Does not look like a pineapple - ratio too low or invalid dimensions")
                    is_pineapple = False
                    max_confidence = 0.0
            else:
                # Real YOLOv8 SavedModel inference
                print("🚀 Running real YOLOv8 SavedModel inference...")
                img_preprocessed = self._preprocess_image(img)
                print(f"📸 Preprocessed image shape: {img_preprocessed.shape}")
                
                # Run inference
                try:
                    # YOLOv8 SavedModel requires specific signature calling
                    print(f"📊 Model signatures: {list(self.model.signatures.keys())}")
                    
                    # Try different ways to call the model
                    if hasattr(self.model, 'signatures') and 'serving_default' in self.model.signatures:
                        # Use serving_default signature
                        infer = self.model.signatures['serving_default']
                        print(f"📊 Using serving_default signature")
                        
                        # Convert to tf.Tensor and get the input name
                        img_tensor = tf.convert_to_tensor(img_preprocessed)
                        
                        # Check signature inputs and outputs
                        signature_def = infer.structured_input_signature[1]
                        input_names = list(signature_def.keys())
                        print(f"📊 Expected input names: {input_names}")
                        
                        # Debug: Show input signature details
                        for name, spec in signature_def.items():
                            print(f"📊 Input '{name}': shape={spec.shape}, dtype={spec.dtype}")
                        
                        if input_names:
                            input_name = input_names[0]
                            print(f"📊 Using input name: {input_name}")
                            predictions = infer(**{input_name: img_tensor})
                        else:
                            predictions = infer(img_tensor)
                    elif hasattr(self.model, '__call__'):
                        # Direct call method
                        print(f"📊 Using direct __call__ method")
                        img_tensor = tf.convert_to_tensor(img_preprocessed)
                        predictions = self.model(img_tensor)
                    else:
                        # Try to find any callable signature
                        signatures = list(self.model.signatures.keys())
                        if signatures:
                            signature_name = signatures[0]
                            print(f"📊 Using signature: {signature_name}")
                            infer = self.model.signatures[signature_name]
                            img_tensor = tf.convert_to_tensor(img_preprocessed)
                            predictions = infer(img_tensor)
                        else:
                            raise Exception("No callable signatures found in SavedModel")
                    
                    print(f"📊 Model predictions type: {type(predictions)}")
                    
                    # Handle different output formats
                    if isinstance(predictions, dict):
                        # If model returns a dictionary, find the output tensor
                        print(f"📊 Model output keys: {list(predictions.keys())}")
                        # Usually the output key is 'output_0' or similar
                        pred_tensor = predictions[list(predictions.keys())[0]]
                    else:
                        # Direct tensor output
                        pred_tensor = predictions
                    
                    print(f"📊 Prediction tensor shape: {pred_tensor.shape}")
                    
                    # Convert to numpy for post-processing
                    pred_numpy = pred_tensor.numpy()
                    print(f"📊 Prediction numpy shape: {pred_numpy.shape}")
                    
                    # Real YOLOv8 post-processing
                    # YOLOv8 output format: (1, 7, 8400) where 7 = [x, y, w, h, conf, class1_conf, class2_conf, ...]
                    # For 3-class model: [x, y, w, h, conf, ripe_conf, semi_ripe_conf, unripe_conf]
                    
                    predictions_array = pred_numpy[0]  # Remove batch dimension: (7, 8400)
                    predictions_array = predictions_array.T  # Transpose to (8400, 7)
                    
                    print(f"📊 Transposed predictions shape: {predictions_array.shape}")
                    
                    # Debug: Show top predictions before filtering
                    max_confidences = []
                    
                    # Filter predictions by confidence threshold
                    for i, prediction in enumerate(predictions_array):
                        if len(prediction) >= 7:  # Ensure we have enough values
                            x_center, y_center, width, height = prediction[:4]
                            obj_conf = prediction[4]  # Objectness confidence
                            
                            # For 3-class model, get class confidences
                            if len(prediction) >= 7:
                                class_confs = prediction[5:8]  # [ripe, semi_ripe, unripe]
                                best_class_idx = np.argmax(class_confs)
                                best_class_conf = class_confs[best_class_idx]
                                
                                # Final confidence = objectness * class_confidence
                                final_confidence = obj_conf * best_class_conf
                                max_confidences.append(final_confidence)
                                
                                # Debug: Show top 5 predictions
                                if i < 5:
                                    class_names = ["Ripe", "Semi_Ripe", "Un_Ripe"]
                                    class_name = class_names[best_class_idx]
                                    print(f"  🔍 Prediction {i}: {class_name} obj={obj_conf:.4f} class={best_class_conf:.4f} final={final_confidence:.4f}")
                                
                                # Only keep detections above confidence threshold 
                                if final_confidence >= 0.3:  # Raised from 0.1 to 0.3 to reduce false positives
                                    # Convert from center format to corner format
                                    x1 = float(x_center - width / 2)
                                    y1 = float(y_center - height / 2)
                                    x2 = float(x_center + width / 2)
                                    y2 = float(y_center + height / 2)
                                    
                                    # Map class index to class name
                                    class_names = ["Ripe", "Semi_Ripe", "Un_Ripe"]
                                    class_name = class_names[best_class_idx]
                                    
                                    detection = {
                                        "bbox": [x1, y1, x2, y2],
                                        "confidence": float(final_confidence),
                                        "class": class_name,
                                        "class_id": int(best_class_idx)
                                    }
                                    
                                    all_detections.append(detection)
                                    
                                    # Apply detection threshold for final results
                                    if final_confidence >= detection_threshold:
                                        detections.append(detection)
                                        max_confidence = max(max_confidence, final_confidence)
                                        is_pineapple = True
                                        print(f"  ✅ Pineapple detected: {class_name} (conf: {final_confidence:.3f})")
                                    else:
                                        print(f"  ❌ Low confidence: {class_name} (conf: {final_confidence:.3f}, required: {detection_threshold})")
                    
                    # Show overall statistics
                    if max_confidences:
                        top_confidence = max(max_confidences)
                        print(f"📊 Best confidence found: {top_confidence:.4f}")
                        print(f"📊 Average confidence: {np.mean(max_confidences):.4f}")
                    else:
                        print(f"📊 No valid predictions found")
                    
                    print(f"📊 Total detections found: {len(all_detections)}")
                    print(f"📊 High-confidence detections: {len(detections)}")
                    
                    if len(detections) > 0:
                        print(f"✅ Real YOLOv8 pineapple detection successful! Best confidence: {max_confidence:.3f}")
                    else:
                        print(f"❌ No pineapples detected above threshold {detection_threshold}")
                    
                except Exception as model_error:
                    print(f"❌ Model inference failed: {model_error}")
                    raise model_error
        
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            # Return no detection on error
            pass
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # ENHANCED FALSE POSITIVE REDUCTION
        # The YOLOv8 model appears to have overfitting issues, so add multiple validation layers
        
        # Layer 1: Basic thresholds (balanced for both false positive reduction and true positive detection)
        high_confidence_threshold = 0.92  # Lowered from 0.95 to 0.92 to allow real pineapples
        min_detections_required = 2       # Lowered back to 2 from 3
        
        # Layer 2: Detection consistency checks
        confident_detection = max_confidence >= high_confidence_threshold
        multiple_detections = len(detections) >= min_detections_required
        
        # Layer 3: Suspicious detection patterns (potential overfitting indicators)
        # Adjusted thresholds to be less aggressive - real pineapples can have many detections too
        extremely_high_detections = len(detections) > 30  # Raised from 20 to 30
        too_many_perfect_scores = len([d for d in detections if d["confidence"] > 0.9995]) > 8  # Raised threshold and count
        
        # Layer 4: Color-based validation as backup check
        backup_color_validation = False
        try:
            # Use the same color analysis from simulated detection as validation
            pil_img = Image.open(BytesIO(img_bytes))
            img_hsv = np.array(pil_img.convert('HSV'))
            
            # Look for pineapple-specific colors
            yellow_mask = (
                (img_hsv[:,:,0] >= 40) & (img_hsv[:,:,0] <= 60) &
                (img_hsv[:,:,1] >= 100) & (img_hsv[:,:,1] <= 255) &
                (img_hsv[:,:,2] >= 150) & (img_hsv[:,:,2] <= 255)
            )
            brown_mask = (
                (img_hsv[:,:,0] >= 10) & (img_hsv[:,:,0] <= 30) &
                (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,1] <= 200) &
                (img_hsv[:,:,2] >= 80) & (img_hsv[:,:,2] <= 180)
            )
            
            pineapple_mask = yellow_mask | brown_mask
            pineapple_ratio = np.sum(pineapple_mask) / (img_hsv.shape[0] * img_hsv.shape[1])
            backup_color_validation = pineapple_ratio >= 0.10  # Lowered from 15% to 10% for real pineapples
            
            print(f"🎨 Color validation: {backup_color_validation} (ratio: {pineapple_ratio:.3f})")
        except Exception as e:
            print(f"⚠️ Color validation failed: {e}")
            backup_color_validation = True  # Don't block if color check fails
        
        # Layer 5: Combine all validation layers
        basic_criteria_met = confident_detection or multiple_detections
        not_suspicious_pattern = not (extremely_high_detections or too_many_perfect_scores)
        
        final_result = (is_pineapple and 
                       basic_criteria_met and 
                       not_suspicious_pattern and 
                       backup_color_validation)
        
        print(f"🎯 FINAL DECISION: {final_result}")
        print(f"🎯 Max confidence: {max_confidence:.3f}")
        print(f"🎯 Total valid detections: {len(detections)}")
        print(f"🎯 High confidence (>92%): {confident_detection}")
        print(f"🎯 Multiple detections (≥2): {multiple_detections}")
        print(f"🎯 Suspicious patterns detected: {extremely_high_detections or too_many_perfect_scores}")
        print(f"🎯   - Too many detections (>30): {extremely_high_detections}")
        print(f"🎯   - Too many perfect scores (>8 with 99.95%+): {too_many_perfect_scores}")
        print(f"🎯 Color validation passed: {backup_color_validation}")
        print(f"🎯 All validation layers: basic={basic_criteria_met}, not_suspicious={not_suspicious_pattern}, color={backup_color_validation}")
        
        return {
            "is_pineapple": bool(final_result),
            "confidence": float(max_confidence),
            "threshold": float(detection_threshold),
            "confidence_threshold": float(confidence_threshold),
            "detections": detections,
            "total_detections": int(len(detections)),
            "all_detections": all_detections,
            "debug_info": {
                "pineapple_detected": bool(is_pineapple),
                "max_confidence": float(max_confidence),
                "meets_threshold": bool(max_confidence >= detection_threshold),
                "final_result": bool(final_result),
                "model_type": "tensorflow_js"
            }
        }

# Initialize classifier only (detection handled by React Native)
try:
    classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)
    print(f"✅ Sweetness classifier loaded from: {MODEL_PATH}")
    print(f"🍯 Classes: {CLASS_NAMES}")
    print(f"🚀 Backend ready for sweetness analysis only (detection handled by React Native)")
    detector = None  # Disabled - detection handled by React Native
except Exception as e:
    print(f"⚠️  Warning: Could not load classifier: {e}")
    print("Classifier will be loaded on first prediction request")
    detector = None
    classifier = None

# ---------------- FastAPI ----------------
app = FastAPI(title="Pineapple Sweetness API - TensorFlow.js Backend Edition", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "sweetness_only",
        "detection_method": "react_native",
        "detector_path": "disabled",
        "classifier_path": MODEL_PATH,
        "class_names": CLASS_NAMES,
        "tf_version": tf.__version__,
        "server_time": datetime.datetime.utcnow().isoformat() + "Z",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, PNG, or WebP.")

    data = await file.read()

    # Save uploaded file for auditing/history
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(data)

    try:
        # Lazy load classifier only (detection handled by React Native)
        global classifier
        if classifier is None:
            print(f"🔄 Loading classifier from: {MODEL_PATH}")
            classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)

        print("🍍 Backend: Sweetness analysis only (detection handled by React Native)")
        
        # Perform sweetness classification directly (assuming valid pineapple image from React Native)
        sweetness_result = classifier.predict(data)
        
        result = {
            "is_pineapple": True,  # Assumed since detection is handled by React Native
            "detection_confidence": None,  # Not applicable - handled by React Native
            "detection_threshold": None,
            "confidence_threshold": None,
            "detections": None,
            "total_detections": None,
            "all_detections": [],
            "debug_info": {"detection_method": "react_native"},
            "prediction": sweetness_result["prediction"],
            "confidence": sweetness_result["confidence"],
            "probabilities": sweetness_result["probabilities"]
        }
            
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    # Store in DB
    row = Prediction(
        image_path=os.path.relpath(save_path, BASE_DIR),
        is_pineapple=result["is_pineapple"],
        detection_confidence=result["detection_confidence"],
        bounding_boxes=json.dumps(result["detections"]) if result["detections"] else None,
        prediction=result.get("prediction"),
        confidence=result.get("confidence"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "id": row.id,
        "timestamp": row.timestamp.isoformat() + "Z",
        "image_path": row.image_path,
        "is_pineapple": result["is_pineapple"],
        "detection_confidence": result["detection_confidence"],
        "detection_threshold": result["detection_threshold"],
        "confidence_threshold": result["confidence_threshold"],
        "detections": result["detections"],
        "total_detections": result["total_detections"],
        "all_detections": result.get("all_detections", []),
        "debug_info": result.get("debug_info", {}),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "probabilities": result.get("probabilities"),
    }

@app.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() + "Z",
            "image_path": r.image_path,
            "is_pineapple": r.is_pineapple,
            "detection_confidence": r.detection_confidence,
            "bounding_boxes": json.loads(r.bounding_boxes) if r.bounding_boxes else None,
            "prediction": r.prediction,
            "confidence": r.confidence,
        }
        for r in rows
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
