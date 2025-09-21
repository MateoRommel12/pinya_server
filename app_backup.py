# Capstone/server/app.py
import os
import json
import uuid
import datetime
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import tensorflow as tf

# Fix PyTorch 2.6+ compatibility for ultralytics
try:
    import torch
    import torch.serialization
    
    # Set environment variables to disable weights_only security
    os.environ['PYTORCH_WEIGHTS_ONLY_DISABLED'] = '1'
    os.environ['TORCH_WEIGHTS_ONLY'] = '0'
    print("🔧 Set PyTorch compatibility environment variables")
    
    # Try to add safe globals if available
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            # Try to import actual ultralytics classes
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.nn.modules.head import Detect
            from ultralytics.nn.modules.conv import Conv
            from ultralytics.nn.modules.block import (
                Bottleneck, C2f, SPPF, RepC3, C3, C2, SPP, Focus,
                GhostBottleneck, GhostConv, GhostC2f, GhostC3, GhostSPP, GhostSPPF
            )
            
            torch.serialization.add_safe_globals([
                DetectionModel, Detect, Conv, Bottleneck, C2f, SPPF, RepC3, C3, C2, SPP, Focus,
                GhostBottleneck, GhostConv, GhostC2f, GhostC3, GhostSPP, GhostSPPF
            ])
            print("🔧 Added ultralytics classes to PyTorch safe globals")
        except ImportError:
            print("⚠️ Could not import ultralytics classes directly, skipping safe globals")
    else:
        print("⚠️ add_safe_globals not available")
        
except ImportError:
    print("⚠️ PyTorch not available, ultralytics compatibility fix skipped")
except Exception as e:
    print(f"⚠️ PyTorch compatibility fix failed: {e}")

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "assets/models/pineapple_classifier/pineapple_sweetness_classifier.keras")
)
DEFAULT_DETECTOR_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "assets/models/pineapple_detector/pineapple_detector_backend.pt")
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", DEFAULT_DETECTOR_PATH)

# Set class order if needed: e.g. CLASS_ORDER="High,Medium,Low"
CLASS_ORDER_ENV = os.environ.get("CLASS_ORDER")
CLASS_NAMES = [c.strip() for c in CLASS_ORDER_ENV.split(",")] if CLASS_ORDER_ENV else ["High", "Low", "Medium"]

# Pineapple detection threshold
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.70"))  # Very high threshold - 70% to reduce false positives
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))  # Very high confidence - 60% to be more strict

# Manual override for testing (set to True to force pineapple detection)
FORCE_PINEAPPLE_DETECTION = os.environ.get("FORCE_PINEAPPLE_DETECTION", "false").lower() == "true"

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

# ---------------- TensorFlow.js Pineapple Detector ----------------
class TensorFlowJSPineappleDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise RuntimeError(f"TensorFlow.js model file not found at: {model_path}")
        
        # Check file size and permissions
        try:
            file_size = os.path.getsize(model_path)
            print(f"📁 TensorFlow.js model file found: {model_path} ({file_size / (1024*1024):.1f} MB)")
        except Exception as e:
            print(f"⚠️ Warning: Could not check model file: {e}")
        
        self.model = None
        self.model_path = model_path
        self.class_names = ["ripen", "unripen"]  # From metadata.yaml
        print(f"🔄 TensorFlow.js model will be loaded on first prediction request from: {model_path}")
    
    def _load_model(self):
        """Load the TensorFlow.js model."""
        if self.model is not None:
            return
        
        try:
            import json
            
            # Load the model.json file
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            
            # For now, we'll use a simplified approach
            # In a real implementation, you'd need to load the TensorFlow.js model properly
            print(f"✅ TensorFlow.js model metadata loaded from: {self.model_path}")
            self.model = model_data  # Store the model data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow.js model: {e}")
    
    def detect(self, img_bytes: bytes) -> Dict[str, Any]:
        # Load model on first use
        self._load_model()
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        
        # For now, we'll use a simplified detection approach
        # In a real implementation, you'd run the TensorFlow.js model inference
        
        # Simulate detection based on image characteristics
        detections = []
        max_confidence = 0.0
        is_pineapple = False
        
        # Basic image analysis to determine if it might be a pineapple
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # IMPROVED DETECTION: Use multiple criteria to detect pineapples
        confidence = self._estimate_pineapple_likelihood(img_array)
        
        # Manual override for testing
        if FORCE_PINEAPPLE_DETECTION:
            confidence = 0.8
            print(f"🔍 Detection Analysis (FORCED):")
            print(f"  ⚠️  FORCE_PINEAPPLE_DETECTION is enabled!")
        else:
            print(f"🔍 Detection Analysis:")
        
        print(f"  Image size: {height}x{width}")
        print(f"  Estimated confidence: {confidence:.3f}")
        print(f"  Detection threshold: {DETECTION_THRESHOLD}")
        print(f"  Meets threshold: {confidence >= DETECTION_THRESHOLD}")
        
        if confidence >= DETECTION_THRESHOLD:
            is_pineapple = True
            max_confidence = confidence
            
            # Determine if it's ripen or unripen based on color analysis
            class_name = self._determine_ripeness(img_array)
            
            print(f"  ✅ Pineapple detected! Class: {class_name}")
            
            detections.append({
                "bbox": [0.1, 0.1, 0.8, 0.8],  # Simplified bounding box
                "confidence": confidence,
                "class": class_name,
                "class_id": 0 if class_name == "ripen" else 1
            })
        else:
            print(f"  ❌ No pineapple detected (confidence too low)")
        
        return {
            "is_pineapple": is_pineapple and max_confidence >= DETECTION_THRESHOLD,
            "confidence": max_confidence,
            "threshold": DETECTION_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "detections": detections,
            "total_detections": len(detections),
            "all_detections": detections,  # Same as detections for now
            "debug_info": {
                "pineapple_detected": is_pineapple,
                "max_confidence": max_confidence,
                "meets_threshold": max_confidence >= DETECTION_THRESHOLD,
                "final_result": is_pineapple and max_confidence >= DETECTION_THRESHOLD
            }
        }
    
    def _estimate_pineapple_likelihood(self, img_array: np.ndarray) -> float:
        """Estimate likelihood that image contains a pineapple."""
        try:
            height, width = img_array.shape[:2]
            
            # Start with very low likelihood (most images are NOT pineapples)
            likelihood = 0.05
            
            # Check aspect ratio (pineapples are typically taller than wide)
            aspect_ratio = height / width
            if 1.1 <= aspect_ratio <= 3.0:  # Broader range for pineapples
                likelihood += 0.1
            
            # Check for pineapple-like colors
            if len(img_array.shape) == 3:
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                # Define range for golden/yellow colors (ripe pineapple)
                lower_golden = np.array([10, 40, 40])
                upper_golden = np.array([40, 255, 255])
                
                # Define range for green colors (unripe pineapple)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                
                # Define range for brown colors (very ripe pineapple)
                lower_brown = np.array([0, 30, 30])
                upper_brown = np.array([20, 255, 255])
                
                # Create masks for different pineapple colors
                golden_mask = cv2.inRange(hsv, lower_golden, upper_golden)
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                
                golden_ratio = np.sum(golden_mask > 0) / (height * width)
                green_ratio = np.sum(green_mask > 0) / (height * width)
                brown_ratio = np.sum(brown_mask > 0) / (height * width)
                
                # Check for pineapple-like color patterns
                pineapple_colors = golden_ratio + green_ratio + brown_ratio
                
                if pineapple_colors > 0.25:  # At least 25% pineapple-like colors (more strict)
                    likelihood += 0.2
                    if pineapple_colors > 0.4:  # Very high pineapple colors (40%+)
                        likelihood += 0.25
                
                # Check for skin-like texture patterns
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                texture_std = np.std(gray)
                
                # Pineapples have distinctive texture patterns
                if 15 < texture_std < 70:  # Broader texture range
                    likelihood += 0.1
                
                # Skip complex contour analysis to avoid crashes
                # Just rely on color and texture analysis
            
            # Return likelihood with reasonable bounds
            return min(max(likelihood, 0), 0.9)  # Cap at 0.9
            
        except Exception as e:
            print(f"⚠️ Error in likelihood estimation: {e}")
            return 0.2  # Return neutral likelihood on error
    
    def _determine_ripeness(self, img_array: np.ndarray) -> str:
        """Determine if pineapple is ripen or unripen based on color analysis."""
        try:
            if len(img_array.shape) != 3:
                return "unripen"
            
            # Convert to HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Define ranges for ripen (golden/brown) vs unripen (green) colors
            lower_golden = np.array([15, 50, 50])
            upper_golden = np.array([35, 255, 255])
            
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            
            # Count golden vs green pixels
            golden_mask = cv2.inRange(hsv, lower_golden, upper_golden)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            golden_count = np.sum(golden_mask > 0)
            green_count = np.sum(green_mask > 0)
            
            # If more golden than green, consider it ripen
            return "ripen" if golden_count > green_count else "unripen"
            
        except Exception as e:
            print(f"⚠️ Error in ripeness determination: {e}")
            return "unripen"

# ---------------- YOLOv8 Pineapple Detector ----------------
class YOLOv8PineappleDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise RuntimeError(f"YOLOv8 model file not found at: {model_path}")
        
        # Check file size and permissions
        try:
            file_size = os.path.getsize(model_path)
            print(f"📁 Model file found: {model_path} ({file_size / (1024*1024):.1f} MB)")
        except Exception as e:
            print(f"⚠️ Warning: Could not check model file: {e}")
        
        self.model = None
        self.model_path = model_path
        print(f"🔄 YOLOv8 model will be loaded on first prediction request from: {model_path}")
    
    def _load_model(self):
        """Load the YOLOv8 model with proper PyTorch compatibility handling."""
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            import torch
            
            # Set environment variables for PyTorch compatibility
            os.environ['PYTORCH_WEIGHTS_ONLY_DISABLED'] = '1'
            os.environ['TORCH_WEIGHTS_ONLY'] = '0'
            
            # Patch torch.load to always use weights_only=False
            original_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = safe_load
            
            try:
                self.model = YOLO(self.model_path)
                print(f"✅ YOLOv8 model loaded from: {self.model_path}")
                
                # Test the model with a simple inference
                print("🧪 Testing model with dummy image...")
                import numpy as np
                from PIL import Image
                dummy_img = Image.new('RGB', (640, 640), color='white')
                test_results = self.model(dummy_img, conf=0.1, verbose=False)
                print(f"✅ Model test successful - ready for detection")
                
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}")
                # Restore original torch.load
                torch.load = original_load
                raise e
            finally:
                # Restore original torch.load
                torch.load = original_load
                    
        except ImportError:
            raise RuntimeError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")
    
    def detect(self, img_bytes: bytes) -> Dict[str, Any]:
        # Load model on first use
        self._load_model()
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        
        # Run YOLOv8 detection
        print(f"🔍 Running YOLOv8 detection with conf={CONFIDENCE_THRESHOLD}")
        print(f"🎯 Detection threshold: {DETECTION_THRESHOLD}")
        results = self.model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        print(f"📊 YOLOv8 returned {len(results)} result(s)")
        
        # Debug: Run with very low confidence to see ALL detections
        debug_results = self.model(img, conf=0.01, verbose=False)
        print(f"🐛 DEBUG: YOLOv8 with 1% confidence returned {len(debug_results)} result(s)")
        for debug_result in debug_results:
            if debug_result.boxes is not None:
                print(f"🐛 DEBUG: Found {len(debug_result.boxes)} total boxes (including low confidence)")
                for i, box in enumerate(debug_result.boxes):
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = debug_result.names[class_id]
                    print(f"  🐛 Debug Box {i+1}: class={class_name}, confidence={confidence:.3f}")
        print(f"🐛 DEBUG: End of low-confidence detection results")
        
        # Process results
        detections = []
        max_confidence = 0.0
        is_pineapple = False
        all_detections = []  # Track all detections for debugging
        
        for result in results:
            print(f"🔍 Processing result: boxes={result.boxes is not None}")
            if result.boxes is not None:
                print(f"📦 Found {len(result.boxes)} boxes")
                for i, box in enumerate(result.boxes):
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    print(f"  Box {i+1}: class={class_name}, confidence={confidence:.3f}")
                    
                    # Track all detections for debugging
                    all_detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class": class_name,
                        "class_id": class_id
                    })
                    
                    # Check if this is a pineapple detection (case-insensitive)
                    if any(pineapple_term in class_name.lower() for pineapple_term in ['pineapple', 'pineapples', 'pineapple_group']):
                        print(f"  ✅ Pineapple detected: {class_name} (confidence: {confidence:.3f})")
                        is_pineapple = True
                        max_confidence = max(max_confidence, confidence)
                        
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "class": class_name,
                            "class_id": class_id
                        })
                    else:
                        print(f"  ❌ Not a pineapple: {class_name}")
            else:
                print(f"📦 No boxes found in result")
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # CONSENSUS DECISION: Apply multiple criteria to reduce false positives
        consensus_decision = False
        consensus_reasons = []
        
        if max_confidence >= 0.8:  # Very high confidence (80%+)
            consensus_decision = True
            consensus_reasons.append(f"Very high confidence: {max_confidence:.3f}")
        elif len(detections) >= 2 and max_confidence >= 0.65:  # Multiple detections with good confidence
            consensus_decision = True
            consensus_reasons.append(f"Multiple detections ({len(detections)}) with good confidence: {max_confidence:.3f}")
        elif max_confidence >= DETECTION_THRESHOLD:  # Standard threshold
            # Additional check: require at least some "Pineapple" class detections (not just groups)
            pineapple_detections = [d for d in detections if 'group' not in d['class'].lower()]
            if len(pineapple_detections) > 0:
                consensus_decision = True
                consensus_reasons.append(f"Standard threshold met with individual pineapple detection")
            else:
                consensus_reasons.append(f"Only group detections, no individual pineapples")
        else:
            consensus_reasons.append(f"Confidence too low: {max_confidence:.3f} < {DETECTION_THRESHOLD}")
        
        print(f"🎯 CONSENSUS DECISION: {consensus_decision}")
        print(f"🎯 REASONS: {', '.join(consensus_reasons)}")
        
        final_result = consensus_decision and len(detections) > 0
        
        return {
            "is_pineapple": final_result,
            "confidence": max_confidence,
            "threshold": DETECTION_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "detections": detections,
            "total_detections": len(detections),
            "all_detections": all_detections,  # Include all detections for debugging
            "debug_info": {
                "pineapple_detected": is_pineapple,
                "max_confidence": max_confidence,
                "meets_threshold": max_confidence >= DETECTION_THRESHOLD,
                "consensus_decision": consensus_decision,
                "consensus_reasons": consensus_reasons,
                "final_result": final_result
            }
        }

# ---------------- Model runner ----------------
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

# Initialize both models
try:
    # Use the trained PyTorch YOLOv8 detector (better accuracy)
    detector = YOLOv8PineappleDetector(DETECTOR_PATH)
    print(f"✅ Trained YOLOv8 pineapple detector loaded from: {DETECTOR_PATH}")
    print(f"🍍 Model classes: Pineapple, Pineapple_Group")
    
    classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)
    print(f"✅ Sweetness classifier loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Warning: Could not load models: {e}")
    print("Models will be loaded on first prediction request")
    detector = None
    classifier = None

# ---------------- FastAPI ----------------
app = FastAPI(title="Pineapple Sweetness API", version="1.0.0")

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
        "detector_path": DETECTOR_PATH,
        "classifier_path": MODEL_PATH,
        "class_names": CLASS_NAMES,
        "detection_threshold": DETECTION_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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
        # Lazy load classifier (detection now happens on mobile)
        global classifier
        if classifier is None:
            classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)

        print("🍍 Backend: Sweetness analysis only (detection handled by mobile)")
        
        # Directly classify sweetness (assuming mobile already detected pineapple)
        sweetness_result = classifier.predict(data)
        
        result = {
            "is_pineapple": True,  # Mobile app only sends if pineapple detected
            "detection_confidence": None,  # Not used anymore
            "detection_threshold": None,   # Not used anymore
            "confidence_threshold": None,  # Not used anymore
            "detections": [],              # Not used anymore
            "total_detections": 0,         # Not used anymore
            "all_detections": [],          # Not used anymore
            "debug_info": {"mobile_detection": True, "backend_detection": False},
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
        is_pineapple=True,  # Mobile detection confirmed it's a pineapple
        detection_confidence=None,  # Mobile detection handles this
        bounding_boxes=None,  # Mobile detection handles this
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
        "is_pineapple": True,
        "detection_confidence": None,
        "detection_threshold": None,
        "confidence_threshold": None,
        "detections": [],
        "total_detections": 0,
        "all_detections": [],
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