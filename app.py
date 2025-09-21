# Capstone/server/app.py - Full Pineapple Detection + Sweetness Analysis
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
                    f"Failed to load model at {model_path}.\\n"
                    f"tf.keras error: {e1}\\n"
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
        # Updated for your trained 3-class model
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
            
            # Handle PyTorch security settings for model loading
            try:
                # First try with weights_only=False (for trusted models)
                print("🔧 Attempting to load with weights_only=False...")
                
                # Set PyTorch to allow unsafe loading for YOLOv8 models
                original_weights_only = getattr(torch, '_weights_only_unpickler', None)
                
                # Add safe globals for ultralytics classes
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
                
                # Alternative: Patch torch.load temporarily
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
            
            # Test the model with a simple inference
            print("🧪 Testing model with dummy inference...")
            
            # Create a dummy 640x640 RGB image
            dummy_img = Image.new('RGB', (640, 640), color='white')
            
            # Run a test prediction
            results = self.model(dummy_img, conf=0.1, verbose=False)
            print(f"✅ Model test successful - ready for detection!")
            
        except ImportError as e:
            print(f"❌ ultralytics package not found: {e}")
            raise RuntimeError("ultralytics package required. Install with: pip install ultralytics")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 model: {e}")
            raise RuntimeError(f"Failed to load .pt model from {self.pt_path}: {e}")
    
    def detect(self, img_bytes: bytes) -> Dict[str, Any]:
        # Load model on first use
        self._load_model()
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        
        # Set thresholds
        confidence_threshold = 0.25  # 25% confidence threshold
        detection_threshold = 0.50   # 50% detection threshold (balanced)
        
        print(f"🔍 Running YOLOv8 detection with conf={confidence_threshold}")
        print(f"🎯 Detection threshold: {detection_threshold}")
        
        detections = []
        max_confidence = 0.0
        is_pineapple = False
        all_detections = []
        
        try:
            # Run YOLOv8 inference using ultralytics
            print(f"🚀 Running YOLOv8 inference...")
            results = self.model(img, conf=confidence_threshold, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get box data
                        conf = float(box.conf.item())
                        cls = int(box.cls.item())
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Map class ID to class name
                        class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                        
                        detection = {
                            "bbox": xyxy,
                            "confidence": conf,
                            "class": class_name,
                            "class_id": cls
                        }
                        
                        all_detections.append(detection)
                        
                        # Only include high-confidence detections
                        if conf >= detection_threshold:
                            detections.append(detection)
                            max_confidence = max(max_confidence, conf)
                            is_pineapple = True
                            print(f"  ✅ Pineapple detected: {class_name} (conf: {conf:.3f})")
                        else:
                            print(f"  ❌ Low confidence: {class_name} (conf: {conf:.3f}, required: {detection_threshold})")
                else:
                    print("  ❌ No detections found")
            
            print(f"📊 Total detections: {len(all_detections)}")
            print(f"📊 High-confidence detections: {len(detections)}")
            print(f"📊 Best confidence: {max_confidence:.3f}")
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            # Continue with no detections on error
            
        # Return results
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

# Initialize both detector and classifier
try:
    # Load detector first
    detector = UltralyticsYOLODetector(DETECTOR_PATH)
    print(f"✅ Pineapple detector loaded from: {DETECTOR_PATH}")
    
    # Load classifier
    classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)
    print(f"✅ Sweetness classifier loaded from: {MODEL_PATH}")
    print(f"🍯 Classes: {CLASS_NAMES}")
    print(f"🚀 Backend ready for full processing (detection + sweetness analysis)")
except Exception as e:
    print(f"⚠️  Warning: Could not load models: {e}")
    print("Models will be loaded on first prediction request")
    detector = None
    classifier = None

# ---------------- FastAPI ----------------
app = FastAPI(title="Pineapple Sweetness API - YOLOv8 Backend Edition", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    # Debug: Check what files actually exist in the detector directory
    detector_files = []
    if os.path.exists(DETECTOR_PATH):
        detector_files = os.listdir(DETECTOR_PATH)
    
    return {
        "status": "ok",
        "mode": "full_processing",
        "detection_method": "backend_yolov8",
        "detector_path": DETECTOR_PATH,
        "detector_files": detector_files,  # Debug info
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
        # Lazy load both models if needed
        global detector, classifier
        if detector is None:
            print(f"🔄 Loading detector from: {DETECTOR_PATH}")
            detector = UltralyticsYOLODetector(DETECTOR_PATH)
        if classifier is None:
            print(f"🔄 Loading classifier from: {MODEL_PATH}")
            classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)

        print("🍍 Backend: Full processing (detection + sweetness analysis)")
        
        # Step 1: Detect if image contains pineapple using YOLOv8
        detection_result = detector.detect(data)
        
        result = {
            "is_pineapple": detection_result["is_pineapple"],
            "detection_confidence": detection_result["confidence"],
            "detection_threshold": detection_result["threshold"],
            "confidence_threshold": detection_result["confidence_threshold"],
            "detections": detection_result["detections"],
            "total_detections": detection_result["total_detections"],
            "all_detections": detection_result.get("all_detections", []),
            "debug_info": detection_result.get("debug_info", {})
        }
        
        # Step 2: Only classify sweetness if pineapple is detected
        if detection_result["is_pineapple"]:
            sweetness_result = classifier.predict(data)
            result.update({
                "prediction": sweetness_result["prediction"],
                "confidence": sweetness_result["confidence"],
                "probabilities": sweetness_result["probabilities"],
                "message": f"Pineapple detected with {detection_result['confidence']:.2%} confidence. Predicted sweetness: {sweetness_result['prediction']} ({sweetness_result['confidence']:.2%} confidence)"
            })
        else:
            result.update({
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "message": f"No pineapple detected (confidence: {detection_result['confidence']:.2%}). Sweetness analysis skipped."
            })
            
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
        "message": result.get("message"),
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
