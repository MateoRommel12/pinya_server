# FastAPI Backend - Pineapple Detection + Sweetness Analysis
import os
import json
import uuid
import datetime
from io import BytesIO
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

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

# ---------------- Database Setup ----------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./pineapple_predictions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_path = Column(String)
    is_pineapple = Column(Boolean)
    detection_confidence = Column(Float)
    bounding_boxes = Column(Text)
    prediction = Column(String)
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- FastAPI App ----------------
app = FastAPI(
    title="Pineapple Detection & Sweetness Analysis API",
    description="API for detecting pineapples and analyzing their sweetness level",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- API Endpoints ----------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🍍 Pineapple Detection & Sweetness Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    detector_files = []
    if os.path.exists(DETECTOR_PATH):
        detector_files = [f for f in os.listdir(DETECTOR_PATH) if f.endswith('.pt')]
    
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "models": {
            "classifier_path": MODEL_PATH,
            "classifier_exists": os.path.exists(MODEL_PATH),
            "detector_path": DETECTOR_PATH,
            "detector_exists": os.path.exists(DETECTOR_PATH),
            "detector_files": detector_files
        },
        "processing_mode": "Full (Detection + Sweetness Analysis)"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Predict pineapple detection and sweetness from uploaded image."""
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
        # Get models
        detector_model = get_detector()
        classifier_model = get_classifier()
        
        if detector_model is None or classifier_model is None:
            raise HTTPException(status_code=500, detail="Failed to load ML models")
        
        print("🍍 Backend: Full processing (detection + sweetness analysis)")
        
        # Step 1: Detect if image contains pineapple using YOLOv8
        detection_result = detector_model.detect(data, detection_threshold=0.5, confidence_threshold=0.3)
        
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
            sweetness_result = classifier_model.predict(data)
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
    
    # Clean up memory after prediction to prevent OOM
    unload_models()
    
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

@app.get("/predictions")
async def get_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get prediction history."""
    predictions = db.query(Prediction).offset(skip).limit(limit).all()
    return {
        "predictions": [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat() + "Z",
                "image_path": p.image_path,
                "is_pineapple": p.is_pineapple,
                "detection_confidence": p.detection_confidence,
                "prediction": p.prediction,
                "confidence": p.confidence,
            }
            for p in predictions
        ],
        "total": db.query(Prediction).count()
    }

# Launch the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
