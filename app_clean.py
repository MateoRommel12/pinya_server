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
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

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

# Initialize classifier
try:
    classifier = KerasClassifier(MODEL_PATH, CLASS_NAMES)
    print(f"✅ Sweetness classifier loaded from: {MODEL_PATH}")
    print(f"🍯 Classes: {CLASS_NAMES}")
    print(f"🚀 Backend ready for sweetness analysis (detection handled by mobile)")
except Exception as e:
    print(f"⚠️  Warning: Could not load classifier: {e}")
    print("Classifier will be loaded on first prediction request")
    classifier = None

# ---------------- FastAPI ----------------
app = FastAPI(title="Pineapple Sweetness API - Mobile Detection Edition", version="2.0.0")

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
        "mode": "sweetness_analysis_only",
        "detection_method": "mobile_app",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
