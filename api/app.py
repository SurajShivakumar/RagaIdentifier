"""
FastAPI Backend for Raga Identifier
Provides REST API endpoints for raga identification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import tempfile
import librosa
import numpy as np
from typing import Dict, List
import pickle

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processing import PitchDetector, AudioPreprocessor
from src.feature_extraction import NoteExtractor, RAGA_TEMPLATES
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
from src.model.crnn_model import RagaCRNN
from src.utils.helpers import ensure_dir

# Import rule-based identifier
rule_based_path = os.path.join(os.path.dirname(__file__), '..', 'rule-based')
if rule_based_path not in sys.path:
    sys.path.append(rule_based_path)

try:
    from identifier import RuleBasedRagaIdentifier
except ImportError:
    RuleBasedRagaIdentifier = None
    print("⚠️ Could not import rule-based identifier")

# Initialize FastAPI app
app = FastAPI(
    title="Raga Identifier API",
    description="API for identifying Carnatic ragas from audio",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
crnn_model = None
label_encoder = None
pitch_detector = None
note_extractor = None
preprocessor = None
feature_extractor = None
rule_based_identifier = None

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'raga_model.h5')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup with new CRNN architecture"""
    global crnn_model, label_encoder, pitch_detector, note_extractor
    global preprocessor, feature_extractor, rule_based_identifier
    
    print("🚀 Initializing Raga Identifier API with advanced ML pipeline...")
    
    # Initialize preprocessing and feature extraction
    preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
    feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)
    pitch_detector = PitchDetector(sample_rate=22050, method='pyin')
    note_extractor = NoteExtractor()
    
    # Initialize rule-based identifier
    try:
        rule_based_identifier = RuleBasedRagaIdentifier(sr=22050)
        print(f"✅ Rule-based identifier loaded: {len(rule_based_identifier.get_supported_ragas())} ragas")
        print(f"   Ragas: {', '.join(rule_based_identifier.get_supported_ragas())}")
    except Exception as e:
        print(f"⚠️ Could not load rule-based identifier: {e}")
    
    # Try to load CRNN model
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            # Load TensorFlow/Keras model
            import tensorflow as tf
            crnn_model = tf.keras.models.load_model(MODEL_PATH)
            
            # Load label encoder
            with open(ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            
            print(f"✅ CRNN model loaded: {len(label_encoder.classes_)} ragas")
            print(f"   Ragas: {', '.join(label_encoder.classes_)}")
        else:
            print("⚠️ CRNN model not found at:")
            print(f"   Model: {MODEL_PATH}")
            print(f"   Encoder: {ENCODER_PATH}")
            print("   Train the model first using: python src/model/train_crnn.py")
    except Exception as e:
        print(f"⚠️ Could not load CRNN model: {e}")
    
    print("✅ API initialized successfully")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    index_path = os.path.join(static_dir, 'index.html')
    
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback if UI not found
    return """
    <html>
        <body>
            <h1>Raga Identifier API</h1>
            <p>API is running. UI not found at /static/index.html</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Raga Identifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Upload audio for ML-based prediction",
            "/ragas": "Get list of known ragas",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "crnn_model_loaded": crnn_model is not None,
        "pitch_detector": pitch_detector is not None,
        "note_extractor": note_extractor is not None
    }


@app.get("/ragas")
async def get_ragas():
    """Get list of known ragas"""
    ml_ragas = []
    rule_ragas = []
    
    if label_encoder is not None:
        ml_ragas = list(label_encoder.classes_)
    
    if rule_based_identifier is not None:
        rule_ragas = rule_based_identifier.get_supported_ragas()
    
    all_ragas = sorted(set(ml_ragas + rule_ragas))
    
    return {
        "ragas": all_ragas,
        "count": len(all_ragas),
        "ml_ragas": sorted(ml_ragas),
        "rule_based_ragas": sorted(rule_ragas)
    }


@app.post("/predict")
async def predict_raga(file: UploadFile = File(...)):
    """
    Predict raga using advanced CRNN model with HPS + bandpass preprocessing
    
    Pipeline:
    1. Audio → HPS (remove mridangam) + Bandpass (80-1800 Hz)
    2. Extract Mel-Spectrogram (CNN input)
    3. CRNN Model → Raga Probabilities
    
    Args:
        file: Audio file (mp3, wav, m4a, flac)
        
    Returns:
        Prediction results with confidence scores
    """
    if crnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please train the model first using: python src/model/train_crnn.py"
        )
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load audio
        audio, sr = preprocessor.load_audio(tmp_path)
        
        # DISABLED preprocessing to match training pipeline
        # Training uses raw audio without HPS/bandpass filtering
        # audio_clean = preprocessor.preprocess_for_raga_detection(
        #     audio, apply_hpss=True, apply_bandpass=True
        # )
        
        # Extract mel spectrogram (CNN input) from raw audio
        mel_spec = feature_extractor.extract_mel_spectrogram(audio)
        
        # Prepare for CNN (resize and normalize)
        mel_spec_normalized = feature_extractor.prepare_for_cnn(mel_spec, target_shape=(128, 128))
        
        # Add batch and channel dimensions
        mel_spec_input = np.expand_dims(mel_spec_normalized, axis=0)  # Batch
        mel_spec_input = np.expand_dims(mel_spec_input, axis=-1)  # Channel
        
        # Predict
        probabilities = crnn_model.predict(mel_spec_input, verbose=0)[0]
        
        # Get top-k predictions
        top_k = 5
        top_k_idx = np.argsort(probabilities)[::-1][:top_k]
        
        top_predictions = [
            {
                "raga": label_encoder.classes_[idx],
                "score": float(probabilities[idx])
            }
            for idx in top_k_idx
        ]
        
        # Convert spectrogram to base64 for visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_normalized,
            sr=22050,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        ax.set_title(f'Mel Spectrogram - {file.filename}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        spectrogram_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "method": "crnn",
            "pipeline": "HPS + Bandpass Filter + Mel-Spectrogram + CNN",
            "filename": file.filename,
            "predicted_raga": label_encoder.classes_[top_k_idx[0]],
            "confidence": float(probabilities[top_k_idx[0]]),
            "top_predictions": top_predictions,
            "spectrogram": f"data:image/png;base64,{spectrogram_base64}",
            "audio_info": {
                "duration": float(len(audio) / 22050),
                "sample_rate": 22050
            }
        }
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rule-based")
async def predict_raga_rule_based(file: UploadFile = File(...)):
    """
    Predict raga using rule-based system (arohanam/avarohanam pattern matching)
    
    Pipeline:
    1. Audio → Note Detection (pYIN)
    2. Extract Note Sequence
    3. Pattern Matching → Raga Identification
    
    Args:
        file: Audio file (mp3, wav, m4a, flac)
        
    Returns:
        Prediction results with pattern matching scores
    """
    if rule_based_identifier is None:
        raise HTTPException(
            status_code=503,
            detail="Rule-based identifier not available."
        )
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Identify raga using rule-based system
        predicted_raga, confidence_scores = rule_based_identifier.identify_from_audio(tmp_path, verbose=False)
        
        # Get top predictions sorted by confidence
        top_predictions = [
            {
                "raga": raga,
                "score": float(score)
            }
            for raga, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "method": "rule-based",
            "pipeline": "Note Detection (pYIN) + Arohanam/Avarohanam Pattern Matching",
            "filename": file.filename,
            "predicted_raga": predicted_raga,
            "confidence": confidence_scores.get(predicted_raga, 0.0),
            "top_predictions": top_predictions,
            "audio_info": {
                "supported_ragas": rule_based_identifier.get_supported_ragas()
            }
        }
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    
    print("Starting Raga Identifier API...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
