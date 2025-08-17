import pickle
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import numpy as np

from typing import Optional, List, Dict

logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

predictor = None
data_processor = None

class UserFeatures(BaseModel):
    gender: str
    level: str
    account_age_days: float
    total_sessions: int
    total_songs: int
    total_events: int
    avg_session_length: float
    unique_artists: int
    unique_songs: int
    avg_song_length: float
    thumbs_up: int
    thumbs_down: int
    thumbs_ratio: float
    add_to_playlist: int
    add_friend: int
    errors: int
    logout_events: int
    help_events: int
    avg_hour_of_day: float
    weekend_activity_ratio: float
    days_active: int
    recent_sessions: int
    recent_songs: int
    recent_activity_ratio: float

@app.on_event("startup")
async def load_model():
    global predictor, data_processor
    try:

        predictor = pickle.load(open('models/churn_predictor.pkl', 'rb'))
        data_processor = pickle.load(open('models/data_processor.pkl', 'rb'))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.post("/predict")
async def predict_churn(features: UserFeatures):
    if predictor is None or data_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        features_dict = features.dict()
        features_df = pd.DataFrame([features_dict])
        
        # Process features
        X, _, _ = data_processor.prepare_model_data(features_df)
        
        # Make prediction
        churn_probability = predictor.predict_churn_probability(X)[0]
        
        return {
            "user_features": features_dict,
            "churn_probability": float(churn_probability),
            "risk_level": "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low"
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

