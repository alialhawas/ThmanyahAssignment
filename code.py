# Customer Churn Prediction System
# Complete implementation with data processing, modeling, API, and monitoring

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# FastAPI for serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Monitoring and drift detection
from scipy.stats import ks_2samp
from evidently import Report
# from evidently.metrics import DataDriftMetric, DataQualityMetric

# Utilities
import pickle
import logging
import os
from pathlib import Path
import yaml
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== DATA PROCESSING MODULE ====================

class DataProcessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load JSON Lines data into DataFrame"""
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the raw data"""
        logger.info("Starting data cleaning...")
        
        # Convert timestamp to datetime
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df['registration'] = pd.to_datetime(df['registration'], unit='ms')
        
        # Remove records with missing critical fields
        critical_fields = ['userId', 'ts', 'page']
        df = df.dropna(subset=critical_fields)
        
        # Handle missing values
        df['artist'] = df['artist'].fillna('Unknown')
        df['song'] = df['song'].fillna('Unknown')
        df['length'] = df['length'].fillna(df['length'].median())
        
        # Create date features
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Account age in days
        df['account_age_days'] = (df['ts'] - df['registration']).dt.days
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def define_churn(self, df: pd.DataFrame, observation_window: int = 30, 
                    churn_window: int = 30) -> pd.DataFrame:
        """Define churn based on user activity patterns"""
        logger.info("Defining churn labels...")
        
        # Sort by user and timestamp
        df = df.sort_values(['userId', 'ts'])
        
        # Get last activity date for each user
        user_last_activity = df.groupby('userId')['ts'].max().reset_index()
        user_last_activity.columns = ['userId', 'last_activity']
        
        # Define churn cutoff date
        max_date = df['ts'].max()
        churn_cutoff = max_date - timedelta(days=churn_window)
        
        # Users who haven't been active in the churn window are considered churned
        user_last_activity['is_churned'] = (
            user_last_activity['last_activity'] < churn_cutoff
        ).astype(int)
        
        # Additional churn indicators
        # Users who downgraded from paid to free
        downgrades = df[df['page'] == 'Submit Downgrade']['userId'].unique()
        
        # Users who visited cancellation confirmation page
        cancellation_pages = ['Cancellation Confirmation']
        cancellations = df[df['page'].isin(cancellation_pages)]['userId'].unique()
        
        # Mark additional churn cases
        user_last_activity.loc[
            user_last_activity['userId'].isin(downgrades), 'is_churned'
        ] = 1
        user_last_activity.loc[
            user_last_activity['userId'].isin(cancellations), 'is_churned'
        ] = 1
        
        churn_rate = user_last_activity['is_churned'].mean()
        logger.info(f"Churn rate: {churn_rate:.3f}")
        
        return user_last_activity
    
    def engineer_features(self, df: pd.DataFrame, churn_labels: pd.DataFrame) -> pd.DataFrame:
        """Create features for churn prediction"""
        logger.info("Engineering features...")
        
        # Aggregate features by user
        features = []
        
        for user_id in df['userId'].unique():
            user_data = df[df['userId'] == user_id]
            
            # Basic user info
            user_features = {
                'userId': user_id,
                'gender': user_data['gender'].iloc[0],
                'level': user_data['level'].iloc[-1],  # Most recent subscription level
                'account_age_days': user_data['account_age_days'].iloc[-1],
            }
            
            # Activity features
            user_features.update({
                'total_sessions': user_data['sessionId'].nunique(),
                'total_songs': len(user_data[user_data['page'] == 'NextSong']),
                'total_events': len(user_data),
                'avg_session_length': user_data.groupby('sessionId').size().mean(),
                'unique_artists': user_data['artist'].nunique(),
                'unique_songs': user_data['song'].nunique(),
                'avg_song_length': user_data[user_data['page'] == 'NextSong']['length'].mean(),
            })
            
            # Engagement features
            thumbs_up = len(user_data[user_data['page'] == 'Thumbs Up'])
            thumbs_down = len(user_data[user_data['page'] == 'Thumbs Down'])
            user_features.update({
                'thumbs_up': thumbs_up,
                'thumbs_down': thumbs_down,
                'thumbs_ratio': thumbs_up / (thumbs_up + thumbs_down + 1),
                'add_to_playlist': len(user_data[user_data['page'] == 'Add to Playlist']),
                'add_friend': len(user_data[user_data['page'] == 'Add Friend']),
            })
            
            # Error and negative events
            user_features.update({
                'errors': len(user_data[user_data['status'] != 200]),
                'logout_events': len(user_data[user_data['page'] == 'Logout']),
                'help_events': len(user_data[user_data['page'] == 'Help']),
            })
            
            # Time-based patterns
            user_features.update({
                'avg_hour_of_day': user_data['hour'].mean(),
                'weekend_activity_ratio': user_data['is_weekend'].mean(),
                'days_active': user_data['ts'].dt.date.nunique(),
            })
            
            # Recent activity (last 7 days)
            recent_cutoff = user_data['ts'].max() - timedelta(days=7)
            recent_data = user_data[user_data['ts'] > recent_cutoff]
            user_features.update({
                'recent_sessions': recent_data['sessionId'].nunique(),
                'recent_songs': len(recent_data[recent_data['page'] == 'NextSong']),
                'recent_activity_ratio': len(recent_data) / len(user_data) if len(user_data) > 0 else 0,
            })
            
            features.append(user_features)
        
        features_df = pd.DataFrame(features)
        
        # Merge with churn labels
        features_df = features_df.merge(churn_labels[['userId', 'is_churned']], on='userId')
        
        # Handle missing values
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        logger.info(f"Features engineered. Shape: {features_df.shape}")
        return features_df
    
    def prepare_model_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for modeling"""
        # Encode categorical variables
        categorical_columns = ['gender', 'level']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns if col not in ['userId', 'is_churned']]
        X = features_df[feature_columns].values
        y = features_df['is_churned'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        logger.info(f"Model data prepared. X shape: {X.shape}, y distribution: {np.bincount(y)}")
        return X, y, feature_columns


# ==================== MODEL TRAINING MODULE ====================

class ChurnPredictor:
    """Handles model training, evaluation, and prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   feature_names: List[str]) -> Dict[str, Any]:
        """Train the churn prediction model"""
        logger.info("Starting model training...")
        
        # Handle class imbalance with class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Define models to try
        models = {
            'logistic_regression': LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                scale_pos_weight=class_weights[1]/class_weights[0],
                random_state=42
            )
        }
        
        # Evaluate models with cross-validation
        best_score = 0
        best_model_name = None
        model_scores = {}
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            avg_score = scores.mean()
            model_scores[name] = avg_score
            
            logger.info(f"{name}: ROC-AUC = {avg_score:.3f} (+/- {scores.std() * 2:.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name
        
        # Train the best model
        self.model = models[best_model_name]
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = list(zip(feature_names, self.model.feature_importances_))
            self.feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_score:.3f}")
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'all_scores': model_scores
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model evaluation complete. ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self.feature_importance
        }
    
    def predict_churn_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict churn probability for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(features)[:, 1]


# ==================== API MODULE ====================

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Global model instance
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
        # Load trained model and processor
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


# ==================== MONITORING MODULE ====================

class ModelMonitor:
    """Handles model performance monitoring and drift detection"""
    
    def __init__(self):
        self.reference_data = None
        self.performance_history = []
        
    def set_reference_data(self, X_reference: np.ndarray, y_reference: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data = {'X': X_reference, 'y': y_reference}
        
    def detect_data_drift(self, X_new: np.ndarray, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using Kolmogorov-Smirnov test"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        drift_results = {}
        n_features = X_new.shape[1]
        
        for i in range(n_features):
            statistic, p_value = ks_2samp(self.reference_data['X'][:, i], X_new[:, i])
            drift_results[f'feature_{i}'] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_drift': p_value < threshold
            }
        
        overall_drift = any(result['is_drift'] for result in drift_results.values())
        
        return {
            'overall_drift': overall_drift,
            'feature_drifts': drift_results,
            'drift_score': np.mean([result['statistic'] for result in drift_results.values()])
        }
    
    def track_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray, timestamp: datetime = None):
        """Track model performance over time"""
        if timestamp is None:
            timestamp = datetime.now()
        
        performance = {
            'timestamp': timestamp,
            'accuracy': np.mean(y_true == y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
        }
        
        self.performance_history.append(performance)
        return performance


# ==================== TRAINING PIPELINE ====================

class TrainingPipeline:
    """Complete training pipeline with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "churn_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def run_training(self, data_path: str, model_output_path: str = "models/"):
        """Run the complete training pipeline"""
        with mlflow.start_run():
            # Initialize components
            processor = DataProcessor()
            predictor = ChurnPredictor()
            monitor = ModelMonitor()
            
            # Load and process data
            raw_data = processor.load_data(data_path)
            clean_data = processor.clean_data(raw_data)
            churn_labels = processor.define_churn(clean_data)
            features_df = processor.engineer_features(clean_data, churn_labels)
            
            # Prepare model data
            X, y, feature_names = processor.prepare_model_data(features_df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log data info
            mlflow.log_params({
                'total_samples': len(X),
                'n_features': len(feature_names),
                'churn_rate': y.mean(),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            # Train model
            training_results = predictor.train_model(X_train, y_train, feature_names)
            mlflow.log_params(training_results)
            
            # Evaluate model
            evaluation_results = predictor.evaluate_model(X_test, y_test)
            mlflow.log_metrics(evaluation_results['metrics'])
            
            # Set up monitoring
            monitor.set_reference_data(X_train, y_train)
            
            # Save models
            os.makedirs(model_output_path, exist_ok=True)
            
            with open(f"{model_output_path}/churn_predictor.pkl", 'wb') as f:
                pickle.dump(predictor, f)
            
            with open(f"{model_output_path}/data_processor.pkl", 'wb') as f:
                pickle.dump(processor, f)
            
            with open(f"{model_output_path}/monitor.pkl", 'wb') as f:
                pickle.dump(monitor, f)
            
            # Log model with MLflow
            mlflow.sklearn.log_model(predictor.model, "churn_model")
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'feature_names': feature_names
            }


# ==================== RETRAINING MODULE ====================

class RetrainingScheduler:
    """Handles periodic model retraining"""
    
    def __init__(self, pipeline: TrainingPipeline, monitor: ModelMonitor):
        self.pipeline = pipeline
        self.monitor = monitor
        
    def should_retrain(self, drift_threshold: float = 0.1, 
                      performance_threshold: float = 0.05) -> bool:
        """Determine if model should be retrained based on drift and performance"""
        
        # Check for significant performance degradation
        if len(self.monitor.performance_history) >= 2:
            recent_performance = self.monitor.performance_history[-1]['roc_auc']
            baseline_performance = self.monitor.performance_history[0]['roc_auc']
            
            if baseline_performance - recent_performance > performance_threshold:
                logger.info("Performance degradation detected. Retraining recommended.")
                return True
        
        # Additional logic for data drift checking would go here
        # This would require new data to compare against reference
        
        return False
    
    def retrain_model(self, new_data_path: str):
        """Retrain the model with new data"""
        logger.info("Starting model retraining...")
        return self.pipeline.run_training(new_data_path)


# ==================== EXAMPLE USAGE ====================

def main():
    """Example usage of the complete system"""
    
    # Sample data for demonstration (you would use your actual JSON file)
    sample_data = [
        {"ts":1538352117000,"userId":"30","sessionId":29,"page":"NextSong","auth":"Logged In","method":"PUT","status":200,"level":"paid","itemInSession":50,"location":"Bakersfield, CA","userAgent":"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0","lastName":"Freeman","firstName":"Colin","registration":1538173362000,"gender":"M","artist":"Martha Tilston","song":"Rockpools","length":277.89016},
        {"ts":1538352180000,"userId":"9","sessionId":8,"page":"NextSong","auth":"Logged In","method":"PUT","status":200,"level":"free","itemInSession":79,"location":"Boston-Cambridge-Newton, MA-NH","userAgent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36","lastName":"Long","firstName":"Micah","registration":1538331630000,"gender":"M","artist":"Five Iron Frenzy","song":"Canada","length":236.09424}
    ]
    
    # Save sample data
    with open('sample_data.jsonl', 'w') as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run_training('sample_data.jsonl')
    
    print("Training Results:")
    print(f"Best Model: {results['training_results']['best_model']}")
    print(f"ROC-AUC: {results['evaluation_results']['metrics']['roc_auc']:.3f}")
    
    # Start API server (uncomment to run)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()