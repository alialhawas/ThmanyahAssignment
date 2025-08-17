
# Customer Churn Prediction System
# Complete implementation with data processing, modeling, API, and monitoring

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import logging


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


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


from scipy.stats import ks_2samp
# from evidently import Report
# from evidently.metrics import DataDriftMetric, DataQualityMetric

import pickle
import logging
import os
from pathlib import Path
import yaml
from dataclasses import dataclass


logger = logging.getLogger(__name__)


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
