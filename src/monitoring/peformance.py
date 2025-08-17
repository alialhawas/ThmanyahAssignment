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


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Monitoring and drift detection
from scipy.stats import ks_2samp
# from evidently import Report
# from evidently.metrics import DataDriftMetric, DataQualityMetric

import pickle
import logging
import os
from pathlib import Path
import yaml
from dataclasses import dataclass

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

