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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        

        df = df.sort_values(['userId', 'ts'])
        

        user_last_activity = df.groupby('userId')['ts'].max().reset_index()
        user_last_activity.columns = ['userId', 'last_activity']
        

        max_date = df['ts'].max()
        churn_cutoff = max_date - timedelta(days=churn_window)
        

        user_last_activity['is_churned'] = (
            user_last_activity['last_activity'] < churn_cutoff
        ).astype(int)
        

        downgrades = df[df['page'] == 'Submit Downgrade']['userId'].unique()
        

        cancellation_pages = ['Cancellation Confirmation']
        cancellations = df[df['page'].isin(cancellation_pages)]['userId'].unique()

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
        

        features = []
        
        for user_id in df['userId'].unique():
            user_data = df[df['userId'] == user_id]
            

            user_features = {
                'userId': user_id,
                'gender': user_data['gender'].iloc[0],
                'level': user_data['level'].iloc[-1],  # Most recent subscription level
                'account_age_days': user_data['account_age_days'].iloc[-1],
            }
            

            user_features.update({
                'total_sessions': user_data['sessionId'].nunique(),
                'total_songs': len(user_data[user_data['page'] == 'NextSong']),
                'total_events': len(user_data),
                'avg_session_length': user_data.groupby('sessionId').size().mean(),
                'unique_artists': user_data['artist'].nunique(),
                'unique_songs': user_data['song'].nunique(),
                'avg_song_length': user_data[user_data['page'] == 'NextSong']['length'].mean(),
            })
            

            thumbs_up = len(user_data[user_data['page'] == 'Thumbs Up'])
            thumbs_down = len(user_data[user_data['page'] == 'Thumbs Down'])
            user_features.update({
                'thumbs_up': thumbs_up,
                'thumbs_down': thumbs_down,
                'thumbs_ratio': thumbs_up / (thumbs_up + thumbs_down + 1),
                'add_to_playlist': len(user_data[user_data['page'] == 'Add to Playlist']),
                'add_friend': len(user_data[user_data['page'] == 'Add Friend']),
            })
            

            user_features.update({
                'errors': len(user_data[user_data['status'] != 200]),
                'logout_events': len(user_data[user_data['page'] == 'Logout']),
                'help_events': len(user_data[user_data['page'] == 'Help']),
            })
            

            user_features.update({
                'avg_hour_of_day': user_data['hour'].mean(),
                'weekend_activity_ratio': user_data['is_weekend'].mean(),
                'days_active': user_data['ts'].dt.date.nunique(),
            })
            

            recent_cutoff = user_data['ts'].max() - timedelta(days=7)
            recent_data = user_data[user_data['ts'] > recent_cutoff]
            user_features.update({
                'recent_sessions': recent_data['sessionId'].nunique(),
                'recent_songs': len(recent_data[recent_data['page'] == 'NextSong']),
                'recent_activity_ratio': len(recent_data) / len(user_data) if len(user_data) > 0 else 0,
            })
            
            features.append(user_features)
        
        features_df = pd.DataFrame(features)

        features_df = features_df.merge(churn_labels[['userId', 'is_churned']], on='userId')
        

        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        logger.info(f"Features engineered. Shape: {features_df.shape}")
        return features_df
    
    def prepare_model_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for modeling"""

        categorical_columns = ['gender', 'level']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        

        feature_columns = [col for col in features_df.columns if col not in ['userId', 'is_churned']]
        X = features_df[feature_columns].values
        y = features_df['is_churned'].values
        

        X = self.scaler.fit_transform(X)
        
        logger.info(f"Model data prepared. X shape: {X.shape}, y distribution: {np.bincount(y)}")
        return X, y, feature_columns
