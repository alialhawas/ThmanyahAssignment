

import mlflow
from src.models.model import ChurnPredictor
from src.data.preprocessing import DataProcessor
from src.monitoring.peformance import ModelMonitor

import os

import pickle
import logging

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

class TrainingPipeline:
    """Complete training pipeline with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "churn_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def run_training(self, data_path: str, model_output_path: str = "models/"):
        """Run the complete training pipeline"""
        with mlflow.start_run():

            processor = DataProcessor()
            predictor = ChurnPredictor()
            monitor = ModelMonitor()
            

            raw_data = processor.load_data(data_path)
            clean_data = processor.clean_data(raw_data)
            churn_labels = processor.define_churn(clean_data)
            features_df = processor.engineer_features(clean_data, churn_labels)
            

            X, y, feature_names = processor.prepare_model_data(features_df)
            
 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            mlflow.log_params({
                'total_samples': len(X),
                'n_features': len(feature_names),
                'churn_rate': y.mean(),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
            training_results = predictor.train_model(X_train, y_train, feature_names)
            mlflow.log_params(training_results)

            evaluation_results = predictor.evaluate_model(X_test, y_test)
            mlflow.log_metrics(evaluation_results['metrics'])

            monitor.set_reference_data(X_train, y_train)    

            os.makedirs(model_output_path, exist_ok=True)
            
            with open(f"{model_output_path}/churn_predictor.pkl", 'wb') as f:
                pickle.dump(predictor, f)
            
            with open(f"{model_output_path}/data_processor.pkl", 'wb') as f:
                pickle.dump(processor, f)
            
            with open(f"{model_output_path}/monitor.pkl", 'wb') as f:
                pickle.dump(monitor, f)
            
            mlflow.sklearn.log_model(predictor.model, "churn_model")
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'feature_names': feature_names
            }

class RetrainingScheduler:
    """Handles periodic model retraining"""
    
    def __init__(self, pipeline: TrainingPipeline, monitor: ModelMonitor):
        self.pipeline = pipeline
        self.monitor = monitor
        
    def should_retrain(self, drift_threshold: float = 0.1, 
                      performance_threshold: float = 0.05) -> bool:
        """Determine if model should be retrained based on drift and performance"""
        

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
