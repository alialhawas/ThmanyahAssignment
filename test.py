import json

from src.models.training import TrainingPipeline

def main():
    """Example usage of the complete system"""


    file = './data/customer_churn_mini.json'

    pipeline = TrainingPipeline()
    results = pipeline.run_training(file)
    
    print("Training Results:")
    print(f"Best Model: {results['training_results']['best_model']}")
    print(f"ROC-AUC: {results['evaluation_results']['metrics']['roc_auc']:.3f}")
    


if __name__ == "__main__":
    main()