# main.py

"""
Main pipeline script for fraud detection project
"""

import argparse
import sys
from src.data_processing.data_loader import DataLoader
from src.data_processing.eda import EDAAnalyzer
from src.data_processing.data_cleaner import DataCleaner
from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.feature_selector import FeatureSelector
from src.modeling.model_trainer import ModelTrainer
from src.modeling.model_evaluator import ModelEvaluator
from src.modeling.predictor import Predictor
from src.utils.checkpoint_manager import CheckpointManager

def run_complete_pipeline():
    """Run the complete fraud detection pipeline"""
    print("Starting Complete Fraud Detection Pipeline...")
    
    # Initialize components
    data_loader = DataLoader()
    eda_analyzer = EDAAnalyzer()
    data_cleaner = DataCleaner()
    feature_creator = FeatureCreator()
    feature_selector = FeatureSelector()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    predictor = Predictor()
    
    try:
        # Phase 1: Data Loading and EDA
        print("\n=== Data Loading and EDA ===")
        df_train, df_test = data_loader.load_and_inspect_data()
        eda_results = eda_analyzer.perform_eda(df_train)
        
        # Phase 2: Data Cleaning
        print("\n=== Data Cleaning ===")
        train_clean, test_clean, cleaning_summary = data_cleaner.clean_data(df_train, df_test)
        
        # Phase 3: Feature Engineering
        print("\n=== Feature Engineering ===")
        train_featured, test_featured, feature_summary = feature_creator.engineer_features(train_clean, test_clean)
        
        # Phase 4: Feature Preparation
        print("\n=== Feature Preparation ===")
        X_train, y_train, X_test, encoders, scaler, prep_summary = feature_selector.prepare_features(train_featured, test_featured)
        
        # Phase 5: Model Training
        print("\n=== Model Training ===")
        trained_models, model_scores, best_model_name, best_model = model_trainer.train_models(X_train, y_train)
        
        # Phase 6: Model Evaluation
        print("\n=== Model Evaluation ===")
        validation_data = CheckpointManager().load_checkpoint('models_trained.pkl')['validation_data']
        evaluation_results = model_evaluator.evaluate_models(trained_models, validation_data['X_val'], validation_data['y_val'])
        
        # Phase 7: Final Predictions
        print("\n=== Final Predictions ===")
        final_model, predictions, feature_importance = predictor.generate_final_predictions(best_model, X_train, y_train, X_test)
        model_pipeline = predictor.save_production_model(final_model, encoders, scaler, X_train.columns.tolist())
        
        print("\n=== Pipeline Complete! ===")
        print(f"Best Model: {best_model_name}")
        print(f"Test Predictions Shape: {predictions.shape}")
        if feature_importance is not None:
            print("Top 5 Important Features:")
            print(feature_importance.head())
        
        return predictions, feature_importance
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)

def resume_from_checkpoint(checkpoint_name: str):
    """Resume pipeline from specific checkpoint"""
    checkpoint_manager = CheckpointManager()
    checkpoint_data = checkpoint_manager.resume_from_checkpoint(checkpoint_name)
    
    if checkpoint_data is None:
        print(f"Cannot resume from checkpoint: {checkpoint_name}")
        return
    
    print(f"Resuming from checkpoint: {checkpoint_name}")
    
    # Initialize components
    feature_creator = FeatureCreator()
    feature_selector = FeatureSelector()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    predictor = Predictor()
    
    if checkpoint_name == 'data_loaded.pkl':
        df_train = checkpoint_data['df_train']
        df_test = checkpoint_data['df_test']
        
        # Continue from data cleaning
        data_cleaner = DataCleaner()
        train_clean, test_clean, cleaning_summary = data_cleaner.clean_data(df_train, df_test)
        
        # Continue with feature engineering
        train_featured, test_featured, feature_summary = feature_creator.engineer_features(train_clean, test_clean)
        # ... continue pipeline
        
    elif checkpoint_name == 'data_cleaned.pkl':
        train_clean = checkpoint_data['train_clean']
        test_clean = checkpoint_data['test_clean']
        
        # Continue from feature engineering
        train_featured, test_featured, feature_summary = feature_creator.engineer_features(train_clean, test_clean)
        # ... continue pipeline
        
    elif checkpoint_name == 'features_engineered.pkl':
        train_featured = checkpoint_data['train_featured']
        test_featured = checkpoint_data['test_featured']
        
        # Continue from feature preparation
        X_train, y_train, X_test, encoders, scaler, prep_summary = feature_selector.prepare_features(train_featured, test_featured)
        # ... continue pipeline
        
    elif checkpoint_name == 'features_prepared.pkl':
        X_train = checkpoint_data['X_train']
        y_train = checkpoint_data['y_train']
        X_test = checkpoint_data['X_test']
        encoders = checkpoint_data['encoders']
        scaler = checkpoint_data['scaler']
        
        # Continue from model training
        trained_models, model_scores, best_model_name, best_model = model_trainer.train_models(X_train, y_train)
        # ... continue pipeline
        
    elif checkpoint_name == 'models_trained.pkl':
        trained_models = checkpoint_data['trained_models']
        best_model = checkpoint_data['best_model']
        validation_data = checkpoint_data['validation_data']
        
        # Continue from evaluation
        evaluation_results = model_evaluator.evaluate_models(trained_models, validation_data['X_val'], validation_data['y_val'])
        # ... continue pipeline
        
    else:
        print(f"Unknown checkpoint: {checkpoint_name}")

def list_checkpoints():
    """List all available checkpoints"""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print("Available checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint}")

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--action', choices=['run', 'resume', 'list'], 
                       default='run', help='Action to perform')
    parser.add_argument('--checkpoint', type=str, 
                       help='Checkpoint name to resume from')
    
    args = parser.parse_args()
    
    if args.action == 'run':
        run_complete_pipeline()
    elif args.action == 'resume':
        if args.checkpoint:
            resume_from_checkpoint(args.checkpoint)
        else:
            print("Please specify a checkpoint name with --checkpoint")
    elif args.action == 'list':
        list_checkpoints()

if __name__ == '__main__':
    main()