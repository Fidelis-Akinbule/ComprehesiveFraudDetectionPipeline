# src/modeling/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from typing import Tuple, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.config import config

class ModelTrainer:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, Dict, str, Any]:
        """Train multiple models for comparison"""
        
        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, stratify=y_train
        )
        
        trained_models = {}
        model_scores = {}
        
        # Train each model
        for name, model in config.MODELS.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_split, y_train_split)
            
            # Validate
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            trained_models[name] = model
            model_scores[name] = {
                'auc_score': auc_score,
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            
            print(f"{name} AUC Score: {auc_score:.4f}")
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['auc_score'])
        best_model = trained_models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        
        # Save checkpoint
        checkpoint_data = {
            'trained_models': trained_models,
            'model_scores': model_scores,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'validation_data': {
                'X_val': X_val,
                'y_val': y_val
            }
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'models_trained.pkl')
        
        return trained_models, model_scores, best_model_name, best_model