# src/modeling/predictor.py

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Optional, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.config import config

class Predictor:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def generate_final_predictions(self, best_model: Any, X_train: pd.DataFrame, 
                                 y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[Any, np.ndarray, Optional[pd.DataFrame]]:
        """Train final model on full dataset and generate predictions"""
        
        # Train on full training set
        print("Training final model on full dataset...")
        best_model.fit(X_train, y_train)
        
        # Generate predictions
        test_predictions = best_model.predict_proba(X_test)[:, 1]
        test_predictions_binary = best_model.predict(X_test)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Adding business insight mapping
            feature_insights = feature_importance.copy()
            feature_insights['business_insight'] = feature_insights['feature'].map({
                'amt_very_high': 'Large transactions are high risk – suggest real-time verification',
                'hour': 'Late-night transactions have higher fraud rates',
                'category_fraud_rate': 'Some merchant categories are more fraud-prone',
            })
            feature_importance = feature_insights

        # Save checkpoint
        checkpoint_data = {
            'final_model': best_model,
            'test_predictions': test_predictions,
            'test_predictions_binary': test_predictions_binary,
            'feature_importance': feature_importance
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'final_predictions.pkl')
        
        return best_model, test_predictions, feature_importance
    
    def save_production_model(self, final_model: Any, encoders: Dict, 
                            scaler: Any, feature_cols: list) -> Dict[str, Any]:
        """Save complete model pipeline for production"""
        
        # Create model pipeline
        model_pipeline = {
            'model': final_model,
            'encoders': encoders,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'model_type': type(final_model).__name__,
            'timestamp': pd.Timestamp.now()
        }
        
        # Save using joblib for better performance
        model_path = config.MODELS_DIR + 'fraud_detection_pipeline.joblib'
        joblib.dump(model_pipeline, model_path)
        
        # Also save as pickle for compatibility
        self.checkpoint_manager.save_checkpoint(model_pipeline, 'model_saved.pkl')
        
        print("Production model saved successfully!")
        
        return model_pipeline