# src/modeling/model_evaluator.py

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from typing import Dict, Any
from ..utils.checkpoint_manager import CheckpointManager

class ModelEvaluator:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def evaluate_models(self, trained_models: Dict, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        evaluation_results = {}
        
        for name, model in trained_models.items():
            print(f"\n=== {name.upper()} EVALUATION ===")
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            cm = confusion_matrix(y_val, y_pred)
            cr = classification_report(y_val, y_pred, output_dict=True)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            evaluation_results[name] = {
                'confusion_matrix': cm,
                'classification_report': cr,
                'auc_score': auc,
                'precision': cr['1']['precision'],
                'recall': cr['1']['recall'],
                'f1_score': cr['1']['f1-score']
            }

            # Business KPI Calculation
            avg_amt = 100  # TODO: Replace with real average amount or load from EDA
            cost_fp = 1    # Cost of annoying a legit customer or blocking a good transaction
            total_tp = cr['1']['recall'] * sum(y_val)
            total_fp = cr['0']['recall'] * sum(y_val == 0)
            expected_savings = (total_tp * avg_amt) - (total_fp * cost_fp)

            evaluation_results[name]['business_kpis'] = {
                'expected_savings': expected_savings,
                'false_positive_cost': total_fp * cost_fp,
                'true_positive_gain': total_tp * avg_amt,
            }

            print(f"AUC Score: {auc:.4f}")
            print(f"Precision: {cr['1']['precision']:.4f}")
            print(f"Recall: {cr['1']['recall']:.4f}")
            print(f"F1-Score: {cr['1']['f1-score']:.4f}")
            print(f"Expected Business Savings: {expected_savings:.2f}")

            
        
        # Save checkpoint
        checkpoint_data = {
            'evaluation_results': evaluation_results,
            'best_metrics': max(evaluation_results.items(), key=lambda x: x[1]['auc_score'])
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'evaluation_complete.pkl')
        
        return evaluation_results