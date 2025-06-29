# src/data_processing/data_loader.py


import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.config import config


class DataLoader:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
   
    def load_and_inspect_data(self, train_path: str = None, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and perform initial data inspection"""
       
        # Use default paths if not provided
        if train_path is None:
            train_path = config.RAW_DATA_DIR + config.TRAIN_FILE
        if test_path is None:
            test_path = config.RAW_DATA_DIR + config.TEST_FILE
       
        # Load data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
       
        # Basic info
        print("=== TRAINING DATA INFO ===")
        print(f"Shape: {df_train.shape}")
        print(f"Columns: {list(df_train.columns)}")
        print(f"Memory usage: {df_train.memory_usage().sum() / 1024**2:.2f} MB")
       
        print("\n=== TEST DATA INFO ===")
        print(f"Shape: {df_test.shape}")
        print(f"Memory usage: {df_test.memory_usage().sum() / 1024**2:.2f} MB")
       
        # Check target distribution
        if 'is_fraud' in df_train.columns:
            print(f"\n=== TARGET DISTRIBUTION ===")
            print(df_train['is_fraud'].value_counts())
            print(f"Fraud rate: {df_train['is_fraud'].mean():.4f}")
       
        # Save checkpoint
        checkpoint_data = {
            'df_train': df_train,
            'df_test': df_test,
            'basic_stats': {
                'train_shape': df_train.shape,
                'test_shape': df_test.shape,
                'fraud_rate': df_train['is_fraud'].mean() if 'is_fraud' in df_train.columns else None
            }
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'data_loaded.pkl')
       
        return df_train, df_test