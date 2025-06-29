# src/data_processing/data_cleaner.py`

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager

class DataCleaner:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def clean_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Clean and preprocess the data"""
        # Make copies to avoid modifying originals
        train_clean = df_train.copy()
        test_clean = df_test.copy()
        
        # Handle missing values
        numerical_cols = train_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = train_clean.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numerical_cols:
            if train_clean[col].isnull().any():
                median_val = train_clean[col].median()
                train_clean[col].fillna(median_val, inplace=True)
                test_clean[col].fillna(median_val, inplace=True)
        
        for col in categorical_cols:
            if train_clean[col].isnull().any():
                mode_val = train_clean[col].mode()[0]
                train_clean[col].fillna(mode_val, inplace=True)
                test_clean[col].fillna(mode_val, inplace=True)
        
        # Remove duplicates
        initial_train_shape = train_clean.shape[0]
        train_clean = train_clean.drop_duplicates()
        duplicates_removed = initial_train_shape - train_clean.shape[0]
        
        cleaning_summary = {
            'duplicates_removed': duplicates_removed,
            'missing_values_handled': True,
            'final_train_shape': train_clean.shape,
            'final_test_shape': test_clean.shape
        }
        
        # Save checkpoint
        checkpoint_data = {
            'train_clean': train_clean,
            'test_clean': test_clean,
            'cleaning_summary': cleaning_summary
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'data_cleaned.pkl')
        
        return train_clean, test_clean, cleaning_summary
