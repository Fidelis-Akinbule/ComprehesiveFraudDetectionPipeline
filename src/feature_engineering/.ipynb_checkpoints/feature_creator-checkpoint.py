# src/feature_engineering/feature_creator.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from ..utils.checkpoint_manager import CheckpointManager

class FeatureCreator:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def engineer_features(self, train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Create new features for fraud detection"""
        
        # Parse datetime features
        if 'trans_date_trans_time' in train_clean.columns:
            self._create_datetime_features(train_clean, test_clean)
        
        # Amount-based features
        if 'amt' in train_clean.columns:
            self._create_amount_features(train_clean, test_clean)
        
        # Geographic features
        if 'lat' in train_clean.columns and 'long' in train_clean.columns:
            self._create_geographic_features(train_clean, test_clean)
        
        # Merchant category encoding
        if 'category' in train_clean.columns:
            self._create_category_features(train_clean, test_clean)
        
        feature_summary = {
            'datetime_features_created': 'trans_date_trans_time' in train_clean.columns,
            'amount_features_created': 'amt' in train_clean.columns,
            'geographic_features_created': 'lat' in train_clean.columns and 'long' in train_clean.columns,
            'category_encoding_created': 'category' in train_clean.columns,
            'final_feature_count': train_clean.shape[1]
        }
        
        # Save checkpoint
        checkpoint_data = {
            'train_featured': train_clean,
            'test_featured': test_clean,
            'feature_summary': feature_summary
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'features_engineered.pkl')
        
        return train_clean, test_clean, feature_summary
    
    def _create_datetime_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create datetime-based features"""
        for df in [train_df, test_df]:
            df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_datetime'].dt.hour
            df['day_of_week'] = df['trans_datetime'].dt.dayofweek
            df['month'] = df['trans_datetime'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    def _create_amount_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create amount-based features"""
        p75 = np.percentile(train_df['amt'], 75)
        p95 = np.percentile(train_df['amt'], 95)
    
        for df in [train_df, test_df]:
            df['amt_log'] = np.log1p(df['amt'])
            df['amt_high'] = (df['amt'] > p75).astype(int)
            df['amt_very_high'] = (df['amt'] > p95).astype(int)

    
    def _create_geographic_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create geographic features"""
        for df in [train_df, test_df]:
            df['distance_from_center'] = np.sqrt(df['lat']**2 + df['long']**2)
    
    def _create_category_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create category-based features"""
        if 'is_fraud' in train_df.columns:
            category_fraud_rates = train_df.groupby('category')['is_fraud'].mean()
            overall_fraud_rate = train_df['is_fraud'].mean()
            
            for df in [train_df, test_df]:
                df['category_fraud_rate'] = df['category'].map(category_fraud_rates)
                df['category_fraud_rate'].fillna(overall_fraud_rate, inplace=True)