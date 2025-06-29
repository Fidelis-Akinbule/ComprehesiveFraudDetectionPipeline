# src/feature_engineering/feature_selector.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any, List
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.config import config

class FeatureSelector:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def prepare_features(self, train_featured: pd.DataFrame, test_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict, StandardScaler, Dict[str, Any]]:
        """Prepare features for modeling"""
        
        # Separate features and target
        target_col = 'is_fraud'
        
        # Identify feature columns
        exclude_cols = [target_col, 'trans_date_trans_time', 'trans_datetime',
                'cc_num', 'trans_num', 'first', 'last', 'dob', 'street', 'city', 'risk_segment']
        if 'cc_num' in train_featured.columns:
            exclude_cols.append('cc_num')
        if 'trans_num' in train_featured.columns:
            exclude_cols.append('trans_num')
        
        feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
        
        # Get features and target
        X_train = train_featured[feature_cols].copy()
        y_train = train_featured[target_col].copy() if target_col in train_featured.columns else None
        X_test = test_featured[feature_cols].copy()
        
        # Validate and clean target variable if present in test set
        if target_col in test_featured.columns:
            y_test = test_featured[target_col].copy()
            X_test, y_test = self._validate_target_consistency(X_test, y_test, y_train)
        
        # Encode categorical variables
        X_train, X_test, encoders = self._encode_categorical_features(X_train, X_test)
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        preparation_summary = {
            'feature_count': X_train.shape[1],
            'categorical_encoded': len(encoders),
            'numerical_scaled': len(numerical_cols),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape
        }
        
        # Save checkpoint
        checkpoint_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'encoders': encoders,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'preparation_summary': preparation_summary
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data, 'features_prepared.pkl')
        
        return X_train, y_train, X_test, encoders, scaler, preparation_summary
    
    def _validate_target_consistency(self, X_test: pd.DataFrame, y_test: pd.Series, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean target variable consistency between train and test sets"""
        
        if y_train is None:
            print("Warning: No training target available for validation")
            return X_test, y_test
        
        # Get unique values
        train_labels = set(y_train.unique())
        test_labels = set(y_test.unique())
        unseen_labels = test_labels - train_labels
        
        print(f"Training target labels: {sorted(train_labels)}")
        print(f"Test target labels: {sorted(test_labels)}")
        
        if unseen_labels:
            print(f"WARNING: Found unseen target labels in test set: {unseen_labels}")
            
            # Option 1: Remove samples with unseen labels
            mask = y_test.isin(train_labels)
            removed_count = (~mask).sum()
            
            if removed_count > 0:
                print(f"Removing {removed_count} samples with unseen target labels")
                X_test = X_test[mask].reset_index(drop=True)
                y_test = y_test[mask].reset_index(drop=True)
                
                # Show what was removed
                removed_labels = y_test[~mask].value_counts()
                print(f"Removed samples by label: {dict(removed_labels)}")
        
        # Additional validation: Check if target should be binary for fraud detection
        if len(train_labels) <= 2 and any(isinstance(label, str) and label not in ['0', '1', 'True', 'False'] for label in train_labels | test_labels):
            print("WARNING: Target variable contains unexpected string values for fraud detection")
            print("Expected binary values (0/1, True/False), but found string labels")
            
            # Suggest data cleaning
            print("Consider checking your data preprocessing - fraud detection targets should be binary")
        
        return X_test, y_test
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Encode categorical variables"""
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        encoders = {}
        
        for col in categorical_cols:
            print(f"Encoding categorical column: {col}")
            unique_values = X_train[col].nunique()
            
            if unique_values > config.HIGH_CARDINALITY_THRESHOLD:
                # High cardinality - use Label Encoding
                le = LabelEncoder()
                
                # Handle unseen categories in test set
                train_categories = set(X_train[col].astype(str).unique())
                test_categories = set(X_test[col].astype(str).unique())
                unseen_categories = test_categories - train_categories
                
                if unseen_categories:
                    print(f"WARNING: Column '{col}' has unseen categories in test set: {unseen_categories}")
                    # Map unseen categories to 'UNKNOWN'
                    X_test[col] = X_test[col].astype(str)
                    X_test.loc[X_test[col].isin(unseen_categories), col] = 'UNKNOWN'
                    
                    # Add 'UNKNOWN' to training set if not present
                    if 'UNKNOWN' not in train_categories:
                        X_train = X_train.copy()
                        X_train.loc[X_train.index[0], col] = 'UNKNOWN'  # Add one instance
                
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                encoders[col] = le
                
            else:
                # Low cardinality - use One-Hot Encoding
                train_dummies = pd.get_dummies(X_train[col], prefix=col)
                test_dummies = pd.get_dummies(X_test[col], prefix=col)
                
                # Align columns
                all_columns = set(train_dummies.columns) | set(test_dummies.columns)
                
                for dummy_col in all_columns:
                    if dummy_col not in train_dummies.columns:
                        train_dummies[dummy_col] = 0
                    if dummy_col not in test_dummies.columns:
                        test_dummies[dummy_col] = 0
                
                # Ensure column order is consistent
                sorted_columns = sorted(all_columns)
                train_dummies = train_dummies[sorted_columns]
                test_dummies = test_dummies[sorted_columns]
                
                # Drop original column and add dummies
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
                
                X_train = pd.concat([X_train, train_dummies], axis=1)
                X_test = pd.concat([X_test, test_dummies], axis=1)
        
        return X_train, X_test, encoders
    
    def diagnose_data_issues(self, train_featured: pd.DataFrame, test_featured: pd.DataFrame):
        """Diagnose potential data issues"""
        target_col = 'is_fraud'
        
        print("=== DATA DIAGNOSIS ===")
        
        if target_col in train_featured.columns:
            print(f"Training target unique values: {train_featured[target_col].unique()}")
            print(f"Training target value counts:\n{train_featured[target_col].value_counts()}")
        
        if target_col in test_featured.columns:
            print(f"Test target unique values: {test_featured[target_col].unique()}")
            print(f"Test target value counts:\n{test_featured[target_col].value_counts()}")
        
        # Check for data type issues
        if target_col in train_featured.columns and target_col in test_featured.columns:
            train_dtype = train_featured[target_col].dtype
            test_dtype = test_featured[target_col].dtype
            print(f"Training target dtype: {train_dtype}")
            print(f"Test target dtype: {test_dtype}")
            
            if train_dtype != test_dtype:
                print("WARNING: Target variable has different data types in train vs test!")
        
        print("=" * 50)