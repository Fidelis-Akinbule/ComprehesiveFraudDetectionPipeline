# src/utils/config.py

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Config:
    # Paths
    DATA_DIR: str = "data/"
    RAW_DATA_DIR: str = "data/raw/"
    PROCESSED_DATA_DIR: str = "data/processed/"
    CHECKPOINT_DIR: str = "data/checkpoints/"
    MODELS_DIR: str = "models/"
    REPORTS_DIR: str = "reports/"
    
    # Data files
    TRAIN_FILE: str = "fraudTrain.csv"
    TEST_FILE: str = "fraudTest.csv"
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    # Feature engineering
    HIGH_CARDINALITY_THRESHOLD: int = 10
    
    # Model names
    MODELS: Dict[str, Any] = None
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                        self.CHECKPOINT_DIR, self.MODELS_DIR, self.REPORTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Define models if not provided
        if self.MODELS is None:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            
            self.MODELS = {
                'logistic_regression': LogisticRegression(random_state=self.RANDOM_STATE, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.RANDOM_STATE, n_jobs=-1),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.RANDOM_STATE)
            }

# Global config instance
config = Config()