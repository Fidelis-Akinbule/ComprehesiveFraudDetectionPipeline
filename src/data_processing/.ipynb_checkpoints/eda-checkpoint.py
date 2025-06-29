# src/data_processing/eda.py`

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from ..utils.checkpoint_manager import CheckpointManager

class EDAAnalyzer:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
   
    def perform_eda(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive EDA with business-focused visualizations"""
        eda_results = {}

        # Missing values analysis
        eda_results['missing_data'] = df_train.isnull().sum().loc[lambda x: x > 0]

        # Data types analysis
        eda_results['dtypes'] = df_train.dtypes.value_counts()

        # Categorical and numerical columns
        categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numerical_cols:
            numerical_cols.remove('is_fraud')
        eda_results['categorical_cols'] = categorical_cols
        eda_results['numerical_cols'] = numerical_cols

        # Fraud rate by segment
        if 'is_fraud' in df_train.columns:
            eda_results['fraud_rate_overall'] = df_train['is_fraud'].mean()

        # Segment by risk quartile
            df_train['risk_segment'] = pd.qcut(df_train['amt'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
            fraud_by_segment = df_train.groupby('risk_segment')['is_fraud'].agg(['count', 'mean'])
            eda_results['fraud_by_risk_segment'] = fraud_by_segment

            fraud_by_category = {}
            for col in categorical_cols[:5]:
                fraud_by_category[col] = df_train.groupby(col)['is_fraud'].agg(['count', 'mean'])
            eda_results['fraud_by_category'] = fraud_by_category

            # Time-based insights
            if 'trans_date_trans_time' in df_train.columns:
                df_train['hour'] = pd.to_datetime(df_train['trans_date_trans_time']).dt.hour
                fraud_by_hour = df_train.groupby('hour')['is_fraud'].mean()
                eda_results['fraud_by_hour'] = fraud_by_hour

        # Correlation
        eda_results['correlation_matrix'] = df_train[numerical_cols + ['is_fraud']].corr()

        self.checkpoint_manager.save_checkpoint(eda_results, 'eda_complete.pkl')
        return eda_results

    def generate_eda_report(self, eda_results: Dict[str, Any], output_path: str = "reports/eda_report.html"):
        """Generate comprehensive EDA report"""
        # Implementation for generating HTML report
        pass