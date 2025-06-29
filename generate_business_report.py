import pandas as pd
from src.utils.checkpoint_manager import CheckpointManager

def generate_business_report():
    cm = CheckpointManager()

    # Load checkpoints
    eda = cm.load_checkpoint('eda_complete.pkl')
    evaluation = cm.load_checkpoint('evaluation_complete.pkl')
    feature_importance = cm.load_checkpoint('final_predictions.pkl').get('feature_importance')

    with open("reports/business_insight_report.txt", "w") as f:
        f.write("=== BUSINESS INSIGHT REPORT ===\n\n")

        # EDA Risk Segments
        f.write(">> Fraud Rate by Risk Segment (Amount Quartiles):\n")
        f.write(str(eda.get('fraud_by_risk_segment', 'N/A')) + "\n\n")

        # Time-based fraud insights
        f.write(">> Fraud Rate by Hour:\n")
        f.write(str(eda.get('fraud_by_hour', 'N/A')) + "\n\n")

        # Model performance
        best = evaluation['best_metrics']
        f.write(f">> Best Model: {best[0]}\n")
        f.write(f"AUC Score: {best[1]['auc_score']:.4f}\n")
        f.write(f"Expected Business Savings: {best[1]['business_kpis']['expected_savings']:.2f}\n\n")

        # Feature importance with business insight
        if isinstance(feature_importance, pd.DataFrame):
            f.write(">> Top Features and Insights:\n")
            for i, row in feature_importance.head(5).iterrows():
                f.write(f"- {row['feature']}: {row.get('business_insight', 'N/A')} (Importance: {row['importance']:.4f})\n")

    print("✅ Business insight report generated: reports/business_insight_report.txt")

if __name__ == "__main__":
    generate_business_report()
