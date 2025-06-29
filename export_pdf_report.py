from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from src.utils.checkpoint_manager import CheckpointManager

def export_pdf_report(output_path="reports/fraud_business_report.pdf"):
    cm = CheckpointManager()

    # Load necessary data
    evaluation = cm.load_checkpoint("evaluation_complete.pkl")
    best_model, metrics = evaluation['best_metrics']
    feature_df = cm.load_checkpoint("final_predictions.pkl")['feature_importance']

    if feature_df is not None and 'business_insight' in feature_df.columns:
        feature_table = feature_df[['feature', 'importance', 'business_insight']].head(10).to_dict(orient='records')
    else:
        feature_table = []

    # Render template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("business_report.html")
    html_out = template.render(model_name=best_model, metrics=metrics, feature_table=feature_table)

    # Convert to PDF
    HTML(string=html_out).write_pdf(output_path)
    print(f" PDF Report generated at: {output_path}")

if __name__ == "__main__":
    export_pdf_report()
