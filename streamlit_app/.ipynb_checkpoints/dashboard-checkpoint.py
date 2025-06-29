import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tempfile
import pdfkit
import shutil
import os
import sys
from jinja2 import Environment, FileSystemLoader

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.checkpoint_manager import CheckpointManager

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Locate wkhtmltopdf executable
path_to_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\bin\wkhtmltopdf.exe"
if not path_to_wkhtmltopdf:
    raise FileNotFoundError("wkhtmltopdf not found. Please install it and ensure it is in your system PATH.")

pdf_config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

cm = CheckpointManager()

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["EDA", "Model Performance", "Feature Insights", "Fraud Predictions", "Business Summary"])

eda = cm.load_checkpoint('eda_complete.pkl')
evaluation = cm.load_checkpoint('evaluation_complete.pkl')
predictions_data = cm.load_checkpoint('final_predictions.pkl')

def generate_business_pdf(model_name, metrics, feature_df):
    env = Environment(loader=FileSystemLoader("streamlit_app/templates"))
    template = env.get_template("business_report.html")

    if not feature_df.empty and 'business_insight' in feature_df.columns:
        feature_table = feature_df[['feature', 'importance', 'business_insight']].head(10).to_dict(orient='records')
    else:
        feature_table = []

    html = template.render(
        model_name=model_name,
        metrics=metrics,
        feature_table=feature_table
    )

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdfkit.from_string(html, tmp_path, configuration=pdf_config)
    return tmp_path

if section == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Fraud Rate by Risk Segment (Transaction Amount Quartiles)")
    if 'fraud_by_risk_segment' in eda:
        st.dataframe(eda['fraud_by_risk_segment'])
        fig = px.bar(eda['fraud_by_risk_segment'].reset_index(), x='risk_segment', y='mean', title="Fraud Rate by Risk Segment")
        st.plotly_chart(fig)

    st.subheader("Fraud Rate by Hour of Day")
    if 'fraud_by_hour' in eda:
        fig = px.line(eda['fraud_by_hour'].reset_index(), x='hour', y='is_fraud', title="Fraud Rate by Hour")
        st.plotly_chart(fig)

    st.subheader("Correlation Matrix")
    if 'correlation_matrix' in eda:
        corr = eda['correlation_matrix']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif section == "Model Performance":
    st.title("Model Evaluation and KPIs")

    results = evaluation.get('evaluation_results', {})
    for model, res in results.items():
        st.subheader(f"{model.upper()}")

        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{res['auc_score']:.4f}")
        col2.metric("Precision", f"{res['precision']:.2f}")
        col3.metric("Recall", f"{res['recall']:.2f}")
        
        with st.expander("Classification Report"):
            st.json(res['classification_report'])

        if 'business_kpis' in res:
            st.markdown("Business KPIs")
            kpis = res['business_kpis']
            st.write({
                "Expected Savings": round(kpis['expected_savings'], 2),
                "True Positive Gain": round(kpis['true_positive_gain'], 2),
                "False Positive Cost": round(kpis['false_positive_cost'], 2),
            })

elif section == "Feature Insights":
    st.title("Feature Importance and Business Insight")

    if isinstance(predictions_data.get('feature_importance'), pd.DataFrame):
        feat_imp = predictions_data['feature_importance']
        top_n = st.slider("Select Top N Features", min_value=5, max_value=50, value=10)
        top_features = feat_imp.head(top_n)

        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                     hover_data=['business_insight'] if 'business_insight' in top_features else None,
                     title="Top Feature Importances")
        st.plotly_chart(fig)

        if 'business_insight' in top_features.columns:
            st.markdown("Business Insight Annotations")
            st.dataframe(top_features[['feature', 'importance', 'business_insight']])

elif section == "Fraud Predictions":
    st.title("Ranked Fraud Predictions")

    test_preds = predictions_data.get('test_predictions')
    test_binary = predictions_data.get('test_predictions_binary')
    X_test = cm.load_checkpoint('features_prepared.pkl')['X_test']

    if test_preds is not None and X_test is not None:
        df_preds = X_test.copy()
        df_preds['fraud_score'] = test_preds
        df_preds['is_predicted_fraud'] = test_binary

        st.markdown("Filter predictions by fraud score")
        score_range = st.slider("Fraud Score Range", 0.0, 1.0, (0.5, 1.0))
        filtered = df_preds[(df_preds['fraud_score'] >= score_range[0]) & (df_preds['fraud_score'] <= score_range[1])]
        
        st.dataframe(filtered.sort_values('fraud_score', ascending=False).reset_index(drop=True), use_container_width=True)

elif section == "Business Summary":
    st.title("Business Report Summary")

    best_model = evaluation.get("best_metrics", (None, None))
    model_name = best_model[0]
    metrics = best_model[1]

    st.subheader("Best Performing Model")
    if model_name and metrics:
        st.markdown(f"Model: `{model_name}`")
        st.markdown(f"AUC Score: `{metrics['auc_score']:.4f}`")
        st.markdown(f"Precision: `{metrics['precision']:.4f}`")
        st.markdown(f"Recall: `{metrics['recall']:.4f}`")

        if 'business_kpis' in metrics:
            st.subheader("Estimated Business Impact")
            kpi = metrics['business_kpis']
            st.write({
                "Expected Business Savings": round(kpi['expected_savings'], 2),
                "True Positive Gain": round(kpi['true_positive_gain'], 2),
                "False Positive Cost": round(kpi['false_positive_cost'], 2)
            })

    if st.button("Generate & Download PDF Report"):
        feature_df = predictions_data.get("feature_importance", pd.DataFrame())
        pdf_path = generate_business_pdf(model_name, metrics, feature_df)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Business Insight Report (PDF)",
                data=f,
                file_name="fraud_business_report.pdf",
                mime="application/pdf"
            )
