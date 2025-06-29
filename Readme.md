# Fraud Detection Pipeline

A professional, modular, and checkpointed fraud detection system designed for end-to-end machine learning, business insight, and reporting.

---

## 📦 Project Overview

This pipeline is built to:

* Detect fraudulent transactions using advanced modeling techniques
* Support modular, reproducible, and restartable development
* Provide actionable **business insights** and visual summaries
* Enable professional **PDF report generation** and **Streamlit dashboarding**

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Data

Put the following files in `data/raw/`:

* `fraudTrain.csv`
* `fraudTest.csv`

### 3. Run the Full Pipeline

```bash
python main.py --action run
```

### 4. Resume from a Checkpoint

```bash
python main.py --action resume --checkpoint features_prepared.pkl
```

### 5. View Available Checkpoints

```bash
python main.py --action list
```

### 6. Launch the Dashboard

```bash
streamlit run streamlit_app/dashboard.py
```

---

## 📊 Business Insights

This pipeline integrates key business intelligence:

* Fraud rates segmented by transaction time, amount, and category
* Financial impact metrics like expected savings, cost of false positives
* Top contributing features with mapped business context
* Downloadable **PDF business reports** from the Streamlit dashboard

Generate report manually:

```bash
python generate_business_report.py
```

---

## 📁 Project Structure

```
fraud_detection_project/
├── config/                     # Configuration files
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Processed datasets
│   └── checkpoints/            # Pipeline checkpoints
├── models/                     # Saved production-ready models
├── reports/                    # Generated reports (PDF, CSV)
├── src/
│   ├── data_processing/        # Data loading, EDA, cleaning
│   ├── feature_engineering/    # Feature creation and selection
│   ├── modeling/               # Model training and evaluation
│   ├── utils/                  # Checkpointing, config
├── streamlit_app/              # Dashboard UI + templates
├── main.py                     # Pipeline entry script
├── generate_business_report.py # Optional manual PDF report generator
└── requirements.txt
```

---

## 🔁 Pipeline Phases

Each major pipeline step saves a checkpoint:

1. **Data Loading & EDA**        — `eda_complete.pkl`
2. **Data Cleaning**             — `data_cleaned.pkl`
3. **Feature Engineering**       — `features_engineered.pkl`
4. **Feature Preparation**       — `features_prepared.pkl`
5. **Model Training**            — `models_trained.pkl`
6. **Model Evaluation**          — `evaluation_complete.pkl`
7. **Final Predictions**         — `final_predictions.pkl`
8. **Model Exporting**           — `model_saved.pkl`

Resume any phase using `main.py --action resume`

---

## 📌 Usage Examples

### Run Complete Pipeline in Code

```python
from main import run_complete_pipeline
predictions, feature_importance = run_complete_pipeline()
```

### Resume from Specific Phase

```python
from main import resume_from_checkpoint
resume_from_checkpoint('features_prepared.pkl')
```

### Load a Checkpoint

```python
from src.utils.checkpoint_manager import CheckpointManager
cm = CheckpointManager()
data = cm.load_checkpoint('models_trained.pkl')
```

---

## ⚙️ Customization

* Modify configs in: `src/utils/config.py`
* Add features in: `src/feature_engineering/feature_creator.py`
* Add new models to: `MODELS` in `model_trainer.py`
* Adjust metrics in: `src/modeling/model_evaluator.py`

---

## 📑 Setup Instructions (from scratch)

### 1. Create Project Structure

```bash
mkdir fraud_detection_project && cd fraud_detection_project
mkdir -p data/{raw,processed,checkpoints}
mkdir -p src/{data_processing,feature_engineering,modeling,utils}
mkdir -p {models,reports,notebooks,config,streamlit_app/templates}
touch src/__init__.py src/*/__init__.py
```

### 2. Create and Populate Files

Add the following core files:

#### Core utilities

* `src/utils/checkpoint_manager.py`
* `src/utils/config.py`

#### Data processing

* `src/data_processing/data_loader.py`
* `src/data_processing/eda.py`
* `src/data_processing/data_cleaner.py`

#### Feature engineering

* `src/feature_engineering/feature_creator.py`
* `src/feature_engineering/feature_selector.py`

#### Modeling

* `src/modeling/model_trainer.py`
* `src/modeling/model_evaluator.py`
* `src/modeling/predictor.py`

#### Dashboard

* `streamlit_app/dashboard.py`
* `streamlit_app/templates/business_report.html`

#### Entry points

* `main.py`
* `generate_business_report.py`
* `requirements.txt`
* `README.md`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Data

Add your `train.csv` and `test.csv` files to `data/raw/`

### 5. Run or Resume Pipeline

```bash
python main.py --action run
python main.py --action resume --checkpoint features_prepared.pkl
```

### 6. Run Dashboard

```bash
streamlit run streamlit_app/dashboard.py
```

---

## ✅ Key Features

* Modular architecture for clean development
* Checkpointing and resume logic for recoverability
* Streamlit dashboard for business users
* PDF report generation with business-aligned insights
* Easily extendable for additional features or models

---

## 📌 License

This project is licensed for personal or organizational educational use. Contact the author for commercial licensing options.

---

## Acknowledgements

- Dataset gotten from kaggle kartik2112/fraud-detection
- Developed to address the critical need for intelligent fraud detection systems
- Part of ongoing data science research and development

---

## Author

**Fidelis Akinbule**

- GitHub: [Fidelis-Akinbule](https://github.com/Fidelis-Akinbule)
- LinkedIn: [fidelis-akinbule](https://www.linkedin.com/in/fidelis-akinbule/)

---

*"Effective fraud detection systems combine predictive analytics with operational excellence to safeguard financial transactions."*

