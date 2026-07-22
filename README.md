# 📂 Data Pipeline with DVC and MLflow

This project demonstrates an end-to-end **machine learning pipeline** using:
- **[DVC](ca://s?q=Data_Version_Control_DVC)** for data/model versioning and reproducibility.
- **[MLflow](ca://s?q=MLflow_experiment_tracking)** for experiment tracking and comparison.

The pipeline trains a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, with clear stages for preprocessing, training, and evaluation.

---

## ✅ Key Features

- **[Data Version Control](ca://s?q=DVC_for_data_and_model_versioning)**  
  Track datasets, models, and pipeline stages. Re-run automatically when dependencies change.

- **[Experiment Tracking](ca://s?q=MLflow_for_experiment_tracking)**  
  Log hyperparameters, metrics (accuracy), and artifacts. Compare runs easily.

- **[Pipeline Stages](ca://s?q=Pipeline_stages_in_ML_project)**  
  - **Preprocessing** → `preprocess.py` cleans raw data.  
  - **Training** → `train.py` trains Random Forest, saves model, logs to MLflow.  
  - **Evaluation** → `evaluate.py` tests model accuracy, logs metrics.
 
  - **[Training Strategies](ca://s?q=Training_strategies_in_ML_pipeline)**  
  Models were trained using:
  - Grid sampling  
  - Random sampling  

---


