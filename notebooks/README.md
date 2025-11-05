Customer Churn Prediction (Python, SQL, Scikit-learn, Tableau)

This project predicts telecom customer churn using Logistic Regression and Random Forest, with modular feature engineering, scoring export for Tableau, and an A/B test simulation.

Project Structure

.
├─ data/
│  ├─ raw/
│  └─ outputs/
├─ models/
├─ sql/
│  └─ schema.sql
├─ src/
│  ├─ config.py
│  ├─ generate_synthetic_data.py
│  ├─ data_ingest.py
│  ├─ feature_engineering.py
│  ├─ train_models.py
│  ├─ score_export.py
│  └─ ab_test_simulation.py
└─ requirements.txt

Quickstart (Windows PowerShell)

1) python -m venv .venv
2) .\.venv\Scripts\Activate.ps1
3) python -m pip install --upgrade pip
4) pip install -r requirements.txt

Generate synthetic data (optional):
python .\src\generate_synthetic_data.py --num_rows 50000 --out_path .\data\raw\telecom_churn.csv

Ingest CSV to SQLite:
python .\src\data_ingest.py --csv_path .\data\raw\telecom_churn.csv --db_path .\data\churn.db

Build features:
python .\src\feature_engineering.py --db_path .\data\churn.db

Train models:
python .\src\train_models.py --db_path .\data\churn.db --models_dir .\models

Export scores for Tableau:
python .\src\score_export.py --db_path .\data\churn.db --models_dir .\models --out_csv .\data\outputs\churn_scores.csv

A/B test simulation:
python .\src\ab_test_simulation.py --scores_csv .\data\outputs\churn_scores.csv --strategy "target_high_risk" --effect 0.15 --seed 42

Tableau: Connect to data/outputs/churn_scores.csv and build visuals by segment and key drivers.

