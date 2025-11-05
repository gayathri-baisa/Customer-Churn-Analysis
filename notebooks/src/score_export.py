import argparse
import json
import os
import sqlite3
import joblib
import pandas as pd


def load_primary_model(models_dir: str):
    sel_path = os.path.join(models_dir, "model_selection.json")
    if not os.path.exists(sel_path):
        # Fallback to RF if selection file missing
        primary = "rf"
    else:
        with open(sel_path, "r", encoding="utf-8") as f:
            primary = json.load(f)["primary_model"]
    model_path = os.path.join(models_dir, f"{primary}_model.joblib")
    return joblib.load(model_path), primary


def segment_risk(p: float) -> str:
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Medium"
    return "Low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=os.path.join("data", "churn.db"))
    parser.add_argument("--models_dir", default=os.path.join("models"))
    parser.add_argument("--out_csv", default=os.path.join("data", "outputs", "churn_scores.csv"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with sqlite3.connect(args.db_path) as conn:
        df = pd.read_sql("SELECT * FROM customers_features", conn)

    target = "churned"
    feature_cols = [c for c in df.columns if c not in ("customer_id", target)]
    X = df[feature_cols]

    model, primary = load_primary_model(args.models_dir)
    proba = model.predict_proba(X)[:, 1]
    label = (proba >= 0.5).astype(int)
    seg = [segment_risk(p) for p in proba]

    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "churn_proba": proba,
            "churn_label": label,
            "segment": seg,
        }
    )
    out.to_csv(args.out_csv, index=False)
    print(f"Primary model: {primary}. Wrote scores -> {args.out_csv}")


if __name__ == "__main__":
    main()


