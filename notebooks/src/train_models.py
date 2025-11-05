import argparse
import json
import os
import sqlite3
from typing import Dict, Tuple, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


RNG_SEED = 42


def load_feature_table(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM customers_features", conn)


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_monthly_spend",
        "support_calls_rate",
        "is_auto_pay",
        "is_paperless",
        "is_senior_citizen",
        "has_partner",
        "has_dependents",
    ]
    categorical_features = ["contract_type", "internet_service", "payment_method"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def _get_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    names: List[str] = []
    # numeric pipeline -> same names
    names.extend(numeric_features)
    # categorical pipeline -> onehot names
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
    names.extend(cat_ohe_names)
    return names


def train_and_eval(X: pd.DataFrame, y: pd.Series, models_dir: str) -> Dict[str, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    logit = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=200, solver="lbfgs", class_weight="balanced", random_state=RNG_SEED),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )

    results: Dict[str, Dict[str, float]] = {}
    drivers: Dict[str, Any] = {}

    for name, model in [("logit", logit), ("rf", rf)]:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
            results[name] = metrics
            joblib.dump(model, os.path.join(models_dir, f"{name}_model.joblib"))
            print(f"Model {name}: {json.dumps(metrics, indent=2)}")
            print("Classification report:\n" + classification_report(y_test, y_pred, digits=4))

            # Extract key drivers
            pre_fitted: ColumnTransformer = model.named_steps["preprocessor"]
            feat_names = _get_feature_names(pre_fitted, numeric_features, categorical_features)
            if name == "logit":
                clf: LogisticRegression = model.named_steps["clf"]
                coefs = clf.coef_[0]
                top = sorted(
                    [{"feature": f, "coefficient": float(w)} for f, w in zip(feat_names, coefs)],
                    key=lambda d: abs(d["coefficient"]),
                    reverse=True,
                )[:25]
                drivers[name] = {"top_coefficients": top}
            elif name == "rf":
                clf: RandomForestClassifier = model.named_steps["clf"]
                imps = clf.feature_importances_
                top = sorted(
                    [{"feature": f, "importance": float(w)} for f, w in zip(feat_names, imps)],
                    key=lambda d: d["importance"],
                    reverse=True,
                )[:25]
                drivers[name] = {"top_importances": top}
        except Exception as e:
            print(f"Training {name} failed: {e}")

    # Decide primary model by ROC AUC
    primary = max(results.items(), key=lambda kv: kv[1]["roc_auc"])[0]
    with open(os.path.join(models_dir, "model_selection.json"), "w", encoding="utf-8") as f:
        json.dump({"primary_model": primary, "metrics": results}, f, indent=2)

    # Write metrics and key drivers for Tableau/use in docs
    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "drivers": drivers}, f, indent=2)
    print(f"Selected primary model: {primary}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=os.path.join("data", "churn.db"))
    parser.add_argument("--models_dir", default=os.path.join("models"))
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    with sqlite3.connect(args.db_path) as conn:
        df = load_feature_table(conn)

    target = "churned"
    feature_cols = [c for c in df.columns if c not in ("customer_id", target)]
    X = df[feature_cols]
    y = df[target].astype(int)

    train_and_eval(X, y, models_dir=args.models_dir)


if __name__ == "__main__":
    main()


