import argparse
import os
import sqlite3
import pandas as pd
import numpy as np


# Map for adapting alternate column names if needed (extend as needed)
FEATURE_COLUMN_MAP = {
    "customer_id": "customer_id",
    "tenure_months": "tenure_months",
    "monthly_charges": "monthly_charges",
    "total_charges": "total_charges",
    "num_support_calls": "num_support_calls",
    "contract_type": "contract_type",
    "internet_service": "internet_service",
    "payment_method": "payment_method",
    "auto_pay": "auto_pay",
    "has_paperless_billing": "has_paperless_billing",
    "is_senior_citizen": "is_senior_citizen",
    "has_partner": "has_partner",
    "has_dependents": "has_dependents",
    "churned": "churned",
}


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Coerce numeric types and handle edge cases
    numeric_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_calls",
        "auto_pay",
        "has_paperless_billing",
        "is_senior_citizen",
        "has_partner",
        "has_dependents",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature: average monthly spend (robust to missing total/tenure)
    df["avg_monthly_spend"] = (
        df["total_charges"] / np.where(df["tenure_months"] > 0, df["tenure_months"], np.nan)
    )
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["monthly_charges"]).astype(float)

    # Feature: support calls per month
    df["support_calls_rate"] = (
        df["num_support_calls"].fillna(0) / (df["tenure_months"].replace(0, np.nan))
    ).fillna(df["num_support_calls"].fillna(0))

    # Binary cleanups
    df["is_auto_pay"] = (df["auto_pay"] == 1).astype(int)
    df["is_paperless"] = (df["has_paperless_billing"] == 1).astype(int)

    keep_cols = [
        "customer_id",
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
        "contract_type",
        "internet_service",
        "payment_method",
        "churned",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    return df[keep_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=os.path.join("data", "churn.db"))
    args = parser.parse_args()

    with sqlite3.connect(args.db_path) as conn:
        df_raw = pd.read_sql("SELECT * FROM customers_raw", conn)
        df_feat = build_feature_frame(df_raw)
        df_feat.to_sql("customers_features", conn, if_exists="replace", index=False)

    print(f"Built features into table: customers_features (db: {args.db_path})")


if __name__ == "__main__":
    main()


