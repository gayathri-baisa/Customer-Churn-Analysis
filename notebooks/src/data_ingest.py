import argparse
import os
import sqlite3
import pandas as pd
from src.config import ensure_dirs_exist


def create_schema(conn: sqlite3.Connection, schema_path: str) -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    conn.executescript(sql)


def ingest_csv(conn: sqlite3.Connection, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    # Normalize column names to expected set if present
    expected = {
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_calls",
        "contract_type",
        "internet_service",
        "payment_method",
        "auto_pay",
        "has_paperless_billing",
        "is_senior_citizen",
        "has_partner",
        "has_dependents",
        "churned",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df.to_sql("customers_raw", conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--db_path", default=os.path.join("data", "churn.db"))
    parser.add_argument("--schema_path", default=os.path.join("sql", "schema.sql"))
    args = parser.parse_args()

    ensure_dirs_exist()
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    with sqlite3.connect(args.db_path) as conn:
        create_schema(conn, args.schema_path)
        ingest_csv(conn, args.csv_path)

    print(f"Ingested {args.csv_path} into SQLite DB: {args.db_path}")


if __name__ == "__main__":
    main()


