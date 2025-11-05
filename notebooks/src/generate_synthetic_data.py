import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple


RNG_SEED = 42


def simulate_dataset(num_rows: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"C{100000 + i}" for i in range(num_rows)]

    tenure_months = rng.integers(0, 72, size=num_rows)
    monthly_charges = rng.normal(70, 25, size=num_rows).clip(15, 200)
    total_charges = (monthly_charges * (tenure_months + rng.normal(0.0, 1.5, size=num_rows))).clip(0)
    num_support_calls = rng.poisson(1.5, size=num_rows)

    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.6, 0.25, 0.15], size=num_rows)
    internet_service = rng.choice(["DSL", "Fiber optic", "None"], p=[0.4, 0.5, 0.1], size=num_rows)
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=num_rows)
    auto_pay = rng.choice([0, 1], p=[0.55, 0.45], size=num_rows)
    has_paperless_billing = rng.choice([0, 1], p=[0.45, 0.55], size=num_rows)
    is_senior_citizen = rng.choice([0, 1], p=[0.83, 0.17], size=num_rows)
    has_partner = rng.choice([0, 1], p=[0.52, 0.48], size=num_rows)
    has_dependents = rng.choice([0, 1], p=[0.7, 0.3], size=num_rows)

    # Latent churn propensity
    z = (
        0.03 * (70 - monthly_charges)
        + 0.05 * (3 - np.minimum(num_support_calls, 5))
        + 0.06 * (tenure_months < 6).astype(float)
        + 0.08 * (contract_type == "Month-to-month").astype(float)
        + 0.04 * (internet_service == "Fiber optic").astype(float)
        - 0.06 * (auto_pay == 1).astype(float)
        - 0.03 * (has_paperless_billing == 1).astype(float)
        + 0.02 * (is_senior_citizen == 1).astype(float)
    )
    churn_prob = 1 / (1 + np.exp(-(-0.2 + z)))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "auto_pay": auto_pay,
            "has_paperless_billing": has_paperless_billing,
            "is_senior_citizen": is_senior_citizen,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "churned": churned,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", type=int, default=50000)
    parser.add_argument("--out_path", type=str, default=os.path.join("data", "raw", "telecom_churn.csv"))
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df = simulate_dataset(args.num_rows, seed=args.seed)
    df.to_csv(args.out_path, index=False)
    print(f"Wrote synthetic dataset: {args.out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()


