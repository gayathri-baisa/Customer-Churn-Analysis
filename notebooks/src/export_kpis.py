import argparse
import json
import os
import numpy as np
import pandas as pd


def simulate(scores: pd.DataFrame, effect: float, strategy: str = "target_high_risk", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    df = scores.copy()
    base_p = df["churn_proba"].clip(0, 1).to_numpy()
    if strategy == "target_high_risk":
        treat_mask = (df["segment"] == "High").to_numpy()
    elif strategy == "target_medium_high":
        treat_mask = df["segment"].isin(["Medium", "High"]).to_numpy()
    else:
        treat_mask = rng.random(len(df)) < 0.5
    treated_p = base_p * (1 - effect)
    final_p = np.where(treat_mask, treated_p, base_p)
    outcomes_control = rng.binomial(1, base_p)
    outcomes_treated = rng.binomial(1, final_p)
    cr_c = outcomes_control.mean()
    cr_t = outcomes_treated.mean()
    abs_red = cr_c - cr_t
    rel_red = abs_red / max(cr_c, 1e-9)
    return {
        "control_rate": float(cr_c),
        "treated_rate": float(cr_t),
        "absolute_reduction": float(abs_red),
        "relative_reduction": float(rel_red),
        "treated_share": float(treat_mask.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", default=os.path.join("data", "outputs", "churn_scores.csv"))
    parser.add_argument("--out_json", default=os.path.join("data", "outputs", "kpis.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    df = pd.read_csv(args.scores_csv)

    overall = {
        "num_customers": int(len(df)),
        "avg_churn_proba": float(df["churn_proba"].mean()),
        "share_high": float((df["segment"] == "High").mean()),
        "share_medium": float((df["segment"] == "Medium").mean()),
        "share_low": float((df["segment"] == "Low").mean()),
    }

    k10 = simulate(df, effect=0.10, strategy="target_high_risk", seed=42)
    k20 = simulate(df, effect=0.20, strategy="target_high_risk", seed=42)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "ab_10": k10, "ab_20": k20}, f, indent=2)

    print(f"Wrote KPIs -> {args.out_json}")


if __name__ == "__main__":
    main()



