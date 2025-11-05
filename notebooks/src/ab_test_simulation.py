import argparse
import numpy as np
import pandas as pd


RNG_SEED = 42


def simulate_ab(scores: pd.DataFrame, strategy: str, effect: float, seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(seed)

    df = scores.copy()
    # Base churn probability is the model probability
    base_p = df["churn_proba"].clip(0, 1).to_numpy()

    if strategy == "target_high_risk":
        treat_mask = (df["segment"] == "High").to_numpy()
    elif strategy == "target_medium_high":
        treat_mask = df["segment"].isin(["Medium", "High"]).to_numpy()
    else:
        # Uniform random treatment at 50%
        treat_mask = rng.random(len(df)) < 0.5

    # Apply treatment effect: multiplicative reduction on risk for treated
    treated_p = base_p * (1 - effect)
    final_p = np.where(treat_mask, treated_p, base_p)

    # Simulate outcomes
    outcomes_control = rng.binomial(1, base_p)
    outcomes_treated = rng.binomial(1, final_p)

    # Compute metrics
    churn_rate_control = outcomes_control.mean()
    churn_rate_treated = outcomes_treated.mean()
    abs_reduction = churn_rate_control - churn_rate_treated
    rel_reduction = abs_reduction / max(churn_rate_control, 1e-9)

    return {
        "control_rate": float(churn_rate_control),
        "treated_rate": float(churn_rate_treated),
        "absolute_reduction": float(abs_reduction),
        "relative_reduction": float(rel_reduction),
        "treated_share": float(treat_mask.mean()),
    }


def bootstrap_ci(scores: pd.DataFrame, strategy: str, effect: float, n_boot: int = 500, seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(seed)
    reductions = []
    for _ in range(n_boot):
        sample = scores.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1e9)))
        res = simulate_ab(sample, strategy=strategy, effect=effect, seed=int(rng.integers(0, 1e9)))
        reductions.append(res["absolute_reduction"])
    lo, hi = np.percentile(reductions, [2.5, 97.5])
    return {"abs_reduction_ci": [float(lo), float(hi)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--strategy", default="target_high_risk")
    parser.add_argument("--effect", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)

    point = simulate_ab(scores, strategy=args.strategy, effect=args.effect, seed=args.seed)
    ci = bootstrap_ci(scores, strategy=args.strategy, effect=args.effect, n_boot=300, seed=args.seed)

    print("A/B test simulation results:")
    print({**point, **ci})


if __name__ == "__main__":
    main()


