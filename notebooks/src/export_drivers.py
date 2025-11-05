import argparse
import json
import os
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", default=os.path.join("models", "metrics.json"))
    parser.add_argument("--out_dir", default=os.path.join("data", "outputs"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    drivers = obj.get("drivers", {})

    # Logistic coefficients
    logit = drivers.get("logit", {}).get("top_coefficients", [])
    if logit:
        pd.DataFrame(logit).to_csv(os.path.join(args.out_dir, "logit_coeffs.csv"), index=False)

    # RF importances
    rf = drivers.get("rf", {}).get("top_importances", [])
    if rf:
        pd.DataFrame(rf).to_csv(os.path.join(args.out_dir, "rf_importances.csv"), index=False)

    print("Exported:")
    print("-", os.path.join(args.out_dir, "logit_coeffs.csv"))
    print("-", os.path.join(args.out_dir, "rf_importances.csv"))


if __name__ == "__main__":
    main()



