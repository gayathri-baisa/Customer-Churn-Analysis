import argparse
import os
import pandas as pd


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--scores_csv", default=os.path.join("data", "outputs", "churn_scores.csv"))
	parser.add_argument("--threshold", type=float, default=0.66)
	parser.add_argument("--out_csv", default=os.path.join("data", "outputs", "target_list.csv"))
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
	df = pd.read_csv(args.scores_csv)

	selected = df[df["churn_proba"] >= args.threshold].copy()
	selected = selected.sort_values("churn_proba", ascending=False)
	cols = ["customer_id", "churn_proba", "segment", "churn_label"]
	selected[cols].to_csv(args.out_csv, index=False)
	print(f"Wrote {len(selected):,} targets to {args.out_csv} at threshold {args.threshold}")


if __name__ == "__main__":
	main()

