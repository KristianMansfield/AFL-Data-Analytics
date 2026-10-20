#!/usr/bin/env python3

# Predict a season's Brownlow Medal count with the gradient-boosted ranking model
# (rankingModel.py).
#
# Outputs (in output/<year>/): leaderboard.csv, game_predictions.csv, round_by_round.csv, as for
# the Plackett-Luce model, and feature_importance.csv (which stats the trees split on most).
#
# Usage: ../../venv/bin/python predictRanking.py [--year 2026] [--train-years 2012-2025]
#        [--basic-stats] [--objective listmle|lambdarank] [--learning-rate 0.05] [--num-leaves 15]

import argparse
import os

import pandas as pd

from rankingModel import GROUPS, OBJECTIVES, OUTPUT_DIR, BoostedRanker, gameData
from predictionReport import addCommonArguments, loadForArguments, printLeaderboard, writeOutputs


def main():
    parser = addCommonArguments(argparse.ArgumentParser(
        description="Predict a season's Brownlow Medal count with gradient-boosted ranking"))
    parser.add_argument("--objective", choices=OBJECTIVES, default="listmle",
                        help="Ranking loss: listmle (the Plackett-Luce likelihood, default) or "
                        "lambdarank (LightGBM's NDCG ranking objective)")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="How far each tree moves the impacts (default 0.05)")
    parser.add_argument("--num-leaves", type=int, default=15,
                        help="Leaves per tree (default 15)")
    parser.add_argument("--max-rounds", type=int, default=1000,
                        help="Most trees to fit; fewer are used if the held-out season stops "
                        "improving (default 1000)")
    args = parser.parse_args()

    year, train_years, games = loadForArguments(args)
    features = gameData.featureColumns(GROUPS, args.basic_stats)
    model = BoostedRanker(features, objective=args.objective, learning_rate=args.learning_rate,
                          num_leaves=args.num_leaves, max_rounds=args.max_rounds
                          ).fit(games, train_years)
    print(f"{len(features)} features, {args.objective}, {model.n_trees} trees"
          + (f", scale {model.scale:.2f}" if args.objective == "lambdarank" else ""))
    predicted = model.predict(games, year)
    tally = model.tally(predicted)

    out_dir = os.path.join(OUTPUT_DIR, str(year))
    writeOutputs(predicted, tally, out_dir)
    importance = model.featureImportance()
    importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False,
                      float_format="%.4f")
    print("\nMost used features (share of the trees' improvement):")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:<26} {row['importance']:.1%}")
    printLeaderboard(predicted, tally, year, args.top)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
