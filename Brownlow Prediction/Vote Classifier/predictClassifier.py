#!/usr/bin/env python3

# Predict a season's Brownlow Medal count with the vote classifier (classifierModel.py).
#
# Outputs (in output/<year>/): leaderboard.csv, game_predictions.csv, round_by_round.csv, as for
# the Plackett-Luce model, and, when the season's votes are known, feature_importance.csv (how
# much worse the predictions get when each feature is shuffled).
#
# Usage: ../../venv/bin/python predictClassifier.py [--year 2026] [--train-years 2012-2025]
#        [--basic-stats]

import argparse
import os

import pandas as pd

from classifierModel import GROUPS, OUTPUT_DIR, VoteClassifier, gameData
from predictionReport import addCommonArguments, loadForArguments, printLeaderboard, writeOutputs


def main():
    parser = addCommonArguments(argparse.ArgumentParser(
        description="Predict a season's Brownlow Medal count with the vote classifier"))
    args = parser.parse_args()

    year, train_years, games = loadForArguments(args)
    features = gameData.featureColumns(GROUPS, args.basic_stats)
    model = VoteClassifier(features, basic_stats=args.basic_stats).fit(games, train_years)
    print(f"{len(model.columns)} features, {model.model.n_iter_} boosting rounds")
    predicted = model.predict(games, year)
    tally = model.tally(predicted)

    out_dir = os.path.join(OUTPUT_DIR, str(year))
    writeOutputs(predicted, tally, out_dir)
    if not gameData.votedGames(games[games["year"] == year]).empty:
        importance = model.featureImportance(games, year)
        importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False,
                          float_format="%.5f")
        print("\nMost important features (log loss increase when shuffled):")
        for _, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<26} {row['log_loss_increase']:.4f}")
    printLeaderboard(predicted, tally, year, args.top)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
