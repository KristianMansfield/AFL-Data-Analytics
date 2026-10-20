#!/usr/bin/env python3

# Predict a season's Brownlow Medal count with the Plackett-Luce model.
#
# What it does: fits the vote model (brownlowModel.py, rankedLogit.py) on the votes from the training
# seasons, then for every home and away game in the target season gives each player their chance
# of 3, 2 and 1 votes and their expected votes (a share of the 6 votes in the game). It adds
# those up into a leaderboard with each player's chance of winning (seasonTally.py), and the
# alternative strict 3-2-1 count where the model's top three in each game get 3, 2 and 1.
#
# Outputs (in output/<year>/): leaderboard.csv, game_predictions.csv (every player in every game),
# round_by_round.csv (the leaders' running totals) and coefficients.csv (what the model learnt).
# If the target season's votes are known, they're shown alongside for comparison.
#
# With --basic-stats it leaves out the afl.com.au features and trains from 2003 instead of 2012.
#
# Usage: ../../venv/bin/python predictBrownlow.py [--year 2026] [--train-years 2012-2025]
#        [--basic-stats] [--groups basic advanced rating rank result progression form appeal]

import argparse
import os

import pandas as pd

import gameData
from brownlowModel import BrownlowModel
from predictionReport import addCommonArguments, loadForArguments, printLeaderboard, writeOutputs
from seasonTally import DEFAULT_PLAYER_SD


def main():
    parser = addCommonArguments(argparse.ArgumentParser(
        description="Predict a season's Brownlow Medal count with the Plackett-Luce model"))
    parser.add_argument(
        "--groups", nargs="+", default=gameData.DEFAULT_GROUPS,
        help=f"Feature groups to use (default: {' '.join(gameData.DEFAULT_GROUPS)}; "
        f"available: {' '.join(gameData.FEATURE_GROUPS)})",
    )
    parser.add_argument("--l2", type=float, default=0.01, help="L2 penalty (default 0.01)")
    parser.add_argument(
        "--player-sd", type=float, default=DEFAULT_PLAYER_SD,
        help="Spread of each player's season-long umpire appeal, on the impact scale, used for "
        f"the ranges and win chances (default {DEFAULT_PLAYER_SD}; 0 treats every game as "
        "independent)",
    )
    args = parser.parse_args()

    year, train_years, games = loadForArguments(args)
    features = gameData.featureColumns(args.groups, args.basic_stats)
    print(f"{len(features)} features")
    model = BrownlowModel(features, l2=args.l2).fit(games, train_years)
    predicted = model.predict(games, year)
    tally = model.tally(predicted, args.player_sd)

    out_dir = os.path.join(gameData.OUTPUT_DIR, str(year))
    writeOutputs(predicted, tally, out_dir)
    coefficients = model.coefficients()
    coefficients.to_csv(os.path.join(out_dir, "coefficients.csv"), index=False,
                        float_format="%.4f")

    print("\nWhat moves a player's chance of polling (per standard deviation of the feature):")
    for _, row in coefficients.head(15).iterrows():
        print(f"  {row['feature']:<26} x{row['multiplier_per_sd']:.2f}")
    printLeaderboard(predicted, tally, year, args.top)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
