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
# With --ensemble each player's impact is a blend of this model (70%) and the gradient-boosted
# ranker (30%; ensembleModel.py), written to output/<year>-ensemble/. The blend predicts the
# contenders' votes game by game better, but wasn't clearly better at picking the winner or the
# top 10 order in the backtests. --ensemble-weight changes the Plackett-Luce share.
#
# Usage: ../../venv/bin/python predictBrownlow.py [--year 2026] [--train-years 2012-2025]
#        [--basic-stats] [--groups basic advanced rating rank result progression form appeal]
#        [--ensemble [--ensemble-weight 0.7]]

import argparse
import os

import pandas as pd

import gameData
from brownlowModel import BrownlowModel
from ensembleModel import DEFAULT_WEIGHT, EnsembleModel
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
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Blend each player's impact with the gradient-boosted ranker's (needs LightGBM)",
    )
    parser.add_argument(
        "--ensemble-weight", type=float, default=DEFAULT_WEIGHT,
        help=f"Plackett-Luce share of the blend with --ensemble (default {DEFAULT_WEIGHT})",
    )
    args = parser.parse_args()

    year, train_years, games = loadForArguments(args)
    features = gameData.featureColumns(args.groups, args.basic_stats)
    if args.ensemble:
        from rankingModel import GROUPS as RANKING_GROUPS

        print(f"{len(features)} features, blended {args.ensemble_weight:.0%} / "
              f"{1 - args.ensemble_weight:.0%} with the gradient-boosted ranker")
        model = EnsembleModel(features, gameData.featureColumns(RANKING_GROUPS, args.basic_stats),
                              weight=args.ensemble_weight, l2=args.l2)
    else:
        print(f"{len(features)} features")
        model = BrownlowModel(features, l2=args.l2)
    model.fit(games, train_years)
    predicted = model.predict(games, year)
    tally = model.tally(predicted, args.player_sd)

    out_dir = os.path.join(gameData.OUTPUT_DIR, f"{year}-ensemble" if args.ensemble else str(year))
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
