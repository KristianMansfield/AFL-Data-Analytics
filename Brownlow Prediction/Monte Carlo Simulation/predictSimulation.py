#!/usr/bin/env python3

# Predict a season's Brownlow Medal count by simulation (simulationModel.py).
#
# Outputs (in output/<year>/): leaderboard.csv (with each player's chance of a top 3 and top 10
# finish and average finishing position), game_predictions.csv and round_by_round.csv, as for the
# Plackett-Luce model.
#
# Usage: ../../venv/bin/python predictSimulation.py [--year 2026] [--train-years 2012-2025]
#        [--basic-stats] [--sims 10000] [--player-sd 0.3] [--no-weight-uncertainty]
#        [--order votes|position] [--seed 0]

import argparse
import os

import pandas as pd

from simulationModel import DEFAULT_PLAYER_SD, OUTPUT_DIR, SimulationModel, gameData
from predictionReport import addCommonArguments, loadForArguments, printLeaderboard, writeOutputs


def main():
    parser = addCommonArguments(argparse.ArgumentParser(
        description="Predict a season's Brownlow Medal count by Monte Carlo simulation"))
    parser.add_argument("--sims", type=int, default=10000,
                        help="Simulated seasons (default 10000)")
    parser.add_argument("--player-sd", type=float, default=DEFAULT_PLAYER_SD,
                        help=f"Spread of each player's season-long umpire appeal (default "
                        f"{DEFAULT_PLAYER_SD})")
    parser.add_argument("--no-weight-uncertainty", action="store_true",
                        help="Use the fitted weights in every simulation instead of redrawing them")
    parser.add_argument("--order", choices=["votes", "position"], default="votes",
                        help="Order the leaderboard by average votes (default) or average "
                        "finishing position")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    args = parser.parse_args()

    year, train_years, games = loadForArguments(args)
    features = gameData.featureColumns(gameData.DEFAULT_GROUPS, args.basic_stats)
    model = SimulationModel(features, n_sims=args.sims, player_sd=args.player_sd,
                            weight_uncertainty=not args.no_weight_uncertainty, order=args.order,
                            seed=args.seed).fit(games, train_years)
    predicted = model.predict(games, year)
    print(f"{len(features)} features; simulating {args.sims} seasons...")
    tally = model.tally(predicted)

    out_dir = os.path.join(OUTPUT_DIR, str(year))
    writeOutputs(predicted, tally, out_dir)
    printLeaderboard(predicted, tally, year, args.top)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
