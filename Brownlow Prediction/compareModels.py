#!/usr/bin/env python3

# Backtest every Brownlow model the same way and compare them on picking the winner and the top 10
# in order.
#
# What it does: for each test season, fits each model on the earlier seasons only, predicts the
# season and scores the leaderboard:
#   winner rank   where the actual winner sat on the predicted leaderboard (1 = picked)
#   top 10        how many of the actual top 10 were in the predicted top 10
#   top 10 order  Spearman correlation between the actual and predicted positions of the actual
#                 top 10 (1 = exactly the right order)
#   top 10 MAE    mean absolute error of the actual top 10's predicted totals
#   contender ll  log probability of the votes the actual top 10 got in each of their games: a
#                 steadier measure than the winner, as it scores ~230 player-games a season
#   top pick      share of games where the model's most likely 3 vote player got the 3 votes
# The Monte Carlo model is scored twice: ordered by average votes and by average finishing
# position.
#
# Models: pl (Plackett-Luce Model), ranking (Gradient Boosted Ranking), classifier (Vote
# Classifier), simulation (Monte Carlo Simulation).
#
# Usage: ../venv/bin/python compareModels.py [--basic-stats] [--years 2012-2026] [--min-train 3]
#        [--models pl ranking classifier simulation] [--sims 5000]

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
for folder in ("Plackett-Luce Model", "Gradient Boosted Ranking", "Vote Classifier",
               "Monte Carlo Simulation"):
    sys.path.insert(0, os.path.join(HERE, folder))

import gameData  # noqa: E402
from brownlowModel import BrownlowModel  # noqa: E402
from classifierModel import GROUPS as CLASSIFIER_GROUPS, VoteClassifier  # noqa: E402
from rankingModel import GROUPS as RANKING_GROUPS, BoostedRanker  # noqa: E402
from simulationModel import SimulationModel  # noqa: E402

MODELS = ["pl", "ranking", "classifier", "simulation"]
NAMES = {"pl": "Plackett-Luce", "ranking": "Gradient boosted ranking",
         "classifier": "Vote classifier", "simulation": "Monte Carlo"}


def makeModel(name, basic_stats, sims):
    if name == "pl":
        return BrownlowModel(gameData.featureColumns(gameData.DEFAULT_GROUPS, basic_stats))
    if name == "ranking":
        return BoostedRanker(gameData.featureColumns(RANKING_GROUPS, basic_stats))
    if name == "classifier":
        return VoteClassifier(gameData.featureColumns(CLASSIFIER_GROUPS, basic_stats), basic_stats)
    return SimulationModel(gameData.featureColumns(gameData.DEFAULT_GROUPS, basic_stats),
                           n_sims=sims)


def leaderboardMetrics(tally, column, ascending=False):
    ranked = tally.sort_values(column, ascending=ascending, kind="stable").reset_index(drop=True)
    actual = ranked["actual_votes"].fillna(0)
    actual_rank = actual.rank(ascending=False, method="min")
    top = actual_rank <= 10
    return {
        "winner rank": int(np.flatnonzero(actual == actual.max())[0]) + 1,
        "top 10": int(top[:10].sum()),
        "top 10 order": spearmanr(actual_rank[top], ranked.index[top]).statistic,
        "top 10 MAE": (ranked.loc[top, "expected_votes"] - actual[top]).abs().mean(),
    }


def gameMetrics(predicted, tally):
    actual = predicted["brownlow_votes"].fillna(0)
    first = predicted.groupby("game")["expected_votes"].rank(ascending=False, method="first") == 1
    totals = tally.set_index("player")["actual_votes"].fillna(0)
    contenders = totals[totals.rank(ascending=False, method="min") <= 10].index
    rows = predicted[predicted["player"].isin(contenders)]
    votes = rows["brownlow_votes"].fillna(0).astype(int).to_numpy()
    chances = np.column_stack([1 - rows["p_polled"], rows["p1"], rows["p2"], rows["p3"]])
    return {
        "contender ll": np.log(np.clip(chances[np.arange(len(votes)), votes], 1e-12, 1)).sum(),
        "top pick": (first & (actual == 3)).sum() / predicted["game"].nunique(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare the Brownlow models by backtesting")
    parser.add_argument("--basic-stats", action="store_true",
                        help=f"Leave out the afl.com.au features and use seasons from "
                        f"{gameData.FIRST_BASIC_YEAR} (more test seasons)")
    parser.add_argument("--years", type=gameData.parseYears,
                        help=f"Seasons to use (default {gameData.FIRST_AFLCOM_YEAR}, or "
                        f"{gameData.FIRST_BASIC_YEAR} with --basic-stats, to the latest)")
    parser.add_argument("--min-train", type=int, default=3,
                        help="Seasons of training data before the first test season (default 3)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sims", type=int, default=5000,
                        help="Simulated seasons for the Monte Carlo model (default 5000)")
    args = parser.parse_args()

    first_year = gameData.FIRST_BASIC_YEAR if args.basic_stats else gameData.FIRST_AFLCOM_YEAR
    first, last = args.years or (first_year, gameData.latestSeason())
    print(f"Loading {first}-{last}{' (basic stats only)' if args.basic_stats else ''}...")
    games = gameData.loadGames((first, last))
    voted = gameData.votedGames(games)
    years = sorted(int(y) for y in voted["year"].unique())
    test_years = years[args.min_train:]

    rows = []
    for name in args.models:
        started = time.time()
        for year in test_years:
            model = makeModel(name, args.basic_stats, args.sims).fit(voted, [y for y in years if y < year])
            predicted = model.predict(voted, year)
            tally = model.tally(predicted)
            common = {"year": year, **gameMetrics(predicted, tally)}
            rows.append({"model": NAMES[name], **common,
                         **leaderboardMetrics(tally, "expected_votes")})
            if name == "simulation":
                rows.append({"model": "Monte Carlo, by position", **common,
                             **leaderboardMetrics(tally, "mean_rank", ascending=True)})
        print(f"  {NAMES[name]}: {time.time() - started:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    summary = table.groupby("model", sort=False).agg(
        winners_picked=("winner rank", lambda r: f"{(r == 1).sum()}/{len(r)}"),
        winner_rank=("winner rank", "mean"), winner_rank_median=("winner rank", "median"),
        top_10=("top 10", "mean"), top_10_order=("top 10 order", "mean"),
        top_10_MAE=("top 10 MAE", "mean"), contender_ll=("contender ll", "mean"),
        top_pick=("top pick", "mean"),
    )
    print(f"\nTest seasons {test_years[0]}-{test_years[-1]}, each predicted from earlier seasons only")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nWinner's predicted rank by season:")
    print(table.pivot_table(index="year", columns="model", values="winner rank", sort=False)
          .astype(int).to_string())
    print("\nTop 10 order by season:")
    print(table.pivot_table(index="year", columns="model", values="top 10 order", sort=False)
          .to_string(float_format=lambda v: f"{v:.2f}"))

    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_comparison{'_basic_stats' if args.basic_stats else ''}.csv")
    table.to_csv(path, index=False)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
