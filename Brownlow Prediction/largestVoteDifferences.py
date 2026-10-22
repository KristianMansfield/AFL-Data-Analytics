#!/usr/bin/env python3

# The games where the Brownlow votes given differed most from the votes expected.
#
# What it does: predicts every home and away game in the chosen seasons with the Plackett-Luce
# model (Plackett-Luce Model/brownlowModel.py), each season from a model trained only on the
# seasons before it, so a game's own votes never shape its expected votes. Then for each game:
#   votes misplaced  the sum over every player in the game of |actual votes - expected votes|,
#                    halved: how many of the game's 6 votes went somewhere other than expected
#                    (0 = exactly as expected, 6 = every vote went to players expected to get none)
#   surprise         how unlikely the model thought the 3-2-1 actually given was, as a
#                    "1 in N" chance of that exact order
# and, for each player, the gap between their votes and their expected votes.
#
# Why it matters for the analysis: these are the games the model understands least. Looking at
# them shows what the stats miss (a player who dominated in ways the stats don't record, a
# reputation vote, a tagging job) and where umpires saw the game differently from the numbers.
#
# Outputs (in output/): largest_vote_differences_games.csv (every game) and
# largest_vote_differences_players.csv (every player-game).
#
# Usage: ../venv/bin/python largestVoteDifferences.py [--years 2015-2026] [--top 15]
#        [--sort misplaced|surprise] [--players] [--basic-stats]

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Plackett-Luce Model"))
import gameData  # noqa: E402
from brownlowModel import BrownlowModel  # noqa: E402

OUTPUT_DIR = os.path.join(HERE, "output")
TRAINING_SEASONS_MIN = 3


def orderProbability(game):
    """Log probability the Plackett-Luce model gave the 3-2-1 actually given in one game."""
    impact = game["impact"].to_numpy()
    votes = game["brownlow_votes"].fillna(0).to_numpy()
    remaining = np.ones(len(impact), dtype=bool)
    total = 0.0
    for vote in (3, 2, 1):
        chosen = np.flatnonzero(votes == vote)[0]
        total += impact[chosen] - logsumexp(impact[remaining])
        remaining[chosen] = False
    return total


def gameSummaries(predicted):
    rows = []
    for game, players in predicted.groupby("game"):
        first = players.iloc[0]
        teams = players.groupby("team")["team_name"].first()
        home_name, away_name = teams[first["home_team"]], teams[first["away_team"]]
        rows.append({
            "game": game, "year": int(first["year"]), "round": first["round_name"],
            "home": home_name, "away": away_name,
            "home_score": int(first["final_home_score"]), "away_score": int(first["final_away_score"]),
            "votes_misplaced": players["difference"].abs().sum() / 2,
            "log_probability": orderProbability(players),
            "votes_3": players.loc[players["brownlow_votes"] == 3, "name"].iloc[0],
            "expected_3": players.sort_values("expected_votes").iloc[-1]["name"],
        })
    summary = pd.DataFrame(rows)
    summary["one_in"] = np.exp(-summary["log_probability"])
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Games where the Brownlow votes differed most from the expected votes")
    parser.add_argument("--years", type=gameData.parseYears,
                        help="Seasons to look at (default: 2015 to the latest, or 2006 to the "
                        "latest with --basic-stats)")
    parser.add_argument("--basic-stats", action="store_true",
                        help="Leave out the afl.com.au features, so seasons from 2006 can be "
                        "looked at (training from 2003)")
    parser.add_argument("--top", type=int, default=15, help="Games (or players) to list (default 15)")
    parser.add_argument("--sort", choices=["misplaced", "surprise"], default="misplaced",
                        help="Rank games by votes misplaced (default) or by how surprising the "
                        "3-2-1 was")
    parser.add_argument("--players", action="store_true",
                        help="Also list the individual player-games with the biggest gaps")
    args = parser.parse_args()

    first_data = gameData.FIRST_BASIC_YEAR if args.basic_stats else gameData.FIRST_AFLCOM_YEAR
    first, last = args.years or (first_data + TRAINING_SEASONS_MIN, gameData.latestSeason())
    if first - first_data < 1:
        sys.exit(f"The first season needs at least one season before it to train on "
                 f"(data starts in {first_data})")
    print(f"Loading {first_data}-{last}...")
    games = gameData.loadGames((first_data, last))
    voted = gameData.votedGames(games)
    features = gameData.featureColumns(gameData.DEFAULT_GROUPS, args.basic_stats)

    predictions = []
    for year in range(first, last + 1):
        if year not in set(voted["year"]):
            continue
        model = BrownlowModel(features).fit(voted, range(first_data, year))
        predictions.append(model.predict(voted, year))
        print(f"  {year} predicted from {first_data}-{year - 1}", flush=True)
    predicted = pd.concat(predictions)
    predicted["votes"] = predicted["brownlow_votes"].fillna(0)
    predicted["difference"] = predicted["votes"] - predicted["expected_votes"]

    summary = gameSummaries(predicted)
    key = "votes_misplaced" if args.sort == "misplaced" else "log_probability"
    summary = summary.sort_values(key, ascending=args.sort == "surprise")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "largest_vote_differences_games.csv"), index=False,
                   float_format="%.3f")
    predicted[["year", "round_name", "game", "name", "team_name", "votes",
               "expected_votes", "difference", "p3", "p2", "p1"]].sort_values(
        "difference", key=np.abs, ascending=False
    ).to_csv(os.path.join(OUTPUT_DIR, "largest_vote_differences_players.csv"), index=False,
             float_format="%.3f")

    print(f"\n{len(summary)} games {first}-{last}. Votes misplaced: median "
          f"{summary['votes_misplaced'].median():.1f} of 6, a perfectly predicted game would be 0")
    print(f"\nThe {args.top} games with the {'most votes misplaced' if args.sort == 'misplaced' else 'most surprising 3-2-1'}:")
    for _, game in summary.head(args.top).iterrows():
        print(f"\n{game['year']} round {game['round']}: {game['home']} {game['home_score']} v "
              f"{game['away']} {game['away_score']}  |  {game['votes_misplaced']:.1f} of 6 votes "
              f"misplaced, the 3-2-1 given was a 1 in {game['one_in']:,.0f} chance")
        players = predicted[predicted["game"] == game["game"]]
        shown = players[(players["votes"] > 0) | (players["expected_votes"] >= 0.5)]
        shown = shown.sort_values(["votes", "expected_votes"], ascending=False)
        for _, player in shown.iterrows():
            print(f"    {player['name']:<26} {player['team_name']:<5} votes "
                  f"{player['votes']:>3.0f}   expected {player['expected_votes']:>4.2f}   ({player['difference']:+.2f})")

    if args.players:
        print(f"\nThe {args.top} player-games furthest from their expected votes:")
        biggest = predicted.sort_values("difference", key=np.abs, ascending=False)
        teams = summary.set_index("game")
        for _, player in biggest.head(args.top).iterrows():
            game = teams.loc[player["game"]]
            print(f"  {player['year']:.0f} round {player['round_name']:<4} {player['name']:<26} "
                  f"{player['team_name']:<5} votes {player['votes']:>3.0f}  "
                  f"expected {player['expected_votes']:>4.2f}  ({player['difference']:+.2f})  "
                  f"{game['home']} v {game['away']}")
    print(f"\nWritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
