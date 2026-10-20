#!/usr/bin/env python3

# Shared command line options, output files and printed leaderboard for every model's predict
# script, so the models are run the same way and their outputs can be compared like for like.
#
# Every model hands over the same two tables:
#   predicted  one row per player per game: p3, p2, p1 (chance of 3, 2 and 1 votes),
#              expected_votes, votes_321 and the actual brownlow_votes
#   tally      one row per player (seasonTally.seasonTally or the simulation's own): rank,
#              expected_votes, votes_10pct, votes_90pct, votes_321, p_win, actual_votes, and
#              optionally p_top10 and mean_rank

import os

import gameData
from seasonTally import roundByRound


def addCommonArguments(parser):
    parser.add_argument("--year", type=int, help="Season to predict (default: the latest)")
    parser.add_argument(
        "--train-years", type=gameData.parseYears,
        help=f"Seasons to learn from (default: {gameData.FIRST_AFLCOM_YEAR}, or "
        f"{gameData.FIRST_BASIC_YEAR} with --basic-stats, to the season before --year)",
    )
    parser.add_argument(
        "--basic-stats", action="store_true",
        help=f"Leave out the afl.com.au features (only recorded from {gameData.FIRST_AFLCOM_YEAR}) "
        f"and train from {gameData.FIRST_BASIC_YEAR} by default",
    )
    parser.add_argument("--top", type=int, default=25, help="Leaderboard rows to print")
    return parser


def loadForArguments(args):
    """The target season, the training seasons and the games covering both."""
    year = args.year or gameData.latestSeason()
    first_year = gameData.FIRST_BASIC_YEAR if args.basic_stats else gameData.FIRST_AFLCOM_YEAR
    train_years = args.train_years or (first_year, year - 1)
    if train_years[0] <= year <= train_years[1]:
        print(f"Warning: {year} is in the training seasons, so the comparison with the actual "
              "votes is not a fair test")
    first, last = min(train_years[0], year), max(train_years[1], year)
    print(f"Loading {first}-{last}...")
    games = gameData.loadGames((first, last))
    train = gameData.votedGames(games[games["year"].between(*train_years)])
    print(f"Training on {train['game'].nunique()} games ({train_years[0]}-{train_years[1]})")
    return year, range(train_years[0], train_years[1] + 1), games


def writeOutputs(predicted, tally, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tally.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False, float_format="%.4f")
    columns = ["year", "round_name", "game", "player", "name", "team_name", "impact", "p3", "p2",
               "p1", "expected_votes", "votes_321", "brownlow_votes"]
    predicted[[c for c in columns if c in predicted.columns]].sort_values(
        ["game", "expected_votes"], ascending=[True, False]
    ).to_csv(os.path.join(out_dir, "game_predictions.csv"), index=False, float_format="%.4f")
    roundByRound(predicted, tally).to_csv(os.path.join(out_dir, "round_by_round.csv"))


def printLeaderboard(predicted, tally, year, top):
    known = tally["actual_votes"].notna().any()
    extra = "p_top10" in tally.columns
    print(f"\n{year} Brownlow prediction ({predicted['game'].nunique()} games)")
    print(f"{'#':>3} {'Player':<26} {'Team':<6} {'Gms':>3} {'Exp':>5} {'80% range':>9} "
          f"{'3-2-1':>5} {'P(win)':>7}" + (f" {'P(top10)':>8} {'Avg pos':>7}" if extra else "")
          + (f" {'Actual':>6}" if known else ""))
    for _, row in tally.head(top).iterrows():
        line = (f"{row['rank']:>3} {row['name']:<26} {row['team']:<6} {row['games']:>3} "
                f"{row['expected_votes']:>5.1f} {row['votes_10pct']:>4}-{row['votes_90pct']:<4} "
                f"{row['votes_321']:>5} {row['p_win']:>7.1%}")
        if extra:
            line += f" {row['p_top10']:>8.1%} {row['mean_rank']:>7.1f}"
        if known:
            line += f" {row['actual_votes']:>6.0f}"
        print(line)

    if known:
        actual = tally.sort_values("actual_votes", ascending=False)
        best = actual["actual_votes"].iloc[0]
        for _, row in actual[actual["actual_votes"] == best].iterrows():
            print(f"\nActual winner: {row['name']} ({best:.0f} votes), predicted rank "
                  f"{row['rank']} with {row['expected_votes']:.1f} expected votes and a "
                  f"{row['p_win']:.1%} chance of winning")
        hits = (predicted["votes_321"] == predicted["brownlow_votes"].fillna(0)) & (
            predicted["votes_321"] == 3)
        print(f"Predicted 3 vote player got the 3 votes in {hits.sum()} of "
              f"{predicted['game'].nunique()} games")
