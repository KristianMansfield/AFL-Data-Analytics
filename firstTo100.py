#!/usr/bin/env python3

# "First to 100 points wins": how often does the first team to reach 100 points win?
#
# What it does: walks through every game's scoring progression (score_progressions, 2001
# onwards) to find the first score that took either side to 100 or more, and checks whether that
# side won. Only games whose progression ends on the game's final score are used, so a game with
# missing scoring events can't give the wrong answer. It reports:
#   - every game where a team reached 100: how often the first to 100 won
#   - the games where both teams reached 100 (the real test of the saying, since in most games
#     only the winner gets there): how often the first to 100 won, how far into the game it
#     happened, and every game where the first to 100 lost
#
# Outputs (in output/): first_to_100_games.csv, every game where a team reached 100.
#
# Usage: venv/bin/python firstTo100.py [--first-year 2001] [--include-finals] [--list 0]

import argparse
import os

import numpy as np
import pandas as pd

from dbConnection import createReaderConnection

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

GAMES_SQL = """
SELECT g.id AS game, s.year, r.identifier AS round, r.final, g.home_team, g.away_team,
       g.winning_team, g.final_home_score, g.final_away_score,
       COALESCE(ht.aflcom_abbreviation, ht.short_name) AS home_name,
       COALESCE(at.aflcom_abbreviation, at.short_name) AS away_name
FROM games g
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
JOIN teams ht ON ht.id = g.home_team
JOIN teams at ON at.id = g.away_team
WHERE s.year >= ?
"""

PROGRESSIONS_SQL = """
SELECT sp.game, sp.quarter, sp.time, sp.home_score, sp.away_score
FROM score_progressions sp
JOIN games g ON g.id = sp.game
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
WHERE s.year >= ?
ORDER BY sp.game, sp.quarter, sp.time, sp.id
"""


def query(db, sql, params):
    cursor = db.execute(sql, params)
    return pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])


def main():
    parser = argparse.ArgumentParser(description="How often does the first team to 100 points win?")
    parser.add_argument("--first-year", type=int, default=2001,
                        help="First season (default 2001, when scoring progressions start)")
    parser.add_argument("--include-finals", action="store_true",
                        help="Include finals (default: home and away games only)")
    parser.add_argument("--list", type=int, default=0,
                        help="Also list this many of the both-to-100 games where the first to 100 "
                        "won, most recent first (the losses are always listed)")
    args = parser.parse_args()

    db = createReaderConnection()
    games = query(db, GAMES_SQL, (args.first_year,))
    events = query(db, PROGRESSIONS_SQL, (args.first_year,))
    db.close()
    if not args.include_finals:
        games = games[~games["final"].astype(bool)]

    # Only games whose progression finishes on the final score
    last = events.groupby("game").tail(1).set_index("game")
    games = games.join(last[["home_score", "away_score"]], on="game")
    complete = (games["home_score"] == games["final_home_score"]) & (
        games["away_score"] == games["final_away_score"])
    print(f"{len(games)} games from {args.first_year}"
          f"{' including finals' if args.include_finals else ' (home and away)'}; "
          f"{complete.sum()} with a complete scoring progression")
    games = games[complete].drop(columns=["home_score", "away_score"])

    # The first score to take either side to 100
    events = events[events["game"].isin(games["game"])]
    reached = events[(events["home_score"] >= 100) | (events["away_score"] >= 100)]
    first = reached.groupby("game").head(1).set_index("game")
    games = games.join(first, on="game", how="inner")
    games["first_to_100"] = np.where(games["home_score"] >= 100, games["home_team"], games["away_team"])
    games["first_name"] = np.where(games["home_score"] >= 100, games["home_name"], games["away_name"])
    games["result"] = np.select(
        [games["winning_team"].isna(), games["winning_team"] == games["first_to_100"]],
        ["draw", "won"], "lost")
    games["both_to_100"] = (games["final_home_score"] >= 100) & (games["final_away_score"] >= 100)
    # Both sides' scores at the moment the first side reached 100
    games["score_at_100"] = np.where(games["home_score"] >= 100, games["home_score"], games["away_score"])
    games["opponent_at_100"] = np.where(games["home_score"] >= 100, games["away_score"], games["home_score"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    games.drop(columns=["final", "home_score", "away_score"]).to_csv(
        os.path.join(OUTPUT_DIR, "first_to_100_games.csv"), index=False)

    def summary(subset, label):
        counts = subset["result"].value_counts()
        decided = counts.get("won", 0) + counts.get("lost", 0)
        print(f"\n{label}: {len(subset)} games")
        print(f"  first to 100 won {counts.get('won', 0)}, lost {counts.get('lost', 0)}, "
              f"drew {counts.get('draw', 0)}: won {counts.get('won', 0) / decided:.1%} of decided games")

    summary(games, "A team reached 100")
    both = games[games["both_to_100"]]
    summary(both, "Both teams reached 100")

    print("\n  When the first side got to 100 (both-to-100 games):")
    quarters = both.groupby(both["quarter"].clip(upper=4))
    for quarter, subset in quarters:
        decided = subset[subset["result"] != "draw"]
        print(f"    quarter {quarter}{' (or extra time)' if quarter == 4 else ''}: {len(subset)} games, "
              f"first to 100 won {(decided['result'] == 'won').mean():.0%}")

    print("\n  By how far the first side led when it got to 100 (both-to-100 games):")
    lead = both["score_at_100"] - both["opponent_at_100"]
    for low, high, label in ((-np.inf, 6, "under a goal"), (6, 12, "1-2 goals"),
                             (12, 24, "2-4 goals"), (24, np.inf, "4+ goals")):
        subset = both[(lead > low) & (lead <= high) if np.isfinite(high) else (lead > low)]
        decided = subset[subset["result"] != "draw"]
        if len(decided):
            print(f"    led by {label:<12} {len(subset):>4} games, first to 100 won "
                  f"{(decided['result'] == 'won').mean():.0%}")

    print("\n  By era (both-to-100 games):")
    for start in range(args.first_year - args.first_year % 5, int(both["year"].max()) + 1, 5):
        subset = both[both["year"].between(start, start + 4)]
        decided = subset[subset["result"] != "draw"]
        if len(decided):
            print(f"    {start}-{start + 4}: {len(subset):>3} games, first to 100 won "
                  f"{(decided['result'] == 'won').mean():.0%}")

    def describe(row):
        winner_first = row["first_name"]
        return (f"  {row['year']} {'final' if row['final'] else 'round'} {row['round']:<4} "
                f"{row['home_name']} {row['final_home_score']} v {row['away_name']} "
                f"{row['final_away_score']}  -  {winner_first} reached 100 first "
                f"(Q{row['quarter']}, {row['time'] // 60}:{row['time'] % 60:02d}, opponent on "
                f"{row['opponent_at_100']})")

    exceptions = both[both["result"] != "won"].sort_values(["year", "game"], ascending=False)
    print(f"\nBoth reached 100 and the first to 100 didn't win ({len(exceptions)}):")
    for _, row in exceptions.iterrows():
        print(describe(row) + ("  [draw]" if row["result"] == "draw" else ""))

    if args.list:
        held = both[both["result"] == "won"].sort_values(["year", "game"], ascending=False)
        print(f"\nBoth reached 100 and the first to 100 won (most recent {args.list}):")
        for _, row in held.head(args.list).iterrows():
            print(describe(row))
    print(f"\nEvery game where a team reached 100 written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
