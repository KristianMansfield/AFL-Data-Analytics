#!/usr/bin/env python3

# Shared data preparation for the Brownlow prediction scripts: one row per player per home and
# away game, with the features the vote model uses.
#
# What it does: loads every player's games for a range of seasons, joins the basic stats in
# player_game_stats with the advanced afl.com.au stats in aflcom_player_game_stats, and adds:
#   - the game result from the player's side (won, margin)
#   - within-game ranks (who had the most disposals / rating points on the ground), because
#     votes are relative: 30 disposals is best on ground in one game and fifth in another
#   - scoring progression features from score_progressions. A small win probability model
#     (home side's chance of winning given the margin and the time left) is fitted on every
#     scoring event, and each goal or behind is worth the change in win probability it caused
#     for the scorer's side ("win probability added"). A goal that puts a side in front late is
#     worth far more than a goal at three quarter time in a 60 point win. Simpler counts sit
#     alongside it: last quarter goals, goals while the margin was within two goals, and who
#     kicked the score that put the winning side in front for good.
#   - season form: the player's average disposals, goals, contested possessions, clearances and
#     rating points over the rest of the season (every game but this one). Umpires vote for players
#     they've noticed, and a player having a big season gets noticed. Every stat is known by count
#     night, so this uses nothing the umpires' votes could have leaked into.
#   - reputation: the player's votes per game over the previous three seasons, shrunk towards
#     zero for players with few games. Off by default: umpire appeal (added by brownlowModel.py,
#     not here, because it needs a fitted model) measures the same thing better.
#
# Why it matters for the analysis: the Brownlow is decided game by game, by umpires ranking the
# players in that game. Every feature here is something an umpire could plausibly respond to in
# that game, and the model in rankedLogit.py learns how much each one matters from the votes
# actually given. The feature groups let backtest.py switch whole ideas on and off (e.g. does
# favouring the winning side help?) and measure the difference.

import os
import sys

import numpy as np
import pandas as pd

# The repository root, two folders up, holds dbConnection.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from dbConnection import createReaderConnection  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Seasons with every afl.com.au stat; the default range for training and backtesting
FIRST_AFLCOM_YEAR = 2012
# Seasons with every basic stat (goal assists and time on ground start in 2003), for the
# --basic-stats option: about twice as many seasons, without the afl.com.au features
FIRST_BASIC_YEAR = 2003

BASIC_STATS = [
    "kicks", "handballs", "marks", "goals", "behinds", "hitouts", "tackles", "rebound_50s",
    "inside_50s", "clearances", "clangers", "frees_for", "frees_against",
    "contested_possessions", "uncontested_possessions", "contested_marks", "marks_inside_50",
    "one_percenters", "bounces", "goal_assists",
]

ADVANCED_STATS = [
    "metres_gained", "score_involvements", "intercepts", "centre_clearances", "turnovers",
    "effective_disposals", "pressure_acts", "ground_ball_gets", "spoils", "score_launches",
    "hitouts_to_advantage", "contested_possession_rate",
]

# Groups of features that are switched on and off together. "baseline" is the two stats
# brownlowStats.py plots, as a yardstick for everything else.
FEATURE_GROUPS = {
    "baseline": ["disposals", "goals"],
    "basic": BASIC_STATS + ["tog"],
    "advanced": ADVANCED_STATS,
    "rating": ["rating_points"],
    "rank": ["disposals_rank_inv", "rating_rank_inv", "top_goalkicker"],
    "result": ["won", "margin_for"],
    "progression": ["scoring_wpa", "late_goals", "close_goals", "sealing_score"],
    "form": ["form_disposals", "form_goals", "form_contested_possessions", "form_clearances",
             "form_rating_points"],
    "appeal": ["umpire_appeal"],  # added by brownlowModel.py
    "reputation": ["prior_votes_per_game"],
}

# Chosen by backtest.py for predicting the winner and the top 10 in order
DEFAULT_GROUPS = ["basic", "advanced", "rating", "rank", "result", "progression", "form", "appeal"]

# Features that only exist from 2012, left out by --basic-stats
AFLCOM_FEATURES = ADVANCED_STATS + ["rating_points", "rating_rank_inv", "form_rating_points"]

FORM_STATS = ["disposals", "goals", "contested_possessions", "clearances", "rating_points"]

GAMES_SQL = f"""
SELECT
  s.year, r.sequence AS round_sequence, r.identifier AS round_name, g.id AS game,
  g.start_datetime, g.home_team, g.away_team, g.winning_team,
  g.final_home_score, g.final_away_score,
  pgs.player, pgs.team, p.first_name, p.last_name,
  COALESCE(t.aflcom_abbreviation, t.short_name, t.location) AS team_name,
  pgs.brownlow_votes, pgs.disposals,
  COALESCE(a.time_on_ground_percentage, pgs.percentage_played) AS tog,
  {", ".join("pgs." + c for c in BASIC_STATS)},
  {", ".join("a." + c for c in ADVANCED_STATS)},
  a.rating_points
FROM player_game_stats pgs
JOIN games g ON g.id = pgs.game
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
JOIN players p ON p.id = pgs.player
JOIN teams t ON t.id = pgs.team
LEFT JOIN aflcom_player_game_stats a ON a.player = pgs.player AND a.game = pgs.game
WHERE s.year BETWEEN ? AND ? AND NOT r.final
ORDER BY g.start_datetime, g.id, pgs.team, pgs.player
"""

PROGRESSION_SQL = """
SELECT
  sp.game, sp.player, sp.team, sp.quarter, sp.time, sp.result, sp.home_score, sp.away_score,
  g.home_team, g.winning_team, g.q1_seconds, g.q2_seconds, g.q3_seconds, g.q4_seconds
FROM score_progressions sp
JOIN games g ON g.id = sp.game
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
WHERE s.year BETWEEN ? AND ? AND NOT r.final
ORDER BY sp.game, sp.quarter, sp.time, sp.id
"""

# Votes and games in earlier seasons, for the reputation feature
PRIOR_VOTES_SQL = """
SELECT s.year, pgs.player, SUM(pgs.brownlow_votes) AS votes, COUNT(*) AS games
FROM player_game_stats pgs
JOIN games g ON g.id = pgs.game
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
WHERE s.year BETWEEN ? AND ? AND NOT r.final AND pgs.brownlow_votes IS NOT NULL
GROUP BY s.year, pgs.player
"""

REPUTATION_SEASONS = 3
REPUTATION_SHRINKAGE_GAMES = 10
CLOSE_MARGIN = 12  # two goals


def parseYears(text):
    """"2025" -> (2025, 2025); "2012-2025" -> (2012, 2025)."""
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def _query(db, sql, params):
    cursor = db.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def _numeric(frame, skip):
    for column in frame.columns:
        if column not in skip:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame


def latestSeason():
    db = createReaderConnection()
    year = db.fetchone("SELECT MAX(s.year) FROM games g JOIN rounds r ON r.id = g.round "
                       "JOIN seasons s ON s.id = r.season")[0]
    db.close()
    return int(year)


def loadGames(years):
    """One row per player per home and away game in the seasons, with every feature."""
    db = createReaderConnection()
    games = _query(db, GAMES_SQL, years)
    progressions = _query(db, PROGRESSION_SQL, years)
    prior = _query(db, PRIOR_VOTES_SQL, (years[0] - REPUTATION_SEASONS, years[1] - 1))
    db.close()
    if games.empty:
        sys.exit(f"No home and away games found for {years[0]}-{years[1]}")

    games = _numeric(games, {"round_name", "first_name", "last_name", "team_name",
                             "start_datetime"})
    progressions = _numeric(progressions, {"result"})
    prior = _numeric(prior, set())

    games["name"] = games["first_name"] + " " + games["last_name"]
    addResultFeatures(games)
    addRankFeatures(games)
    addProgressionFeatures(games, progressions)
    addFormFeatures(games)
    addReputationFeatures(games, prior)
    return games


def addResultFeatures(games):
    home = games["team"] == games["home_team"]
    score_for = np.where(home, games["final_home_score"], games["final_away_score"])
    score_against = np.where(home, games["final_away_score"], games["final_home_score"])
    games["won"] = np.where(games["winning_team"].isna(), 0.5,
                            (games["winning_team"] == games["team"]).astype(float))
    # Margin in goals, capped at 10 goals: a 150 point win isn't three times a 50 point one
    games["margin_for"] = np.clip((score_for - score_against) / 6, -10, 10)


def addRankFeatures(games):
    """1 / rank within the game (both sides): 1 for best on ground by that stat, 0.5 for second,
    ... Votes go to the best few players on the ground, so being first matters more than the
    raw number."""
    by_game = games.groupby("game")
    games["disposals_rank_inv"] = 1 / by_game["disposals"].rank(ascending=False, method="min")
    games["rating_rank_inv"] = 1 / by_game["rating_points"].rank(ascending=False, method="min")
    most_goals = by_game["goals"].transform("max")
    games["top_goalkicker"] = ((games["goals"] == most_goals) & (most_goals >= 3)).astype(float)


def addFormFeatures(games):
    """The player's average of each FORM_STATS stat over their other games that season."""
    for column in FORM_STATS:
        season = games.groupby(["year", "player"])[column]
        total, count = season.transform("sum"), season.transform("count")
        games[f"form_{column}"] = ((total - games[column]) / (count - 1)).where(count > 1)


def fitWinProbability(events):
    """Logistic model of the home side's chance of winning given the margin and the share of
    the game left: P = sigmoid(a + b * margin / sqrt(time left + c)). The margin is scaled by the
    square root of the time left because the final margin wanders like a random walk: a 12 point
    lead with a quarter left is about as safe as a 24 point lead with the whole game left."""
    from sklearn.linear_model import LogisticRegression

    decided = events[events["winning_team"].notna()]
    x = (decided["margin_after"] / np.sqrt(decided["left_after"] + 0.02)).to_numpy()[:, None]
    y = (decided["winning_team"] == decided["home_team"]).to_numpy()
    model = LogisticRegression(C=1e6).fit(x, y)
    a, b = model.intercept_[0], model.coef_[0, 0]
    return lambda margin, left: 1 / (1 + np.exp(-(a + b * margin / np.sqrt(left + 0.02))))


def addProgressionFeatures(games, events):
    columns = ["scoring_wpa", "late_goals", "close_goals", "sealing_score"]
    for column in columns:
        games[column] = 0.0
    if events.empty:
        return

    events = events.copy()
    points = events["result"].map({"goal": 6, "behind": 1, "rushed behind": 1}).fillna(0)
    home = events["team"] == events["home_team"]
    events["margin_after"] = events["home_score"] - events["away_score"]
    events["margin_before"] = events["margin_after"] - np.where(home, points, -points)

    # Share of the game left after the event; extra time counts as no time left
    quarter_lengths = events[["q1_seconds", "q2_seconds", "q3_seconds", "q4_seconds"]]
    quarter_lengths = quarter_lengths.fillna(quarter_lengths.stack().median()).to_numpy()
    total = quarter_lengths.sum(axis=1)
    quarter = events["quarter"].astype(int).to_numpy()
    starts = np.cumsum(quarter_lengths, axis=1) - quarter_lengths
    before = starts[np.arange(len(events)), np.clip(quarter, 1, 4) - 1]
    elapsed = np.where(quarter <= 4, before + events["time"].to_numpy(), total)
    events["left_after"] = np.clip((total - elapsed) / total, 0, 1)

    win_probability = fitWinProbability(events)
    swing = (win_probability(events["margin_after"], events["left_after"])
             - win_probability(events["margin_before"], events["left_after"]))
    events["wpa"] = np.where(home, swing, -swing)

    # The score that put the eventual winner in front for the last time
    margin_for_winner_after = np.where(events["winning_team"] == events["home_team"],
                                       events["margin_after"], -events["margin_after"])
    margin_for_winner_before = np.where(events["winning_team"] == events["home_team"],
                                        events["margin_before"], -events["margin_before"])
    took_lead = ((events["team"] == events["winning_team"]) & (margin_for_winner_before <= 0)
                 & (margin_for_winner_after > 0))
    last_lead = events[took_lead].groupby("game").tail(1).index
    events["sealing_score"] = 0.0
    events.loc[last_lead, "sealing_score"] = 1.0

    goal = events["result"] == "goal"
    events["late_goal"] = (goal & (events["quarter"] >= 4)).astype(float)
    events["close_goal"] = (goal & (events["margin_before"].abs() <= CLOSE_MARGIN)).astype(float)

    credited = events[events["player"].notna() & events["result"].isin(["goal", "behind"])]
    per_player = credited.groupby(["game", "player"]).agg(
        scoring_wpa=("wpa", "sum"), late_goals=("late_goal", "sum"),
        close_goals=("close_goal", "sum"), sealing_score=("sealing_score", "max"),
    )
    joined = games[["game", "player"]].join(per_player, on=["game", "player"])
    for column in columns:
        games[column] = joined[column].fillna(0.0).to_numpy()


def addReputationFeatures(games, prior):
    """Votes per game over the previous REPUTATION_SEASONS seasons, as votes / (games + k) so a
    player with a handful of games and one lucky vote isn't treated as a star."""
    games["prior_votes_per_game"] = 0.0
    if prior.empty:
        return
    rows = []
    for year in games["year"].unique():
        window = prior[(prior["year"] < year) & (prior["year"] >= year - REPUTATION_SEASONS)]
        totals = window.groupby("player")[["votes", "games"]].sum()
        totals["year"] = year
        rows.append(totals.reset_index())
    totals = pd.concat(rows)
    totals["prior_votes_per_game"] = totals["votes"] / (totals["games"] + REPUTATION_SHRINKAGE_GAMES)
    joined = games[["year", "player"]].merge(totals, on=["year", "player"], how="left")
    games["prior_votes_per_game"] = joined["prior_votes_per_game"].fillna(0.0).to_numpy()


def featureColumns(groups, basic_stats=False):
    """The features in the groups; with basic_stats, without the ones afl.com.au only has from
    2012."""
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        sys.exit(f"Unknown feature groups: {', '.join(unknown)} "
                 f"(known: {', '.join(FEATURE_GROUPS)})")
    columns = []
    for group in groups:
        columns += [c for c in FEATURE_GROUPS[group] if c not in columns
                    and not (basic_stats and c in AFLCOM_FEATURES)]
    return columns


def votedGames(games):
    """Only the games with exactly one 3, 2 and 1 vote recorded, i.e. the games the model can
    learn from or be scored against."""
    votes = games["brownlow_votes"].fillna(0)
    pattern = votes.groupby(games["game"]).agg(lambda v: tuple(sorted(v[v > 0], reverse=True)))
    complete = pattern[pattern.apply(lambda p: p == (3, 2, 1))].index
    return games[games["game"].isin(complete)]
