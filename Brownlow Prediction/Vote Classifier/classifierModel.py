#!/usr/bin/env python3

# The vote classifier: each player-game is classified as 0, 1, 2 or 3 votes on its own, then the
# chances are adjusted so every game gives out exactly one 3, one 2 and one 1.
#
# What it does: a gradient-boosted classifier (scikit-learn's HistGradientBoostingClassifier)
# learns P(0), P(1), P(2), P(3 votes) for a player from their stats, as if every player-game were
# separate. Because it can't see the other players in the game, it's also given each key stat as a
# z-score within the game (how far above the game's average the player was). Its raw chances
# don't add up across a game: the P(3) column might total 1.4 in a game with three standout
# players and 0.6 in a game with none. So for each game the grid of chances (players x 0/1/2/3
# votes) is rescaled with Sinkhorn balancing: alternately scaling each player's row to add to 1,
# and each vote column to add to 1 (and the 0 column to players - 3), until both hold. That keeps
# the classifier's view of who is likelier than whom, while making the game consistent: exactly
# 6 votes, each vote given once.
#
# Why it matters for the analysis: this is the simplest way to model votes, and a common one, so
# it shows how much the ranking models gain by treating each game as a contest between its
# players. It can't express that one player's 3 votes take them away from everyone else, only
# approximate it afterwards.
#
# Season tallies treat games as independent (seasonTally.py with no player appeal spread, as the
# classifier has no impact score to shift).

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Plackett-Luce Model"))
import gameData  # noqa: E402
from rankedLogit import addVoteSummaries  # noqa: E402
from seasonTally import seasonTally  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# As the gradient-boosted ranker: everything but umpire appeal, with reputation instead
GROUPS = ["basic", "advanced", "rating", "rank", "result", "progression", "form", "reputation"]

# Stats also given as z-scores within the game, so the classifier can see the competition
IN_GAME_STATS = ["disposals", "goals", "contested_possessions", "clearances", "marks", "tackles",
                 "rating_points", "metres_gained", "score_involvements"]

SINKHORN_ITERATIONS = 50


def addInGameScores(frame, basic_stats=False):
    frame = frame.copy()
    columns = []
    for stat in IN_GAME_STATS:
        if basic_stats and stat in gameData.AFLCOM_FEATURES:
            continue
        by_game = frame.groupby("game")[stat]
        spread = by_game.transform("std")
        frame[f"in_game_{stat}"] = (frame[stat] - by_game.transform("mean")) / spread.where(spread > 0)
        columns.append(f"in_game_{stat}")
    return frame, columns


def balanceGame(probabilities):
    """Sinkhorn balancing of one game's (players, 4) grid of P(0), P(1), P(2), P(3): rows add
    to 1, the vote columns to 1 each and the 0 column to players - 3."""
    p = np.clip(probabilities, 1e-12, None)
    targets = np.array([len(p) - 3, 1.0, 1.0, 1.0])
    for _ in range(SINKHORN_ITERATIONS):
        p *= (targets / p.sum(axis=0))[None, :]
        p /= p.sum(axis=1, keepdims=True)
    return p


class VoteClassifier:
    def __init__(self, features, basic_stats=False, learning_rate=0.05, max_iter=500,
                 max_leaf_nodes=31, min_samples_leaf=100, seed=0):
        self.features = list(features)
        self.basic_stats = basic_stats
        self.params = dict(learning_rate=learning_rate, max_iter=max_iter,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=30, random_state=seed)

    def fit(self, games, train_years):
        voted = gameData.votedGames(games)
        train, in_game = addInGameScores(voted[voted["year"].isin(set(train_years))],
                                         self.basic_stats)
        self.columns = self.features + in_game
        self.model = HistGradientBoostingClassifier(**self.params).fit(
            train[self.columns], train["brownlow_votes"].fillna(0).astype(int))
        return self

    def predict(self, games, year):
        rows, _ = addInGameScores(games[games["year"] == year], self.basic_stats)
        raw = self.model.predict_proba(rows[self.columns])
        raw = raw[:, np.argsort(self.model.classes_)]  # columns 0, 1, 2, 3
        balanced = np.zeros_like(raw)
        for positions in rows.groupby("game").indices.values():
            balanced[positions] = balanceGame(raw[positions])
        rows = rows.copy()
        rows["p1"], rows["p2"], rows["p3"] = balanced[:, 1], balanced[:, 2], balanced[:, 3]
        return addVoteSummaries(rows)

    def tally(self, predicted):
        return seasonTally(predicted, player_sd=0.0)

    def featureImportance(self, games, year, repeats=3):
        """Permutation importance on a season: how much the log loss worsens when a feature is
        shuffled (scikit-learn's histogram trees don't record split importances)."""
        from sklearn.inspection import permutation_importance

        rows, _ = addInGameScores(gameData.votedGames(games[games["year"] == year]),
                                  self.basic_stats)
        result = permutation_importance(
            self.model, rows[self.columns], rows["brownlow_votes"].fillna(0).astype(int),
            scoring="neg_log_loss", n_repeats=repeats, random_state=0)
        importance = pd.DataFrame({"feature": self.columns,
                                   "log_loss_increase": result.importances_mean})
        return importance.sort_values("log_loss_increase", ascending=False)
