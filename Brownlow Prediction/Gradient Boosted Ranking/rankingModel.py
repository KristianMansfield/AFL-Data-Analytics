#!/usr/bin/env python3

# The gradient-boosted ranking model: decision trees learn each player's impact in a game, ranked
# against the other players in the same game.
#
# What it does: starts every player at an impact of 0 and adds one small decision tree at a time,
# each nudging the impacts in the direction that makes the votes actually given more likely. The
# loss is listwise, the same Plackett-Luce likelihood as the Plackett-Luce model (also called
# ListMLE): the 3 votes drawn from everyone in the game in proportion to exp(impact), the 2 votes
# from the rest, then the 1 vote. Each tree is fitted to the loss's gradient divided by its
# curvature (a Newton step, as XGBoost and LightGBM do), on a random half of the games to reduce
# overfitting. The number of trees is chosen by holding out the last training season, stopping
# once more trees stop improving it, then refitting on every training season.
#
# Because the impacts are on the Plackett-Luce scale, they turn into exact chances of 3, 2 and 1
# votes and a season tally the same way as in the Plackett-Luce model (rankedLogit.py,
# seasonTally.py), so the two are compared like for like.
#
# Why it matters for the analysis: the Plackett-Luce model's impact is a weighted sum, so a stat
# always counts the same. Trees can learn that a stat counts more in some situations than others
# (e.g. goals matter more for forwards in a win, hitouts only for a ruck who also wins the ball)
# and that it stops counting after a point. If those interactions matter to the umpires, this
# model should beat the weighted sum.
#
# Using a dedicated library (LightGBM's LambdaRank) would be faster, but needs the OpenMP runtime
# installed system-wide on macOS; this uses scikit-learn's trees, which are already installed.

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Plackett-Luce Model"))
import gameData  # noqa: E402
from rankedLogit import addVoteProbabilities  # noqa: E402
from seasonTally import DEFAULT_PLAYER_SD, seasonTally  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Every feature except umpire appeal (which needs a Plackett-Luce model of its own); reputation
# stands in for it, as trees can pick out the useful part themselves
GROUPS = ["basic", "advanced", "rating", "rank", "result", "progression", "form", "reputation"]


class GameStructure:
    """The players in each game laid out as a (games, players) grid, for the listwise loss."""

    def __init__(self, frame):
        self.game_ids, self.game_index = np.unique(frame["game"].to_numpy(), return_inverse=True)
        self.slot = frame.groupby("game").cumcount().to_numpy()
        shape = (len(self.game_ids), self.slot.max() + 1)
        self.mask = np.zeros(shape, dtype=bool)
        self.mask[self.game_index, self.slot] = True
        votes = frame["brownlow_votes"].fillna(0).to_numpy()
        rows = np.arange(shape[0])
        self.chosen = []
        for v in (3, 2, 1):
            chosen = np.zeros(shape[0], dtype=int)
            chosen[self.game_index[votes == v]] = self.slot[votes == v]
            self.chosen.append(chosen)
        # Players still in the draw for the 3, 2 and 1 votes
        self.masks = [self.mask.copy()]
        for chosen in self.chosen[:2]:
            next_mask = self.masks[-1].copy()
            next_mask[rows, chosen] = False
            self.masks.append(next_mask)

    def _grid(self, u):
        grid = np.zeros(self.mask.shape)
        grid[self.game_index, self.slot] = u
        return grid

    def loss(self, u):
        """Mean negative log probability per game of the votes actually given."""
        grid = self._grid(u)
        rows = np.arange(len(grid))
        total = 0.0
        for m, chosen in zip(self.masks, self.chosen):
            total += (logsumexp(np.where(m, grid, -np.inf), axis=1) - grid[rows, chosen]).sum()
        return total / len(grid)

    def gradientAndHessian(self, u):
        """Per player: the loss's gradient and (diagonal) curvature with respect to impact."""
        grid = self._grid(u)
        rows = np.arange(len(grid))
        gradient = np.zeros(grid.shape)
        hessian = np.zeros(grid.shape)
        for m, chosen in zip(self.masks, self.chosen):
            masked = np.where(m, grid, -np.inf)
            p = np.exp(masked - logsumexp(masked, axis=1)[:, None])
            gradient += p
            gradient[rows, chosen] -= 1
            hessian += p * (1 - p)
        at = (self.game_index, self.slot)
        return gradient[at], np.maximum(hessian[at], 1e-6)


class BoostedRanker:
    def __init__(self, features, learning_rate=0.1, max_trees=300, max_depth=4,
                 min_samples_leaf=200, subsample=0.5, patience=30, seed=0):
        self.features = list(features)
        self.learning_rate = learning_rate
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.patience = patience
        self.seed = seed

    def _boost(self, train, n_trees, validation=None):
        """Fit up to n_trees trees; with validation, also return the validation loss after each."""
        train = train.sort_values("game", kind="stable")
        structure = GameStructure(train)
        x = train[self.features].to_numpy(dtype=float)
        u = np.zeros(len(train))
        rng = np.random.default_rng(self.seed)
        if validation is not None:
            validation = validation.sort_values("game", kind="stable")
            validation_structure = GameStructure(validation)
            x_validation = validation[self.features].to_numpy(dtype=float)
            u_validation = np.zeros(len(validation))
        trees, curve = [], []
        for _ in range(n_trees):
            gradient, hessian = structure.gradientAndHessian(u)
            # A random half of the games
            sample = rng.random(len(structure.game_ids)) < self.subsample
            rows = sample[structure.game_index]
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=int(rng.integers(1 << 31)))
            tree.fit(x[rows], -gradient[rows] / hessian[rows], sample_weight=hessian[rows])
            trees.append(tree)
            u += self.learning_rate * tree.predict(x)
            if validation is not None:
                u_validation += self.learning_rate * tree.predict(x_validation)
                curve.append(validation_structure.loss(u_validation))
                if len(curve) - 1 - int(np.argmin(curve)) >= self.patience:
                    break
        return trees, curve

    def fit(self, games, train_years):
        voted = gameData.votedGames(games)
        train_years = sorted(set(train_years))
        train = voted[voted["year"].isin(train_years)]
        if len(train_years) >= 4:
            last = train_years[-1]
            _, curve = self._boost(train[train["year"] != last], self.max_trees,
                                   train[train["year"] == last])
            self.n_trees = int(np.argmin(curve)) + 1
        else:
            self.n_trees = self.max_trees // 2
        self.trees, _ = self._boost(train, self.n_trees)
        return self

    def impact(self, frame):
        x = frame[self.features].to_numpy(dtype=float)
        return self.learning_rate * sum(tree.predict(x) for tree in self.trees)

    def predict(self, games, year):
        rows = games[games["year"] == year]
        return addVoteProbabilities(rows, self.impact(rows))

    def tally(self, predicted, player_sd=DEFAULT_PLAYER_SD):
        return seasonTally(predicted, player_sd)

    def featureImportance(self):
        """Share of the loss reduction from splits on each feature."""
        total = sum(tree.feature_importances_ for tree in self.trees)
        importance = pd.DataFrame({"feature": self.features,
                                   "importance": total / total.sum()})
        return importance.sort_values("importance", ascending=False)
