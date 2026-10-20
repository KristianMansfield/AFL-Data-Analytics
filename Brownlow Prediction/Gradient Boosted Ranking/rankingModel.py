#!/usr/bin/env python3

# The gradient-boosted ranking model: LightGBM decision trees learn each player's impact in a game,
# ranked against the other players in the same game.
#
# What it does: starts every player at an impact of 0 and adds one small decision tree at a time,
# each nudging the impacts so the votes actually given become more likely, judged game by game.
# Two ranking objectives are available:
#   - lambdarank: LightGBM's standard learning-to-rank objective. It optimises the order of the
#     top three in each game (NDCG), treating 3 votes as more relevant than 2, and 2 than 1. Its
#     scores rank players but aren't probabilities, so a single scale factor is fitted on a
#     held-out season to turn them into Plackett-Luce impacts.
#   - listmle: the Plackett-Luce likelihood itself, the same loss as the Plackett-Luce model (the
#     3 votes drawn from everyone in the game in proportion to exp(impact), the 2 votes from the
#     rest, then the 1 vote), given to LightGBM as a custom objective. Its scores are Plackett-Luce
#     impacts already.
# The number of trees is chosen by holding out the last training season and stopping once more
# trees stop improving it, then refitting on every training season.
#
# Because the impacts end up on the Plackett-Luce scale, they turn into exact chances of 3, 2 and
# 1 votes and a season tally the same way as in the Plackett-Luce model (rankedLogit.py,
# seasonTally.py), so the models are compared like for like.
#
# Why it matters for the analysis: the Plackett-Luce model's impact is a weighted sum, so a stat
# always counts the same. Trees can learn that a stat counts more in some situations than others
# (e.g. goals matter more for forwards in a win, hitouts only for a ruck who also wins the ball)
# and that it stops counting after a point. If those interactions matter to the umpires, this
# model should beat the weighted sum.
#
# LightGBM needs the OpenMP runtime; on macOS: brew install libomp

import os
import sys

import lightgbm
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Plackett-Luce Model"))
import gameData  # noqa: E402
from rankedLogit import addVoteProbabilities  # noqa: E402
from seasonTally import DEFAULT_PLAYER_SD, seasonTally  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Every feature except umpire appeal (which needs a Plackett-Luce model of its own); reputation
# stands in for it, as trees can pick out the useful part themselves
GROUPS = ["basic", "advanced", "rating", "rank", "result", "progression", "form", "reputation"]

OBJECTIVES = ["lambdarank", "listmle"]


class GameStructure:
    """The players in each game laid out as a (games, players) grid, for the Plackett-Luce loss.
    Rows must be sorted by game."""

    def __init__(self, frame):
        self.game_ids, self.game_index = np.unique(frame["game"].to_numpy(), return_inverse=True)
        self.slot = frame.groupby("game", sort=True).cumcount().to_numpy()
        self.sizes = np.bincount(self.game_index)
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
    def __init__(self, features, objective="listmle", learning_rate=0.05, num_leaves=15,
                 min_data_in_leaf=200, max_rounds=1000, early_stopping=50, seed=0):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.features = list(features)
        self.objective = objective
        self.max_rounds = max_rounds
        self.early_stopping = early_stopping
        self.params = {
            "learning_rate": learning_rate, "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf, "bagging_fraction": 0.5, "bagging_freq": 1,
            "feature_fraction": 0.8, "lambda_l2": 1.0, "seed": seed, "verbosity": -1,
            "num_threads": 0,
        }
        if objective == "lambdarank":
            self.params.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [3]})
        else:
            self.params.update({"metric": "None"})

    def _dataset(self, frame, reference=None):
        frame = frame.sort_values("game", kind="stable")
        structure = GameStructure(frame)
        data = lightgbm.Dataset(frame[self.features].to_numpy(dtype=float),
                                label=frame["brownlow_votes"].fillna(0).to_numpy(),
                                group=structure.sizes, feature_name=self.features,
                                reference=reference, free_raw_data=False)
        return frame, structure, data

    def _train(self, train, rounds, validation=None):
        """Boost up to `rounds` trees, stopping early on the validation season if given."""
        _, structure, data = self._dataset(train)
        params = dict(self.params)
        if self.objective == "listmle":
            params["objective"] = lambda preds, _: structure.gradientAndHessian(preds)
        kwargs = {}
        if validation is not None:
            _, validation_structure, validation_data = self._dataset(validation, reference=data)
            kwargs = {"valid_sets": [validation_data],
                      "callbacks": [lightgbm.early_stopping(self.early_stopping, verbose=False)]}
            if self.objective == "listmle":
                kwargs["feval"] = lambda preds, _: (
                    "plackett_luce", validation_structure.loss(preds), False)
        return lightgbm.train(params, data, num_boost_round=rounds, **kwargs)

    def fit(self, games, train_years):
        voted = gameData.votedGames(games)
        train_years = sorted(set(train_years))
        train = voted[voted["year"].isin(train_years)]
        self.scale = 1.0
        if len(train_years) >= 4:
            last = train_years[-1]
            held_out = train[train["year"] == last].sort_values("game", kind="stable")
            booster = self._train(train[train["year"] != last], self.max_rounds, held_out)
            self.n_trees = booster.best_iteration or self.max_rounds
            if self.objective == "lambdarank":
                # One scale factor turning ranking scores into Plackett-Luce impacts
                scores = booster.predict(held_out[self.features].to_numpy(dtype=float),
                                         num_iteration=self.n_trees)
                structure = GameStructure(held_out)
                self.scale = minimize_scalar(lambda a: structure.loss(a * scores),
                                             bounds=(0.01, 100), method="bounded").x
        else:
            self.n_trees = 200
        self.booster = self._train(train, self.n_trees)
        return self

    def impact(self, frame):
        return self.scale * self.booster.predict(frame[self.features].to_numpy(dtype=float))

    def predict(self, games, year):
        rows = games[games["year"] == year]
        return addVoteProbabilities(rows, self.impact(rows))

    def tally(self, predicted, player_sd=DEFAULT_PLAYER_SD):
        return seasonTally(predicted, player_sd)

    def featureImportance(self):
        """Share of the total loss reduction from splits on each feature."""
        gain = self.booster.feature_importance("gain")
        importance = pd.DataFrame({"feature": self.features, "importance": gain / gain.sum()})
        return importance.sort_values("importance", ascending=False)
