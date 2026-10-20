#!/usr/bin/env python3

# The Monte Carlo model: the Plackett-Luce model's game predictions, played out as thousands of
# simulated seasons.
#
# What it does: fits the Plackett-Luce model (Plackett-Luce Model/brownlowModel.py), then simulates
# the season many times. In each simulated season:
#   1. the model's weights are redrawn from their uncertainty (a normal distribution around the
#      fitted weights, with the covariance from the curvature of the likelihood), because the
#      weights are estimates, not known values
#   2. each player gets one umpire appeal shift (normal, sd `player_sd`) used in all their games
#   3. every game's 3, 2 and 1 votes are drawn from the Plackett-Luce model, using the Gumbel
#      trick: add random Gumbel noise to every player's impact and give 3, 2 and 1 to the top
#      three, which draws them with exactly the Plackett-Luce probabilities
#   4. the votes are added up into a leaderboard
# Counting across simulations gives each player's average votes, chance of winning, chance of
# finishing in the top 10, average finishing position and range of totals.
#
# Why it matters for the analysis: the exact calculation in seasonTally.py treats players as
# independent and the weights as known. Simulation drops both simplifications: two players in the
# same game can never both get the 3 votes, a player's appeal shift is shared by all their games,
# and uncertainty in the weights moves every player at once. It also answers questions the exact
# calculation can't easily, like each player's chance of a top 10 finish. The price is random
# noise in the answers (shrinking with more simulations) and run time.

import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Plackett-Luce Model"))
import gameData  # noqa: E402
from brownlowModel import BrownlowModel  # noqa: E402
from seasonTally import DEFAULT_PLAYER_SD  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CHUNK = 500  # simulations at a time, to bound memory


def weightCovariance(model, rows):
    """Covariance of the fitted Plackett-Luce weights: the inverse of the observed information
    (sum over games and vote draws of the covariance of the features among the players still in
    the draw), plus the L2 penalty's curvature."""
    frame, x, mask, game_index, slot = model._tensorise(rows)
    votes = frame["brownlow_votes"].fillna(0).to_numpy()
    u = x @ model.beta
    games = np.arange(len(x))
    remaining = mask.copy()
    information = np.zeros((x.shape[2], x.shape[2]))
    for v in (3, 2, 1):
        masked = np.where(remaining, u, -np.inf)
        p = np.exp(masked - masked.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean = np.einsum("gn,gnf->gf", p, x)
        information += np.einsum("gn,gnf,gnh->fh", p, x, x) - mean.T @ mean
        chosen = np.zeros(len(x), dtype=int)
        chosen[game_index[votes == v]] = slot[votes == v]
        remaining[games, chosen] = False
    information += 2 * model.l2 * len(x) * np.eye(len(information))
    return np.linalg.inv(information)


class SimulationModel:
    def __init__(self, features, n_sims=10000, player_sd=DEFAULT_PLAYER_SD,
                 weight_uncertainty=True, order="votes", seed=0, l2=0.01):
        self.base = BrownlowModel(features, l2=l2)
        self.features = list(features)
        self.n_sims = n_sims
        self.player_sd = player_sd
        self.weight_uncertainty = weight_uncertainty
        self.order = order  # leaderboard order: "votes" (average) or "position" (average finish)
        self.seed = seed

    def fit(self, games, train_years):
        self.base.fit(games, train_years)
        self.covariance = weightCovariance(self.base.model, self.base.train_rows)
        return self

    def predict(self, games, year):
        """The Plackett-Luce model's exact per-game predictions (the simulation's inputs)."""
        return self.base.model.predict(self.base.prepare(games, year))

    def simulate(self, predicted):
        """Simulated season totals: (players, simulations) and the player ids in row order."""
        model = self.base.model
        x = model._standardise(predicted)
        rng = np.random.default_rng(self.seed)
        players, player_index = np.unique(predicted["player"].to_numpy(), return_inverse=True)
        game_index = np.unique(predicted["game"].to_numpy(), return_inverse=True)[1]
        slot = predicted.groupby("game").cumcount().to_numpy()
        # Row of each (game, slot), -1 for empty slots
        grid = -np.ones((game_index.max() + 1, slot.max() + 1), dtype=int)
        grid[game_index, slot] = np.arange(len(predicted))
        empty = grid < 0
        to_players = sparse.csr_matrix((np.ones(len(predicted)), (player_index, np.arange(len(predicted)))),
                                       shape=(len(players), len(predicted)))
        cholesky = np.linalg.cholesky(self.covariance) if self.weight_uncertainty else None
        game_rows = np.arange(len(grid))[:, None]

        totals = []
        for start in range(0, self.n_sims, CHUNK):
            n = min(CHUNK, self.n_sims - start)
            weights = model.beta[:, None] + (cholesky @ rng.standard_normal((len(model.beta), n))
                                             if cholesky is not None else 0)
            impact = x @ weights  # (rows, sims)
            if self.player_sd > 0:
                impact += self.player_sd * rng.standard_normal((len(players), n))[player_index]
            impact += rng.gumbel(size=impact.shape)
            per_game = impact[np.maximum(grid, 0)]  # (games, slots, sims)
            per_game[empty] = -np.inf
            top3 = np.argpartition(-per_game, 3, axis=1)[:, :3, :]
            top3_values = np.take_along_axis(per_game, top3, axis=1)
            order = np.argsort(-top3_values, axis=1)
            top3 = np.take_along_axis(top3, order, axis=1)  # slots of the 3, 2 and 1 votes
            votes = np.zeros((len(predicted), n))
            sims = np.arange(n)[None, :]
            for place, vote in enumerate((3, 2, 1)):
                votes[grid[game_rows, top3[:, place, :]], sims] = vote
            totals.append(to_players @ votes)
        return np.hstack(totals), players

    def tally(self, predicted):
        totals, players = self.simulate(predicted)
        most = totals.max(axis=0)
        # Finishing position: 1 + the number of players with more votes (ties share a position)
        ordered = np.sort(totals, axis=0)
        position = np.empty_like(totals)
        for s in range(totals.shape[1]):
            position[:, s] = 1 + len(totals) - np.searchsorted(ordered[:, s], totals[:, s], "right")

        info = predicted.groupby("player").agg(
            name=("name", "first"),
            team=("team_name", lambda t: t.value_counts().index[0]),
            games=("game", "count"),
            exact_expected_votes=("expected_votes", "sum"),
            votes_321=("votes_321", "sum"),
            actual_votes=("brownlow_votes", lambda v: v.sum(min_count=1)),
        ).loc[players]
        info["expected_votes"] = totals.mean(axis=1)
        info["sd_votes"] = totals.std(axis=1)
        info["votes_10pct"] = np.percentile(totals, 10, axis=1).round().astype(int)
        info["votes_90pct"] = np.percentile(totals, 90, axis=1).round().astype(int)
        info["p_win"] = (totals == most).mean(axis=1)  # ties count as wins
        info["p_top3"] = (position <= 3).mean(axis=1)
        info["p_top10"] = (position <= 10).mean(axis=1)
        info["mean_rank"] = position.mean(axis=1)

        key = ["mean_rank"] if self.order == "position" else ["expected_votes"]
        tally = info.reset_index().sort_values(key, ascending=self.order == "position")
        tally.insert(0, "rank", np.arange(1, len(tally) + 1))
        return tally
