#!/usr/bin/env python3

# The vote model: a ranked (Plackett-Luce) logit over the players in each game.
#
# What it does: every player in a game gets an impact score, a weighted sum of their features
# (u = x . beta). The umpires' 3 votes are modelled as a draw from all players in the game with
# probability proportional to exp(u); the 2 votes as a draw from the players left; the 1 vote from
# the players left after that. The weights are fitted by maximum likelihood on the votes actually
# given (with a small L2 penalty), so the data decides how much a disposal, a win or a late goal
# is worth, rather than those weights being guessed.
#
# Given the weights, the chance of each player getting 3, 2 or 1 votes in a game is worked out
# exactly by summing over every possible order of the top three (no simulation), and from those
# the expected votes, which always add up to exactly 6 per game.
#
# Why it matters for the analysis: this is the model's answer to "how impactful was each
# player?" in a form that matches how votes are actually given. Only differences between players
# in the same game matter (anything shared by everyone in a game, like the weather, cancels out),
# which is exactly how umpires vote: relative to the other 43 players on the ground, not against
# a fixed bar. The fitted weights are readable as "a one standard deviation increase in this stat
# multiplies a player's chance of polling by exp(beta)".

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


class RankedLogit:
    def __init__(self, features, l2=0.01):
        self.features = list(features)
        self.l2 = l2

    def _standardise(self, frame):
        x = (frame[self.features].to_numpy(dtype=float) - self.means) / self.stds
        return np.nan_to_num(x, nan=0.0)  # missing stats count as average

    def _tensorise(self, frame):
        """(games, players, features) array, a mask of real players, and the game ids."""
        frame = frame.sort_values("game", kind="stable")
        game_ids, game_index = np.unique(frame["game"].to_numpy(), return_inverse=True)
        slot = frame.groupby("game").cumcount().to_numpy()
        x = np.zeros((len(game_ids), slot.max() + 1, len(self.features)))
        mask = np.zeros(x.shape[:2], dtype=bool)
        x[game_index, slot] = self._standardise(frame)
        mask[game_index, slot] = True
        return frame, x, mask, game_index, slot

    def fit(self, frame):
        """Fit the weights on games with one 3, 2 and 1 vote each (see gameData.votedGames)."""
        values = frame[self.features].to_numpy(dtype=float)
        self.means = np.nanmean(values, axis=0)
        self.stds = np.nanstd(values, axis=0)
        self.stds[~(self.stds > 0)] = 1.0

        frame, x, mask, game_index, slot = self._tensorise(frame)
        votes = frame["brownlow_votes"].fillna(0).to_numpy()
        # Slot of the 3, 2 and 1 vote players in each game
        winners = []
        for v in (3, 2, 1):
            chosen = np.zeros(len(x), dtype=int)
            chosen[game_index[votes == v]] = slot[votes == v]
            winners.append(chosen)
        rows = np.arange(len(x))
        # Players still in the draw for the 3, 2 and 1 votes
        masks = [mask.copy()]
        for chosen in winners[:2]:
            next_mask = masks[-1].copy()
            next_mask[rows, chosen] = False
            masks.append(next_mask)
        chosen_x = sum(x[rows, chosen] for chosen in winners)
        n_games = len(x)

        def objective(beta):
            u = x @ beta
            nll = -(u[rows, winners[0]] + u[rows, winners[1]] + u[rows, winners[2]]).sum()
            expected = np.zeros_like(chosen_x)
            for m in masks:
                masked = np.where(m, u, -np.inf)
                lse = logsumexp(masked, axis=1)
                nll += lse.sum()
                p = np.exp(masked - lse[:, None])
                expected += np.einsum("gn,gnf->gf", p, x)
            grad = -(chosen_x - expected).sum(axis=0)
            return (nll / n_games + self.l2 * beta @ beta,
                    grad / n_games + 2 * self.l2 * beta)

        result = minimize(objective, np.zeros(len(self.features)), jac=True, method="L-BFGS-B")
        self.beta = result.x
        self.train_games = n_games
        self.train_nll = result.fun
        return self

    def coefficients(self):
        """Weights per standard deviation of each feature, and as a multiplier on the chance of
        polling."""
        table = pd.DataFrame({"feature": self.features, "weight_per_sd": self.beta,
                              "multiplier_per_sd": np.exp(self.beta), "sd": self.stds})
        return table.reindex(table["weight_per_sd"].abs().sort_values(ascending=False).index)

    def predict(self, frame):
        """The frame with impact (u), P(3 votes), P(2 votes), P(1 vote), expected votes and the
        model's single most likely 3-2-1 allocation (votes_321) for every player."""
        frame, x, mask, game_index, slot = self._tensorise(frame)
        u = x @ self.beta
        return addVoteProbabilities(frame, u[game_index, slot])

    def logLikelihood(self, frame):
        """Mean log probability per game of the 3-2-1 votes actually given."""
        predicted = self.predict(frame)
        votes = predicted["brownlow_votes"].fillna(0)
        total = 0.0
        for _, game in predicted.groupby("game"):
            u = game["impact"].to_numpy()
            v = votes.loc[game.index].to_numpy()
            remaining = np.ones(len(u), dtype=bool)
            for vote in (3, 2, 1):
                chosen = np.flatnonzero(v == vote)[0]
                total += u[chosen] - logsumexp(u[remaining])
                remaining[chosen] = False
        return total / predicted["game"].nunique()


def addVoteProbabilities(frame, impact):
    """The frame with impact, P(3), P(2), P(1), expected votes and the most likely 3-2-1
    (votes_321) for every player, from Plackett-Luce impacts (one per row, from any model)."""
    frame = frame.copy()
    impact = np.asarray(impact, dtype=float)
    p3, p2, p1 = (np.zeros(len(frame)) for _ in range(3))
    for rows in frame.groupby("game").indices.values():
        p3[rows], p2[rows], p1[rows] = placeProbabilities(impact[rows])
    frame["impact"] = impact
    frame["p3"], frame["p2"], frame["p1"] = p3, p2, p1
    return addVoteSummaries(frame)


def addVoteSummaries(frame):
    """Expected votes, P(polled) and the most likely 3-2-1 from the p3, p2, p1 columns."""
    frame["p_polled"] = frame["p3"] + frame["p2"] + frame["p1"]
    frame["expected_votes"] = 3 * frame["p3"] + 2 * frame["p2"] + frame["p1"]
    order = frame.groupby("game")["expected_votes"].rank(ascending=False, method="first")
    frame["votes_321"] = np.select([order == 1, order == 2, order == 3], [3, 2, 1], 0)
    return frame


def placeProbabilities(u, shift=0.0):
    """Exact chance of each player being drawn first, second and third under the ranked logit
    with impacts u, by summing over every ordered pair of players who could come before them.

    With a shift, each player's chances are worked out as if their own impact (only theirs) were
    `shift` higher: what the game would look like if the umpires rated that player more (or
    less, for a negative shift) than the model does."""
    w = np.exp(u - u.max())
    own = w * np.exp(shift)  # player i's weight, shifted
    total = w.sum() - w + own  # the game's total weight from player i's point of view
    n = len(w)
    idx = np.arange(n)
    tiny = 1e-300

    first = own / total
    # P(j first, then i second), [i, j]
    second_via = (w[None, :] / total[:, None]) * own[:, None] / np.maximum(
        total[:, None] - w[None, :], tiny)
    second_via[idx, idx] = 0.0
    second = second_via.sum(axis=1)
    # P(j first, k second, then i third), [i, j, k]
    left_after_j = np.maximum(total[:, None] - w[None, :], tiny)
    with np.errstate(over="ignore", invalid="ignore"):  # only in entries zeroed below
        third_via = ((w[None, :] / total[:, None])[:, :, None]
                     * w[None, None, :] / left_after_j[:, :, None]
                     * own[:, None, None]
                     / np.maximum(left_after_j[:, :, None] - w[None, None, :], tiny))
    third_via[:, idx, idx] = 0.0  # j and k are different players
    third_via[idx, idx, :] = 0.0  # neither is i
    third_via[idx, :, idx] = 0.0
    third = third_via.sum(axis=(1, 2))
    return first, second, third
