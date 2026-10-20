#!/usr/bin/env python3

# Turning game-by-game vote probabilities into a season leaderboard, without simulation.
#
# What it does: for each player, each game is a small distribution over 0, 1, 2 or 3 votes
# (from rankedLogit.py). The distribution of the player's season total is those game
# distributions convolved together, worked out exactly. The chance a player wins (or ties for)
# the medal is then sum over t of P(player polls t) * P(every other player polls t or fewer).
#
# Games aren't independent for one player, though: umpires tend to rate some players higher (or
# lower) than their stats suggest all season long, so the model's misses for a player lean the
# same way week after week. That's allowed for with a player "appeal" effect, a normal shift of
# standard deviation `player_sd` to the player's impact in every game. The season distribution
# is averaged over that shift with Gauss-Hermite quadrature: a handful of fixed shifts and
# weights that integrate a normal distribution exactly enough, rather than random draws.
# backtest.py measures which player_sd gives honest ranges and win chances.
#
# Why it matters for the analysis: this is what Monte Carlo models get by simulating the season
# thousands of times, computed directly, so it gives the same answer every run. It treats
# players as independent of each other, which isn't quite true: two players in the same game
# can't both get the 3 votes. The contenders rarely share games apart from teammates, so the
# effect on win chances is small.

import numpy as np

from rankedLogit import placeProbabilities

DEFAULT_PLAYER_SD = 0.3
QUADRATURE_POINTS = 9


def voteDistribution(p1, p2, p3):
    """Distribution of a season total (index = votes) from per-game P(1), P(2), P(3)."""
    total = np.array([1.0])
    for one, two, three in zip(p1, p2, p3):
        game = np.array([max(1 - one - two - three, 0.0), one, two, three])
        total = np.convolve(total, game)
    return total


def seasonDistributions(predicted, player_sd):
    """Season vote distribution for each player (in groupby player order), averaged over the
    player appeal effect. With player_sd 0 the predicted P(3), P(2), P(1) are used as they are,
    so this works for any model; otherwise the predictions need a Plackett-Luce impact."""
    if player_sd <= 0:
        shifts, weights = np.array([0.0]), np.array([1.0])
        columns = {0: (predicted["p1"].to_numpy(), predicted["p2"].to_numpy(),
                       predicted["p3"].to_numpy())}
    else:
        nodes, weights = np.polynomial.hermite.hermgauss(QUADRATURE_POINTS)
        shifts, weights = np.sqrt(2) * player_sd * nodes, weights / np.sqrt(np.pi)
        columns = {}

    # Per game, per shift: every player's P(3), P(2), P(1) if only their impact were shifted
    for k, shift in enumerate(shifts if player_sd > 0 else []):
        p3, p2, p1 = (np.zeros(len(predicted)) for _ in range(3))
        for _, rows in predicted.groupby("game").indices.items():
            p3[rows], p2[rows], p1[rows] = placeProbabilities(
                predicted["impact"].to_numpy()[rows], shift)
        columns[k] = (p1, p2, p3)

    positions = predicted.groupby("player").indices
    distributions = []
    for player, rows in positions.items():
        mixture = None
        for k, weight in enumerate(weights):
            p1, p2, p3 = (p[rows] for p in columns[k])
            d = weight * voteDistribution(p1, p2, p3)
            mixture = d if mixture is None else mixture + d
        distributions.append(mixture)
    return list(positions), distributions


def seasonTally(predicted, player_sd=DEFAULT_PLAYER_SD):
    """One row per player: games, expected votes (the model's share of each game's 6 votes),
    the strict 3-2-1 tally, an 80% range and the chance of winning (ties count as wins, as joint
    winners are awarded the medal) allowing for player appeal, and, where the votes are known,
    the actual votes."""
    tally = predicted.groupby("player").agg(
        name=("name", "first"),
        team=("team_name", lambda t: t.value_counts().index[0]),
        games=("game", "count"),
        expected_votes=("expected_votes", "sum"),
        votes_321=("votes_321", "sum"),
        polled_games=("p_polled", "sum"),
        actual_votes=("brownlow_votes", lambda v: v.sum(min_count=1)),
    )

    players, distributions = seasonDistributions(predicted, player_sd)
    pmf = np.zeros((len(distributions), max(len(d) for d in distributions)))
    for i, d in enumerate(distributions):
        pmf[i, : len(d)] = d
    tally = tally.loc[players]
    tally["p_win"] = winProbabilities(pmf)
    cdf = pmf.cumsum(axis=1)
    tally["votes_10pct"] = (cdf < 0.1).sum(axis=1)
    tally["votes_90pct"] = (cdf < 0.9).sum(axis=1)

    tally = tally.reset_index().sort_values("expected_votes", ascending=False)
    tally.insert(0, "rank", np.arange(1, len(tally) + 1))
    tally.attrs["pmf"] = dict(zip(players, distributions))
    return tally


def winProbabilities(pmf):
    cdf = np.clip(pmf.cumsum(axis=1), 1e-300, 1.0)
    log_cdf = np.log(cdf)
    everyone = log_cdf.sum(axis=0)
    # P(win_i) = sum_t P_i(t) * prod_{j != i} F_j(t)
    others = np.exp(everyone[None, :] - log_cdf)
    return (pmf * others).sum(axis=1)


def roundByRound(predicted, tally, top=10):
    """Cumulative expected votes after each round for the leaders, as on count night."""
    leaders = tally.head(top)
    subset = predicted[predicted["player"].isin(leaders["player"])]
    table = subset.pivot_table(index="round_sequence", columns="player", values="expected_votes",
                               aggfunc="sum").fillna(0).cumsum()
    rounds = predicted.drop_duplicates("round_sequence").set_index("round_sequence")["round_name"]
    table.index = rounds.reindex(table.index)
    table.index.name = "round"
    table = table[leaders["player"]]
    table.columns = leaders["name"]
    return table.round(1)
