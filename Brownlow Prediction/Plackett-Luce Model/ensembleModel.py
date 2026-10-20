#!/usr/bin/env python3

# The Plackett-Luce model blended with the gradient-boosted ranker (predictBrownlow.py --ensemble).
#
# What it does: fits both the Plackett-Luce model (brownlowModel.py) and the LightGBM ListMLE
# ranker (Gradient Boosted Ranking/rankingModel.py) on the same seasons. Both give every player an
# impact on the same Plackett-Luce scale, so each player's impact in a game is a weighted average
# of the two (70% Plackett-Luce, 30% ranker by default), and the chances of 3, 2 and 1 votes and
# the season tally follow from the blend as usual.
#
# Why it matters for the analysis: the two models make different mistakes, so the blend predicts
# the contenders' votes game by game better than either (better than the Plackett-Luce model alone
# in 12 of 12 seasons for 2015-2026 and 19 of 21 for 2006-2026). It wasn't clearly better at
# picking the winner or ordering the top 10, which is why it's an option rather than the default.

import os
import sys

from brownlowModel import BrownlowModel
from rankedLogit import addVoteProbabilities
from seasonTally import DEFAULT_PLAYER_SD, seasonTally

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Gradient Boosted Ranking"))
from rankingModel import BoostedRanker  # noqa: E402

DEFAULT_WEIGHT = 0.7  # share of the Plackett-Luce model in the blend


class EnsembleModel:
    def __init__(self, features, ranking_features, weight=DEFAULT_WEIGHT, l2=0.01):
        self.weight = weight
        self.plackett_luce = BrownlowModel(features, l2=l2)
        self.ranker = BoostedRanker(ranking_features)

    def fit(self, games, train_years):
        self.plackett_luce.fit(games, train_years)
        self.ranker.fit(games, train_years)
        return self

    def predict(self, games, year):
        predicted = self.plackett_luce.predict(games, year)
        blend = (self.weight * predicted["impact"].to_numpy()
                 + (1 - self.weight) * self.ranker.impact(predicted))
        return addVoteProbabilities(predicted.drop(columns="impact"), blend)

    def tally(self, predicted, player_sd=DEFAULT_PLAYER_SD):
        return seasonTally(predicted, player_sd)

    def coefficients(self):
        """The Plackett-Luce part's weights (the ranker's trees have none to show)."""
        return self.plackett_luce.coefficients()
