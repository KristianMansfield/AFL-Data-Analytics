#!/usr/bin/env python3

# The full Brownlow model: the ranked logit (rankedLogit.py) plus the umpire appeal feature, which
# needs a fitted model of its own.
#
# What it does: umpire appeal is how many more (or fewer) votes a player got over the previous
# three seasons than their play earned, by the model's reckoning: (actual - expected votes) /
# (games + 10), the 10 shrinking it towards zero for players with few games. To work it out, a
# first model is fitted without it (stage one) and used to predict the earlier seasons; the
# differences from the votes actually given become the feature, and the final model is fitted
# with it (stage two). Only seasons before the one a row belongs to are ever used, so a season's
# own votes never feed into its own predictions.
#
# Why it matters for the analysis: some players poll more than their stats suggest year after
# year (and some less), and that carries on into the next season. Reputation (votes per game)
# mixes that up with simply being a good player, which the stats already capture; appeal is only
# the part the stats miss. In the backtests it improved the winner's predicted rank and the order
# of the top 10.
#
# Without the umpire_appeal feature this is just the ranked logit.

import pandas as pd

import gameData
from rankedLogit import RankedLogit
from seasonTally import DEFAULT_PLAYER_SD, seasonTally

APPEAL = "umpire_appeal"
APPEAL_SEASONS = 3
APPEAL_SHRINKAGE_GAMES = 10


class BrownlowModel:
    def __init__(self, features, l2=0.01):
        self.features = list(features)
        self.l2 = l2
        self.uses_appeal = APPEAL in self.features
        self.stage_one = None
        self._residuals = None

    def _seasonResiduals(self, games, before):
        """Actual - expected votes and games for each player in each voted season before `before`,
        predicted by the stage one model (cached across calls)."""
        voted = gameData.votedGames(games)
        years = sorted(set(voted["year"][voted["year"] < before]))
        missing = [y for y in years if self._residuals is None or y not in set(self._residuals["year"])]
        if missing:
            predicted = self.stage_one.predict(voted[voted["year"].isin(missing)])
            predicted["residual"] = predicted["brownlow_votes"].fillna(0) - predicted["expected_votes"]
            new = predicted.groupby(["year", "player"]).agg(
                residual=("residual", "sum"), games=("residual", "count")).reset_index()
            self._residuals = pd.concat([self._residuals, new]) if self._residuals is not None else new
        return self._residuals

    def _withAppeal(self, games, rows):
        """`rows` (a subset of games) with the umpire_appeal column."""
        if not self.uses_appeal:
            return rows
        residuals = self._seasonResiduals(games, rows["year"].max())
        appeal = []
        for year in rows["year"].unique():
            window = residuals[(residuals["year"] < year) & (residuals["year"] >= year - APPEAL_SEASONS)]
            totals = window.groupby("player")[["residual", "games"]].sum()
            appeal.append(pd.DataFrame({
                "year": year, "player": totals.index,
                APPEAL: (totals["residual"] / (totals["games"] + APPEAL_SHRINKAGE_GAMES)).to_numpy(),
            }))
        appeal = pd.concat(appeal)
        joined = rows[["year", "player"]].merge(appeal, on=["year", "player"], how="left")
        rows = rows.copy()
        rows[APPEAL] = joined[APPEAL].fillna(0.0).to_numpy()  # no history: no appeal either way
        return rows

    def fit(self, games, train_years):
        """Fit on the voted games in the seasons train_years. `games` may hold other seasons too:
        earlier ones are used for umpire appeal, later ones are ignored."""
        train_years = set(train_years)
        voted = gameData.votedGames(games)
        train = voted[voted["year"].isin(train_years)]
        if self.uses_appeal:
            stage_one_features = [f for f in self.features if f != APPEAL]
            self.stage_one = RankedLogit(stage_one_features, self.l2).fit(train)
            self._residuals = None
            train = self._withAppeal(voted, train)
        self.model = RankedLogit(self.features, self.l2).fit(train)
        self.train_rows = train  # with every feature, for the simulation model's uncertainty
        return self

    def predict(self, games, year):
        """Predictions for every game in `year` (see RankedLogit.predict)."""
        return self.model.predict(self.prepare(games, year))

    def logLikelihood(self, games, year):
        rows = gameData.votedGames(games[games["year"] == year])
        return self.model.logLikelihood(self._withAppeal(games, rows))

    def prepare(self, games, year):
        """The rows for `year` with every feature (including umpire appeal), as the final model
        sees them."""
        return self._withAppeal(games, games[games["year"] == year])

    def tally(self, predicted, player_sd=DEFAULT_PLAYER_SD):
        return seasonTally(predicted, player_sd)

    def coefficients(self):
        return self.model.coefficients()
