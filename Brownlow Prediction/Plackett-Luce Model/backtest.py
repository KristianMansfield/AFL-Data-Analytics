#!/usr/bin/env python3

# Backtest the Brownlow model: how well does it predict seasons it hasn't seen?
#
# What it does: for each test season, fits the vote model on the earlier seasons only (an
# expanding window, as it would have been used at the time; --loso trains on every other season
# instead) and predicts the test season. It does this for several variants of the model that
# switch feature groups on and off, so each idea is judged by whether it improves predictions:
#   - "disposals+goals" is what brownlowStats.py looks at, as a yardstick
#   - "- result" leaves out the winning side features, answering "should winners be favoured?"
#   - "- progression" leaves out the scoring progression features (late and match-winning goals)
#   - "full" adds umpire appeal (brownlowModel.py) to "+ form"; "+ reputation" adds back the
#     votes-per-game feature that appeal replaced
#
# By default it uses every feature and the seasons afl.com.au has full stats for (2012 onwards, so
# 2015 onwards is tested). With --basic-stats it leaves out the afl.com.au features and goes back to
# 2003, testing 2006 onwards: about twice as many test seasons, which matters when judging the
# winner and top 10, where each season gives only one result. Variants that become identical
# without the afl.com.au features (e.g. "+ advanced") are skipped.
#
# Game-level measures:
#   log-lik     mean log probability of the 3-2-1 actually given (higher is better; picking at
#               random from ~44 players scores about -11.2)
#   top pick    share of games where the model's most likely 3 vote player got the 3 votes
#   top 3       share of vote getters who were in the model's top three for the game
# Season-level measures, for both ways of turning games into a tally: expected votes (each
# player's share of the 6 votes in each game) and strict 3-2-1 (the model's top three get 3, 2, 1):
#   MAE         mean absolute error of season totals, over players who polled or were predicted to
#   winner rank where the actual winner sat on the predicted leaderboard (1 = picked the winner)
#   top 10      how many of the actual top 10 were in the predicted top 10
#   top 10 order Spearman correlation between the actual and predicted positions of the actual
#               top 10 (1 = exactly the right order)
# Calibration of the season ranges (only for the default variant) for several player_sd values:
#   in 80%      share of players' actual totals inside their predicted 80% range (should be ~80%)
#   season lik  mean log probability of each contender's actual total
#   P(winner)   the chance the model gave the actual winner
#
# Why it matters for the analysis: a model with 44 features can fit the seasons it's trained on
# well and still predict new seasons badly. Scoring each variant on unseen seasons is the only
# fair way to decide which ideas belong in the model, and how confident it should be.
#
# Usage: ../../venv/bin/python backtest.py [--basic-stats] [--years 2012-2026] [--min-train 3]
#        [--loso] [--variants full "full - result"]

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import gameData
from brownlowModel import BrownlowModel
from seasonTally import seasonTally

ALL = gameData.DEFAULT_GROUPS
VARIANTS = {
    "disposals+goals": ["baseline"],
    "basic stats": ["basic"],
    "+ advanced": ["basic", "advanced", "rating"],
    "+ rank": ["basic", "advanced", "rating", "rank"],
    "+ result": ["basic", "advanced", "rating", "rank", "result"],
    "+ progression": ["basic", "advanced", "rating", "rank", "result", "progression"],
    "+ form": ["basic", "advanced", "rating", "rank", "result", "progression", "form"],
    "full": ALL,  # + umpire appeal
    "full - result": [g for g in ALL if g != "result"],
    "full - progression": [g for g in ALL if g != "progression"],
    "full - form": [g for g in ALL if g != "form"],
    "full + reputation": ALL + ["reputation"],
}

PLAYER_SDS = [0.0, 0.2, 0.3, 0.4, 0.5]


def seasonMetrics(tally, column):
    ranked = tally.sort_values(column, ascending=False, kind="stable").reset_index(drop=True)
    actual = ranked["actual_votes"].fillna(0)
    winners = actual == actual.max()
    contenders = (actual > 0) | (ranked[column] >= 1)
    actual_rank = actual.rank(ascending=False, method="min")
    in_top10 = actual_rank <= 10
    predicted_top10 = set(ranked.head(10)["player"])
    actual_top10 = set(ranked.loc[in_top10, "player"])
    return {
        "MAE": (ranked.loc[contenders, column] - actual[contenders]).abs().mean(),
        "winner rank": int(np.flatnonzero(winners)[0]) + 1,  # best placed of joint winners
        "top 10": len(predicted_top10 & actual_top10),
        # ranked's index is the predicted position - 1
        "top 10 order": spearmanr(actual_rank[in_top10], ranked.index[in_top10]).statistic,
    }


def calibration(tally):
    pmf = tally.attrs["pmf"]
    actual = tally.set_index("player")["actual_votes"].fillna(0)
    contenders = tally[(tally["actual_votes"].fillna(0) > 0) | (tally["expected_votes"] >= 1)]
    inside, loglik = [], []
    for _, row in contenders.iterrows():
        votes = int(actual[row["player"]])
        inside.append(row["votes_10pct"] <= votes <= row["votes_90pct"])
        d = pmf[row["player"]]
        loglik.append(np.log(max(d[votes] if votes < len(d) else 0.0, 1e-12)))
    best = actual.max()
    return {"in 80%": np.mean(inside), "season lik": np.mean(loglik),
            "P(winner)": tally.loc[tally["actual_votes"] == best, "p_win"].sum()}


def main():
    parser = argparse.ArgumentParser(description="Backtest the Brownlow vote model")
    parser.add_argument("--basic-stats", action="store_true",
                        help=f"Leave out the afl.com.au features (only recorded from "
                        f"{gameData.FIRST_AFLCOM_YEAR}) and use seasons from "
                        f"{gameData.FIRST_BASIC_YEAR}, for more test seasons")
    parser.add_argument("--years", type=gameData.parseYears,
                        help=f"Seasons to use (default {gameData.FIRST_AFLCOM_YEAR} to the latest, "
                        f"or {gameData.FIRST_BASIC_YEAR} to the latest with --basic-stats)")
    parser.add_argument("--min-train", type=int, default=3,
                        help="Seasons of training data before the first test season (default 3)")
    parser.add_argument("--loso", action="store_true",
                        help="Train on every other season (leave one season out) instead of "
                        "only earlier ones")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        help=f"Variants to run (default all: {', '.join(VARIANTS)})")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 penalty (default 0.01)")
    args = parser.parse_args()

    first_year = gameData.FIRST_BASIC_YEAR if args.basic_stats else gameData.FIRST_AFLCOM_YEAR
    first, last = args.years or (first_year, gameData.latestSeason())
    print(f"Loading {first}-{last}{' (basic stats only)' if args.basic_stats else ''}...")
    games = gameData.loadGames((first, last))
    voted = gameData.votedGames(games)
    years = sorted(int(y) for y in voted["year"].unique())
    test_years = years if args.loso else years[args.min_train:]

    variants = {}
    for variant in args.variants:
        features = gameData.featureColumns(VARIANTS[variant], args.basic_stats)
        if features not in variants.values():
            variants[variant] = features

    game_rows, season_rows, calibration_rows = [], [], []
    for variant, features in variants.items():
        for year in test_years:
            train_years = [y for y in years if (y != year if args.loso else y < year)]
            model = BrownlowModel(features, l2=args.l2).fit(voted, train_years)
            predicted = model.predict(voted, year)
            actual = predicted["brownlow_votes"].fillna(0)
            ranked = predicted.groupby("game")["expected_votes"].rank(ascending=False,
                                                                      method="first")
            game_rows.append({
                "variant": variant, "year": year,
                "log-lik": model.logLikelihood(voted, year),
                "top pick": ((ranked == 1) & (actual == 3)).sum() / predicted["game"].nunique(),
                "top 3": ((ranked <= 3) & (actual > 0)).sum() / (actual > 0).sum(),
            })

            sds = PLAYER_SDS if variant == "full" else [0.0]
            for sd in sds:
                tally = seasonTally(predicted, sd)
                if sd == 0.0:
                    for column, method in (("expected_votes", "expected"),
                                           ("votes_321", "3-2-1")):
                        season_rows.append({"variant": variant, "year": year, "tally": method,
                                            **seasonMetrics(tally, column)})
                if variant == "full":
                    calibration_rows.append({"player_sd": sd, "year": year,
                                             **calibration(tally)})
        summary = pd.DataFrame(game_rows)
        summary = summary[summary["variant"] == variant].mean(numeric_only=True)
        print(f"  {variant:<20} log-lik {summary['log-lik']:.3f}  top pick "
              f"{summary['top pick']:.1%}  top 3 {summary['top 3']:.1%}")

    game_table = pd.DataFrame(game_rows)
    season_table = pd.DataFrame(season_rows)
    calibration_table = pd.DataFrame(calibration_rows)

    out_dir = os.path.join(gameData.OUTPUT_DIR,
                           "backtest_basic_stats" if args.basic_stats else "backtest")
    os.makedirs(out_dir, exist_ok=True)
    game_table.to_csv(os.path.join(out_dir, "game_metrics.csv"), index=False)
    season_table.to_csv(os.path.join(out_dir, "season_metrics.csv"), index=False)
    calibration_table.to_csv(os.path.join(out_dir, "calibration.csv"), index=False)

    print(f"\nTest seasons: {test_years[0]}-{test_years[-1]} "
          f"({'leave one season out' if args.loso else 'trained on earlier seasons only'})")
    print("\nGame level (mean over test seasons):")
    print(game_table.groupby("variant", sort=False)[["log-lik", "top pick", "top 3"]].mean()
          .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nSeason level (mean over test seasons, apart from winner_rank_median and winner_picked):")
    season_summary = season_table.groupby(["variant", "tally"], sort=False).agg(
        MAE=("MAE", "mean"), winner_rank=("winner rank", "mean"),
        winner_rank_median=("winner rank", "median"),
        winner_picked=("winner rank", lambda r: (r == 1).sum()), top_10=("top 10", "mean"),
        top_10_order=("top 10 order", "mean"),
    )
    print(season_summary.to_string(float_format=lambda v: f"{v:.2f}"))

    if not calibration_table.empty:
        print("\nCalibration of the full model's season ranges by player_sd:")
        print(calibration_table.groupby("player_sd").mean(numeric_only=True)
              .drop(columns="year").to_string(float_format=lambda v: f"{v:.3f}"))

        full = season_table[(season_table["variant"] == "full")
                            & (season_table["tally"] == "expected")]
        print("\nFull model by season:")
        print(full[["year", "MAE", "winner rank", "top 10", "top 10 order"]].to_string(index=False,
              float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
