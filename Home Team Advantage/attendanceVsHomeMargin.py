#!/usr/bin/env python3

# Is there a correlation between attendance and the home team's margin?
#
# What it does: for every home and away game in the chosen seasons, compares the crowd with the
# home side's margin three ways:
#   1. raw: attendance against home margin
#   2. allowing for the teams and the venue: is a crowd bigger than that venue's usual that season
#      linked to the home side doing better than expected? "Expected" is fitted by least squares
#      over every game: one strength per team per season and one home advantage per team, so a
#      good team drawing big crowds isn't mistaken for crowds helping the home side
#   3. shared grounds: games where the away team also plays home games at the venue that season
#      (e.g. Collingwood v Carlton at the MCG). These draw the biggest crowds, and the "home" side
#      there has almost no home advantage, which drags the raw relationship down
# and draws attendance against home margin, coloured by shared or not, with each group's average
# margin across bands of crowd size.
#
# The 2020 and 2021 seasons are left out by default (games played with no or capped crowds), as
# are games with no attendance recorded. Venues have no capacity recorded, so crowd size is
# compared with the venue's median crowd that season instead.
#
# Outputs: output/attendanceVsHomeMargin_<years>.png
#
# Usage: venv/bin/python attendanceVsHomeMargin.py [--years 1990-2026] [--include-covid]

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dbConnection import createReaderConnection, finalRound

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
COVID_SEASONS = (2020, 2021)

# Validated categorical pair (blue, orange); text stays in ink, never the series colour
NOT_SHARED_COLOUR = "#2a78d6"
SHARED_COLOUR = "#eb6834"
INK = "#3d3d3a"
MUTED = "#73726c"
GRID = "#e4e3df"
BAND_WIDTH = 5000  # crowd bands for the average lines
MIN_BAND_GAMES = 30

GAMES_SQL = f"""
SELECT g.id AS game, s.year, g.home_team, g.away_team, g.venue, g.attendance,
       g.final_home_score - g.final_away_score AS margin
FROM games g
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
WHERE NOT {finalRound()} AND s.year BETWEEN ? AND ? AND g.attendance > 0
"""


def parseYears(text):
    """"2024" -> (2024, 2024); "1990-2026" -> (1990, 2026)."""
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def loadGames(years, include_covid):
    db = createReaderConnection()
    cursor = db.execute(GAMES_SQL, years)
    games = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description]).astype(float)
    db.close()
    if not include_covid:
        games = games[~games["year"].isin(COVID_SEASONS)].reset_index(drop=True)
    return games


def expectedMargin(games):
    """Home margin expected from each team's strength that season and the home team's own home
    advantage, by least squares."""
    home = games["home_team"].astype(int).astype(str) + "_" + games["year"].astype(int).astype(str)
    away = games["away_team"].astype(int).astype(str) + "_" + games["year"].astype(int).astype(str)
    team_seasons = {ts: i for i, ts in enumerate(sorted(set(home) | set(away)))}
    home_teams = {t: len(team_seasons) + i for i, t in enumerate(sorted(games["home_team"].unique()))}
    design = np.zeros((len(games), len(team_seasons) + len(home_teams)))
    rows = np.arange(len(games))
    design[rows, home.map(team_seasons)] += 1
    design[rows, away.map(team_seasons)] -= 1
    design[rows, games["home_team"].map(home_teams)] = 1
    coefficients, *_ = np.linalg.lstsq(design, games["margin"].to_numpy(), rcond=None)
    return design @ coefficients


def correlation(x, y):
    r, p = stats.pearsonr(x, y)
    return f"r = {r:+.3f} (p = {p:.2g})"


def bandMeans(games):
    """Average home margin in each crowd band with enough games, at the band's average crowd."""
    bands = (games["attendance"] // BAND_WIDTH).astype(int)
    grouped = games.groupby(bands)
    means = grouped.agg(attendance=("attendance", "mean"), margin=("margin", "mean"),
                        n=("margin", "size"))
    return means[means["n"] >= MIN_BAND_GAMES]


def drawGraph(games, label, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    groups = ((False, NOT_SHARED_COLOUR, "Away team has its own home ground"),
              (True, SHARED_COLOUR, "Shared ground"))
    for shared, colour, name in groups:
        subset = games[games["shared"] == shared]
        ax.scatter(subset["attendance"], subset["margin"], s=6, color=colour, alpha=0.18,
                   linewidths=0, zorder=2)
    for shared, colour, name in groups:
        subset = games[games["shared"] == shared]
        means = bandMeans(subset)
        ax.plot(means["attendance"], means["margin"], color=colour, linewidth=2,
                solid_capstyle="round", zorder=4,
                label=f"{name}: {len(subset):,} games, average {subset['margin'].mean():+.1f}")
        ax.scatter(means["attendance"], means["margin"], s=64, color=colour,
                   edgecolors="white", linewidths=2, zorder=5)
        end = means.iloc[-1]
        ax.annotate(name, (end["attendance"], end["margin"]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)

    ax.set_ylim(np.percentile(games["margin"], [0.5, 99.5]) * 1.05)
    ax.set_title(f"Home margin by attendance ({label}): the home edge comes from not sharing "
                 "the ground, not crowd size", fontsize=11)
    ax.set_xlabel("Attendance")
    ax.set_ylabel("Home team margin (points)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=9,
              title=f"Lines: average margin per {BAND_WIDTH // 1000}K crowd band", title_fontsize=9)
    ax.set_xlim(right=games["attendance"].max() * 1.12)  # room for the end labels
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Correlation between attendance and home margin")
    parser.add_argument("--years", type=parseYears, default=(1990, 2026),
                        help="Season or range of seasons (default 1990-2026)")
    parser.add_argument("--include-covid", action="store_true",
                        help="Keep 2020 and 2021, played with no or capped crowds")
    args = parser.parse_args()

    games = loadGames(args.years, args.include_covid)
    first, last = args.years
    label = f"{first}-{last}" if first != last else str(first)
    left_out = any(first <= y <= last for y in COVID_SEASONS) and not args.include_covid
    print(f"{len(games):,} home and away games, {label}"
          + (" without 2020-21" if left_out else ""))

    # 1. Raw
    fit = stats.linregress(games["attendance"] / 10000, games["margin"])
    print(f"\n1. Attendance v home margin: {correlation(games['attendance'], games['margin'])}, "
          f"{fit.slope:+.1f} points per 10,000 extra people")

    # 2. Allowing for team strength and venue
    games["expected"] = expectedMargin(games)
    games["vs_expected"] = games["margin"] - games["expected"]
    venue_season = games.groupby(["venue", "year"])["attendance"]
    games["crowd_vs_usual"] = np.log(games["attendance"] / venue_season.transform("median"))
    enough = venue_season.transform("count") >= 5
    usual = games[enough]
    fit = stats.linregress(usual["crowd_vs_usual"], usual["vs_expected"])
    print(f"\n2. Allowing for team strength and venue ({len(usual):,} games at venues with 5+ games "
          f"that season):")
    print(f"   crowd v venue's usual, against margin v expected: "
          f"{correlation(usual['crowd_vs_usual'], usual['vs_expected'])}; a crowd 20% above usual "
          f"goes with {fit.slope * np.log(1.2):+.1f} points")
    fifths = pd.qcut(usual["crowd_vs_usual"], 5,
                     labels=["much smaller", "smaller", "usual", "bigger", "much bigger"])
    for name, subset in usual.groupby(fifths, observed=True):
        error = 1.96 * subset["vs_expected"].std() / np.sqrt(len(subset))
        print(f"     {name:<13} crowd {np.exp(subset['crowd_vs_usual'].median()):>5.0%} of usual, "
              f"home margin {subset['margin'].mean():+6.1f}, v expected "
              f"{subset['vs_expected'].mean():+5.1f} (+/- {error:.1f})")

    # 3. Shared grounds
    home_grounds = set(zip(games["home_team"], games["venue"], games["year"]))
    games["shared"] = [(a, v, y) in home_grounds for a, v, y in
                       zip(games["away_team"], games["venue"], games["year"])]
    print("\n3. Shared grounds (the away team also plays home games at the venue that season):")
    for shared, subset in games.groupby("shared"):
        print(f"   {'shared' if shared else 'not shared':<11} {len(subset):>6,} games, median crowd "
              f"{subset['attendance'].median():>7,.0f}, home margin {subset['margin'].mean():+5.1f}, "
              f"home side won {(subset['margin'] > 0).mean():.0%}, attendance v margin "
              f"{correlation(subset['attendance'], subset['margin'])}")
    by_crowd = games.groupby(pd.qcut(games["attendance"], 5), observed=True)
    print("\n   By crowd size:")
    for band, subset in by_crowd:
        print(f"     {int(band.left):>7,}-{int(band.right):<7,} home margin {subset['margin'].mean():+5.1f}, "
              f"home side won {(subset['margin'] > 0).mean():.0%}, on a shared ground "
              f"{subset['shared'].mean():.0%}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"attendanceVsHomeMargin_{label}.png")
    drawGraph(games, label, path)
    print(f"\nGraph saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
