#!/usr/bin/env python3

# Does the distance teams travel affect game outcomes?
#
# What it does: works out how far each team travelled from its home city to the venue
# (locations.py), then estimates the home side's margin for bands of away-team travel, allowing
# for how strong both teams were that season. The fit is least squares over every game: one
# strength per team per season, plus one margin for each travel band. Games are banded by how far
# the away side came when the home side was at home (under 50 km from its city), with a separate
# band for games where the home side travelled too (e.g. Hawthorn in Launceston). It also fits:
#   - a straight line: margin per 1,000 km the away side travelled further than the home side
#   - interstate trips only: does going further still matter once a side is interstate?
#   - the travel bands era by era, to see whether travel matters less than it used to
#
# Finals are left out (venues are set by the finals rules, not the fixture), as is 2020, when
# teams were based in hubs. The ± figures are 95% confidence intervals.
#
# Outputs: output/travelDistance_<years>.png
#
# Usage: venv/bin/python travelDistance.py [--years 1990-2026] [--era-length 10]

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dbConnection import createReaderConnection, finalRound
from locations import travelKm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
HUB_SEASONS = (2020,)
AT_HOME_KM = 50

# (lowest km, highest km, label) for the away side's trip when the home side is at home
BANDS = [
    (0, 50, "Same city"),
    (50, 150, "Under 150 km (mostly Geelong)"),
    (150, 1000, "150-1,000 km (Melb-Adel/Syd/Hob)"),
    (1000, 2000, "1,000-2,000 km (Melb-Bris)"),
    (2000, np.inf, "Over 2,000 km (to or from Perth)"),
]
HOME_TRAVELLED = "Home side travelled too"
BAND_NAMES = [name for _, _, name in BANDS] + [HOME_TRAVELLED]

# Era panel: three travel groups, validated categorical colours (blue, orange, aqua); aqua is
# below 3:1 contrast on white, so every line is labelled directly as well as in the legend
ERA_GROUPS = {
    "Same city": ["Same city"],
    "Interstate, under 2,000 km": ["150-1,000 km (Melb-Adel/Syd/Hob)", "1,000-2,000 km (Melb-Bris)"],
    "Over 2,000 km": ["Over 2,000 km (to or from Perth)"],
}
ERA_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND_COLOUR = "#2a78d6"
INK = "#3d3d3a"
MUTED = "#73726c"
GRID = "#e4e3df"

GAMES_SQL = f"""
SELECT g.id AS game, s.year, ht.location AS home, at.location AS away, v.name AS venue,
       g.final_home_score - g.final_away_score AS margin, g.winning_team, g.home_team
FROM games g
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
JOIN teams ht ON ht.id = g.home_team
JOIN teams at ON at.id = g.away_team
JOIN venues v ON v.id = g.venue
WHERE NOT {finalRound()} AND s.year BETWEEN ? AND ?
"""


def parseYears(text):
    """"2024" -> (2024, 2024); "1990-2026" -> (1990, 2026)."""
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def loadGames(years):
    db = createReaderConnection()
    cursor = db.execute(GAMES_SQL, years)
    games = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    db.close()
    games = games[~games["year"].isin(HUB_SEASONS)].reset_index(drop=True)
    games["year"] = games["year"].astype(int)
    games["margin"] = games["margin"].astype(float)
    games["home_km"] = [travelKm(t, y, v) for t, y, v in zip(games["home"], games["year"], games["venue"])]
    games["away_km"] = [travelKm(t, y, v) for t, y, v in zip(games["away"], games["year"], games["venue"])]
    games["home_won"] = np.where(games["winning_team"].isna(), 0.5,
                                 (games["winning_team"] == games["home_team"]).astype(float))
    games["band"] = [travelBand(h, a) for h, a in zip(games["home_km"], games["away_km"])]
    return games


def travelBand(home_km, away_km):
    if home_km >= AT_HOME_KM:
        return HOME_TRAVELLED
    for low, high, name in BANDS:
        if low <= away_km < high:
            return name


def fitMargins(games, columns):
    """Least squares: home margin = home team-season strength - away team-season strength +
    columns @ effects. Returns the effects and their standard errors."""
    home = games["home"] + "_" + games["year"].astype(str)
    away = games["away"] + "_" + games["year"].astype(str)
    team_seasons = {ts: i for i, ts in enumerate(sorted(set(home) | set(away)))}
    design = np.zeros((len(games), len(team_seasons) + columns.shape[1]))
    rows = np.arange(len(games))
    design[rows, home.map(team_seasons)] += 1
    design[rows, away.map(team_seasons)] -= 1
    design[:, len(team_seasons):] = columns
    margin = games["margin"].to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, margin, rcond=None)
    residuals = margin - design @ coefficients
    variance = residuals @ residuals / (len(margin) - np.linalg.matrix_rank(design))
    covariance = np.linalg.pinv(design.T @ design) * variance
    k = columns.shape[1]
    return coefficients[-k:], np.sqrt(np.diag(covariance)[-k:])


def bandColumns(games):
    return pd.get_dummies(games["band"]).reindex(columns=BAND_NAMES, fill_value=0).to_numpy(float)


def drawGraph(bands, eras, label, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: home margin by travel band, with 95% intervals
    positions = np.arange(len(bands))[::-1]
    left.axvline(0, color=MUTED, linewidth=1, zorder=1)
    left.hlines(positions, bands["low"], bands["high"], color=BAND_COLOUR, linewidth=2, zorder=2)
    left.scatter(bands["margin"], positions, s=64, color=BAND_COLOUR, edgecolors="white",
                 linewidths=2, zorder=3)
    for y, (_, row) in zip(positions, bands.iterrows()):
        left.annotate(f"{row['margin']:+.1f}", (row["high"], y), xytext=(6, 0),
                      textcoords="offset points", va="center", color=INK, fontsize=9)
    left.set_yticks(positions)
    left.set_yticklabels([f"{name}\n{int(n):,} games" for name, n in zip(bands.index, bands["games"])],
                         fontsize=9)
    left.set_xlabel("Home margin allowing for team strength (points, 95% interval)")
    left.set_title("Home advantage by the away side's trip", fontsize=11)

    # Right: travel groups by era
    x = np.arange(len(eras))
    for (group, colour) in zip(ERA_GROUPS, ERA_COLOURS):
        right.plot(x, eras[group], color=colour, linewidth=2, solid_capstyle="round",
                   marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2,
                   label=group, zorder=3)
        right.annotate(group, (x[-1], eras[group].iloc[-1]), xytext=(10, 0),
                       textcoords="offset points", va="center", color=INK, fontsize=9)
    right.axhline(0, color=MUTED, linewidth=1, zorder=1)
    right.set_xticks(x)
    right.set_xticklabels(eras.index, fontsize=9)
    right.set_xlim(-0.3, len(eras) - 1 + 1.6)  # room for the end labels
    right.set_ylabel("Home margin allowing for team strength (points)")
    right.set_title("The travel edge by era", fontsize=11)
    right.legend(frameon=False, loc="upper right", fontsize=9)

    for ax in (left, right):
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Travel and home advantage, home and away games {label}: being interstate "
                 "matters, extra distance much less", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Does travel distance affect game outcomes?")
    parser.add_argument("--years", type=parseYears, default=(1990, 2026),
                        help="Season or range of seasons (default 1990-2026)")
    parser.add_argument("--era-length", type=int, default=10,
                        help="Seasons per era in the era comparison (default 10)")
    args = parser.parse_args()

    games = loadGames(args.years)
    first, last = args.years
    label = f"{first}-{last}" if first != last else str(first)
    print(f"{len(games):,} home and away games, {label}"
          + (" (2020 hub season left out)" if first <= 2020 <= last else ""))

    effects, errors = fitMargins(games, bandColumns(games))
    rows = []
    print("\nHome margin by how far the away side travelled (home side at home unless stated), "
          "allowing for both teams' strength that season:")
    print(f"  {'':<36} {'games':>6} {'raw margin':>10} {'home won':>8} {'adjusted margin':>18}")
    for name, effect, error in zip(BAND_NAMES, effects, errors):
        subset = games[games["band"] == name]
        if subset.empty:
            continue
        rows.append({"band": name, "games": len(subset), "margin": effect,
                     "low": effect - 1.96 * error, "high": effect + 1.96 * error})
        print(f"  {name:<36} {len(subset):>6,} {subset['margin'].mean():>+10.1f} "
              f"{subset['home_won'].mean():>8.0%} {effect:>+10.1f} +/- {1.96 * error:.1f}")
    bands = pd.DataFrame(rows).set_index("band")

    extra = (games["away_km"] - games["home_km"]) / 1000
    effects, errors = fitMargins(games, np.column_stack([np.ones(len(games)), extra]))
    print(f"\nStraight line: home margin = {effects[0]:+.1f} (+/- {1.96 * errors[0]:.1f}) "
          f"{effects[1]:+.1f} (+/- {1.96 * errors[1]:.1f}) per 1,000 km the away side travelled "
          "further than the home side")
    interstate = games[(games["home_km"] < AT_HOME_KM) & (games["away_km"] >= 150)]
    effects, errors = fitMargins(interstate, np.column_stack(
        [np.ones(len(interstate)), interstate["away_km"] / 1000]))
    print(f"Interstate trips only ({len(interstate):,} games): {effects[1]:+.1f} "
          f"(+/- {1.96 * errors[1]:.1f}) points per extra 1,000 km")

    print("\nBy era (adjusted home margin):")
    era_rows = {}
    for start in range(first, last + 1, args.era_length):
        end = min(start + args.era_length - 1, last)
        era = games[games["year"].between(start, end)]
        if era.empty:
            continue
        effects, _ = fitMargins(era, bandColumns(era))
        by_band = dict(zip(BAND_NAMES, effects))
        # Named by the seasons it actually holds (2020 is left out)
        seasons = era["year"]
        name = (f"{seasons.min()}-{seasons.max()}" if seasons.min() != seasons.max()
                else str(seasons.min()))
        print(f"  {name}: " + ", ".join(f"{band.split(' (')[0]} {by_band[band]:+.1f}"
                                        for band in BAND_NAMES if (era["band"] == band).any()))
        # Era groups: game-weighted average of their bands
        counts = era["band"].value_counts()
        era_rows[name] = {group: np.average([by_band[b] for b in members],
                                            weights=[counts.get(b, 0) for b in members])
                          for group, members in ERA_GROUPS.items()
                          if sum(counts.get(b, 0) for b in members) > 0}
    eras = pd.DataFrame(era_rows).T

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"travelDistance_{label}.png")
    drawGraph(bands, eras, label, path)
    print(f"\nGraph saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
