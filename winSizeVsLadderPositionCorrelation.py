#!/usr/bin/env python

# TODO Account for draws

# Correlation between a game's winning margin and the gap between the two teams' ladder positions
# going into the game, for one season or several pooled together.
#
# Each team's position is from the ladder at the time of the game: its latest ladder entry from an
# earlier round of the same season (ladder_entries holds the ladder after each round, including
# rounds where the team had a bye). A team's first game of the season has no ladder yet, so those
# games are left out. Finals use the ladder at the end of the home and away season.
#
# Usage: venv/bin/python winSizeVsLadderPositionCorrelation.py <years> [<years> ...]
#   where each <years> is a season (2024) or a range of seasons (2012-2026), e.g. 2012-2015 2020
# The graph is shown and saved to output/winSizeVsLadderPosition_<years>.png. Each game is coloured
# by its round, light early in the season (when the ladder says little) to dark late in it, with
# finals darkest. Finals share one colour because their round numbers differ between seasons.

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import numpy as np
from scipy import stats
from dbConnection import createReaderConnection, finalRound

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# One blue, light (early rounds) to dark (late rounds); the lightest step still stands out from white
ROUND_COLOURS = LinearSegmentedColormap.from_list(
    "rounds", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
INK = "#3d3d3a"
GRID = "#e4e3df"

# The team's position on the ladder after the latest earlier round of the game's season
LADDER_BEFORE_GAME = """(SELECT le.position FROM ladder_entries le INNER JOIN rounds lr ON lr.id = le.round
 WHERE le.team = {team} AND lr.season = r.season AND lr.sequence < r.sequence
 ORDER BY lr.sequence DESC LIMIT 1)"""

def parseYears(years):
    """Seasons from a year (2024), a range ("2012-2026") or a list of either."""
    if isinstance(years, (int, np.integer)):
        return [int(years)]
    if isinstance(years, str):
        first, _, last = years.partition("-")
        return list(range(int(first), int(last or first) + 1))
    return sorted({year for item in years for year in parseYears(item)})

def yearsLabel(years):
    """2024, 2012-2026 or 2012-2015, 2020: consecutive seasons written as ranges."""
    groups = []
    for year in years:
        if groups and year == groups[-1][1] + 1:
            groups[-1][1] = year
        else:
            groups.append([year, year])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)

def genCorrolationScore(year, show_graph=False, output=False, db=None):
    """r between winning margin and ladder position gap. `year` is a season, a range such as
    "2012-2026", or a list of either; the seasons are pooled together."""
    years = parseYears(year)
    own_connection = db is None
    if own_connection:
        db = createReaderConnection()

    margins = []
    position_differences = []
    rounds = []
    finals = []
    round_names = {}

    sql = ("SELECT * FROM (SELECT games.winning_score, games.losing_score, r.sequence, r.identifier, " + finalRound() + " AS final, "
           + LADDER_BEFORE_GAME.format(team="games.winning_team") + " AS winning_position, "
           + LADDER_BEFORE_GAME.format(team="games.losing_team") + " AS losing_position "
           "FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season "
           f"WHERE s.year IN ({', '.join('?' * len(years))}) AND games.winning_team IS NOT NULL "
           "ORDER BY games.start_datetime ASC) x "
           "WHERE x.winning_position IS NOT NULL AND x.losing_position IS NOT NULL;")
    results = db.fetchall(sql, years)

    if own_connection:
        db.close()

    for res in results:
        winning_score, losing_score, sequence, round_name, final, winning_position, losing_position = res
        margins.append(winning_score - losing_score)
        position_differences.append(losing_position - winning_position)
        rounds.append(sequence)
        finals.append(bool(final))
        if not final:
            round_names.setdefault(sequence, round_name)

    x = np.array(position_differences)
    y = np.array(margins)

    scp_res = stats.linregress(x, y)

    if output:
        print("Seasons:", yearsLabel(years))
        print("Games:", len(x))
        print("Scipy r value:", scp_res.rvalue)
        print("Line of best fit: ", str(scp_res.slope) + "*x + " + str(scp_res.intercept))

        if show_graph:
            # Colour by round; every final takes the step after the last home and away round
            finals = np.array(finals)
            last_round = max(round_names)
            colour = np.where(finals, last_round + 1, np.array(rounds))

            fig, ax = plt.subplots(figsize=(9, 6))
            many = len(x) > 500
            order = np.argsort(colour, kind="stable")  # later rounds drawn on top
            points = ax.scatter(x[order], y[order], c=colour[order], cmap=ROUND_COLOURS,
                                s=18 if many else 48, edgecolors="white",
                                linewidths=0.6 if many else 1.5, zorder=3)
            line_x = np.array([x.min(), x.max()])
            ax.plot(line_x, scp_res.slope*line_x + scp_res.intercept, color=INK, linewidth=2,
                    solid_capstyle="round", zorder=4,
                    label=f"Line of best fit (r = {scp_res.rvalue:.2f}, {len(x)} games)")

            colourbar = fig.colorbar(points, ax=ax, pad=0.02)
            colourbar.set_label("Round (lighter = earlier, when ladder positions mean less)")
            sequences = sorted(round_names)
            ticks = sequences[::max(1, len(sequences) // 7)]
            if sequences[-1] not in ticks:
                ticks.append(sequences[-1])
            labels = [round_names[t] for t in ticks]
            if finals.any():
                ticks.append(last_round + 1)
                labels.append("Finals")
            colourbar.set_ticks(ticks)
            colourbar.set_ticklabels(labels)
            colourbar.outline.set_visible(False)

            label = yearsLabel(years)
            ax.set_title(f"Margins by Position Differences ({label}, ladder at the time of the game)")
            ax.set_ylabel("Margin")
            ax.set_xlabel("Difference of Position between Winning Team vs Losing Team")
            ax.grid(color=GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"winSizeVsLadderPosition_{label.replace(', ', '_')}.png")
            plt.savefig(path, dpi=150)
            print("Graph saved to", path)
            plt.show()

    return float(scp_res.rvalue)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: winSizeVsLadderPositionCorrelation.py <year or range> [<year or range> ...]")
    genCorrolationScore(sys.argv[1:], show_graph=True, output=True)
