#!/usr/bin/env python3

# Shared data preparation and reporting for the positional clustering scripts.
#
# What it does: loads every player's games for one or more seasons, joins the basic stats in
# player_game_stats with the advanced afl.com.au stats in aflcom_player_game_stats (afl.com.au's
# time on ground is used where it exists), and turns each player-season into one row of
# features:
#   - counting stats (kicks, pressure acts, metres gained, ...) as a rate per full game played,
#     i.e. season total / season time on ground, so a player who spends half of each game on the
#     bench isn't mistaken for a low-volume role
#   - percentage and ratio stats (disposal efficiency, hitout win %, ...) as a season average
#     weighted by time on ground
#   - per-game values that shouldn't be scaled by game time (time on ground itself, interchanges,
#     age, captaincy, sub on/off, wins) as plain season averages
#   - height and weight, and the share of games the player was named in each team-sheet position
# Only the features switched on in features.ini (or the file given with --features) are kept,
# and each is then standardised into z-scores within its season.
#
# Why it matters for the analysis: clustering is only as good as the space the players are placed
# in. Aggregating over a whole season smooths out single-game noise. Scaling by time on ground
# separates role from minutes played. Z-scores put stats measured in metres, percentages and
# centimetres on the same scale, so no stat dominates just because its numbers are bigger. Doing
# all of this in one place means every clustering method sees exactly the same data, so their
# results can be compared fairly.

import argparse
import configparser
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbConnection import createReaderConnection  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.ini")

# Counting stats from player_game_stats (filled from afl.com.au where afl.com.au has the game).
# games/wins/draws/losses and the *_to_date/*_with_club columns are career counters, not stats
# from the game, so they aren't used.
BASIC_COUNT_STATS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "hitouts", "tackles",
    "rebound_50s", "inside_50s", "clearances", "clangers", "frees_for", "frees_against",
    "brownlow_votes", "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assists",
]

# Counting stats only afl.com.au records (2012 onwards; some start later)
AFLCOM_COUNT_STATS = [
    "super_goals", "total_possessions", "dream_team_points", "rating_points", "turnovers",
    "intercepts", "tackles_inside_50", "shots_at_goal", "score_involvements", "metres_gained",
    "centre_clearances", "stoppage_clearances", "effective_kicks", "effective_disposals",
    "marks_on_lead", "intercept_marks", "hitouts_to_advantage", "ground_ball_gets",
    "f50_ground_ball_gets", "score_launches", "pressure_acts", "def_half_pressure_acts", "spoils",
    "ruck_contests", "contest_def_one_on_ones", "contest_def_losses", "contest_off_one_on_ones",
    "contest_off_wins", "centre_bounce_attendances", "kickins", "kickins_playon",
]

# Percentages and ratios: averaged over the season (weighted by time on ground), not scaled
AFLCOM_RATE_STATS = [
    "disposal_efficiency", "goal_accuracy", "goal_efficiency", "shot_efficiency",
    "kick_efficiency", "kick_to_handball_ratio", "contested_possession_rate",
    "hitout_win_percentage", "hitout_to_advantage_rate", "contest_def_loss_percentage",
    "contest_off_wins_percentage",
]

# Per-game values averaged over the season as they are
GAME_AVERAGES = [
    "time_on_ground_percentage", "time_on_ground_seconds", "time_on_bench_seconds",
    "interchange_count", "power_rating", "age", "captain", "subbed_on", "subbed_off", "won",
]

# Named team-sheet positions, with left and right sides merged: which pocket a player stands in
# says nothing about their role, and keeping the sides apart would split clusters by side
NAMED_POSITION_GROUPS = {
    "FF": "FF", "FPL": "FP", "FPR": "FP", "CHF": "CHF", "HFFL": "HFF", "HFFR": "HFF",
    "C": "C", "WL": "W", "WR": "W", "RK": "RK", "R": "R", "RR": "RR",
    "CHB": "CHB", "HBFL": "HBF", "HBFR": "HBF", "FB": "FB", "BPL": "BP", "BPR": "BP",
    "INT": "INT", "SUB": "SUB",
}

PLAYER_ATTRIBUTES = ["height", "weight"]

ALL_FEATURES = (
    BASIC_COUNT_STATS + AFLCOM_COUNT_STATS + AFLCOM_RATE_STATS + GAME_AVERAGES + PLAYER_ATTRIBUTES
    + [f"named_{group}" for group in sorted(set(NAMED_POSITION_GROUPS.values()))]
)

GAMES_SQL = f"""
SELECT
  s.year, pgs.player, pgs.team, pgs.game,
  p.first_name, p.last_name, p.aflcom_position,
  NULLIF(p.height, 0) AS height, NULLIF(p.weight, 0) AS weight,
  COALESCE(t.aflcom_abbreviation, t.short_name, t.location) AS team_name,
  COALESCE(a.time_on_ground_percentage, pgs.percentage_played) AS tog,
  a.position AS named_position,
  CASE WHEN a.captain THEN 1.0 WHEN a.captain IS NULL THEN NULL ELSE 0.0 END AS captain,
  CASE WHEN pgs.subbed_on THEN 1.0 WHEN pgs.subbed_on IS NULL THEN NULL ELSE 0.0 END AS subbed_on,
  CASE WHEN pgs.subbed_off THEN 1.0 WHEN pgs.subbed_off IS NULL THEN NULL ELSE 0.0 END AS subbed_off,
  CASE WHEN g.winning_team = pgs.team THEN 1.0 ELSE 0.0 END AS won,
  pgs.age_years + pgs.age_days / 365.25 AS age,
  {", ".join("pgs." + c for c in BASIC_COUNT_STATS)},
  {", ".join("a." + c for c in AFLCOM_COUNT_STATS + AFLCOM_RATE_STATS)},
  a.time_on_ground_percentage, a.time_on_ground_seconds, a.time_on_bench_seconds,
  a.interchange_count, a.power_rating
FROM player_game_stats pgs
JOIN games g ON g.id = pgs.game
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
JOIN players p ON p.id = pgs.player
JOIN teams t ON t.id = pgs.team
LEFT JOIN aflcom_player_game_stats a ON a.player = pgs.player AND a.game = pgs.game
WHERE s.year BETWEEN ? AND ? {{finals}}
"""


def parseYears(text):
    """"2025" -> (2025, 2025); "2021-2025" -> (2021, 2025)."""
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def addCommonArguments(parser):
    """The data preparation options every clustering script shares."""
    parser.add_argument(
        "--years", type=parseYears, default=parseYears("2021-2025"),
        help="Season, or range of seasons pooled together, e.g. 2025 or 2021-2025 (default "
        "2021-2025: the seasons with every afl.com.au stat; one season of ~550 players is "
        "often too few to tell apart more than a handful of roles)",
    )
    parser.add_argument(
        "--min-games", type=int, default=5,
        help="Only cluster players with at least this many games in the season (default 5)",
    )
    parser.add_argument(
        "--include-finals", action="store_true",
        help="Include finals (default: home and away only, as for the Brownlow)",
    )
    parser.add_argument(
        "--max-missing", type=float, default=0.5,
        help="Drop a feature missing for more than this share of players (default 0.5); "
        "the rest of the missing values are treated as the season average",
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.9,
        help="Cluster on the principal components explaining this share of the variance "
        "(default 0.9); 1 clusters on the z-scores directly",
    )
    parser.add_argument(
        "--drop", nargs="*", default=[], metavar="FEATURE",
        help="Features to leave out, e.g. --drop bounces kickins",
    )
    parser.add_argument(
        "--features", default=FEATURES_FILE,
        help="File listing the stats to use (default features.ini next to these scripts)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    return parser


def loadFeatureSelection(path):
    """The features switched on in a features file: every stat listed, in any section, that
    isn't commented out with #."""
    config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=("#",),
                                       delimiters=("=",))
    config.optionxform = str  # keep named_FF etc. in their original case
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config.read_file(fh)
    except OSError as error:
        sys.exit(f"Can't read the features file: {error}")
    selected = [option for section in config.sections() for option in config.options(section)]

    unknown = [f for f in selected if f not in ALL_FEATURES]
    if unknown:
        sys.exit(f"Unknown stats in {path}: {', '.join(unknown)}\n"
                 f"Known stats: {', '.join(ALL_FEATURES)}")
    if not selected:
        sys.exit(f"No stats are switched on in {path}")
    return selected


def loadGames(years, include_finals=False):
    """One row per player per game for the seasons."""
    db = createReaderConnection()
    sql = GAMES_SQL.replace("{finals}", "" if include_finals else "AND NOT r.final")
    cursor = db.execute(sql, years)
    columns = [d[0] for d in cursor.description]
    games = pd.DataFrame(cursor.fetchall(), columns=columns)
    db.close()
    numeric = [c for c in columns if c not in ("first_name", "last_name", "aflcom_position",
                                               "team_name", "named_position")]
    games[numeric] = games[numeric].apply(pd.to_numeric, errors="coerce")
    return games


def seasonFeatures(games):
    """One row per player-season: info columns, and the raw (unstandardised) features."""
    # Share of a full game played; games with no time on ground recorded count as full games
    games = games.copy()
    games["game_share"] = (games["tog"] / 100).fillna(1.0)

    keys = ["year", "player"]
    grouped = games.groupby(keys)

    info = grouped.agg(
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        listed_position=("aflcom_position", "first"),
        games=("game", "count"),
    )
    # The club the player played the most games for that season
    info["team"] = (
        games.groupby(keys + ["team_name"]).size().reset_index(name="n")
        .sort_values("n").groupby(keys).last()["team_name"]
    )
    info["name"] = info["first_name"] + " " + info["last_name"]

    features = {}
    # Counting stats: season total / full games played, only over games where it was recorded
    for column in BASIC_COUNT_STATS + AFLCOM_COUNT_STATS:
        recorded = games[column].notna()
        share = games["game_share"].where(recorded, 0.0)
        totals = games[column].groupby([games["year"], games["player"]]).sum(min_count=1)
        shares = share.groupby([games["year"], games["player"]]).sum()
        features[column] = totals / shares.where(shares > 0)

    # Percentages and ratios: time on ground weighted average
    for column in AFLCOM_RATE_STATS:
        weight = games["game_share"].where(games[column].notna(), 0.0)
        weighted = (games[column] * weight).groupby([games["year"], games["player"]]).sum(min_count=1)
        weights = weight.groupby([games["year"], games["player"]]).sum()
        features[column] = weighted / weights.where(weights > 0)

    for column in GAME_AVERAGES:
        features[column] = grouped[column].mean()

    for column in PLAYER_ATTRIBUTES:
        features[column] = grouped[column].first()

    # Share of games named in each position (only for games afl.com.au has a team sheet for)
    named = games["named_position"].map(NAMED_POSITION_GROUPS)
    has_sheet = named.notna().groupby([games["year"], games["player"]]).sum()
    for group in sorted(set(NAMED_POSITION_GROUPS.values())):
        count = (named == group).groupby([games["year"], games["player"]]).sum()
        features[f"named_{group}"] = count / has_sheet.where(has_sheet > 0)

    return info, pd.DataFrame(features)


def standardise(raw, years, max_missing=0.5, drop=None):
    """Z-scores within each season, after dropping unusable features. Missing values become 0
    (the season average). `drop` maps features to leave out to the reason why. Returns the
    z-scores and a list of (feature, reason) dropped."""
    drop = drop or {}
    raw = raw.drop(columns=[c for c in drop if c in raw.columns])
    dropped = list(drop.items())

    missing = raw.isna().mean()
    too_sparse = missing[missing > max_missing].index
    dropped += [(c, f"missing for {missing[c]:.0%} of players") for c in too_sparse]
    raw = raw.drop(columns=too_sparse)

    means = raw.groupby(years).transform("mean")
    stds = raw.groupby(years).transform("std")
    z = ((raw - means) / stds.where(stds > 0)).fillna(0.0)

    constant = z.columns[(z == 0).all()]
    dropped += [(c, "the same for every player") for c in constant]
    return z.drop(columns=constant), dropped


def prepare(args):
    """Load, aggregate, filter and standardise. Returns (info, z) indexed by (year, player)."""
    first, last = args.years
    print(f"Loading {first}{f'-{last}' if last != first else ''} "
          f"({'including' if args.include_finals else 'excluding'} finals)...")
    games = loadGames(args.years, args.include_finals)
    if games.empty:
        sys.exit("No games found for those seasons")
    info, raw = seasonFeatures(games)

    keep = info["games"] >= args.min_games
    info, raw = info[keep], raw[keep]
    print(f"{len(info)} player-seasons with at least {args.min_games} games "
          f"({(~keep).sum()} below the minimum)")

    selected = loadFeatureSelection(args.features)
    unselected = [c for c in raw.columns if c not in selected]
    print(f"Using {len(selected)} of {len(ALL_FEATURES)} stats from {args.features}"
          + (f" (switched off: {', '.join(unselected)})" if unselected else ""))
    drop = {c: "switched off" for c in unselected}
    drop.update({c: "--drop" for c in args.drop})
    z, dropped = standardise(raw, info.index.get_level_values("year"), args.max_missing, drop)
    for column, reason in dropped:
        if reason != "switched off":
            print(f"  dropped {column}: {reason}")
    print(f"{z.shape[1]} features, standardised to z-scores within each season")
    return info, z


def reduce(z, variance, seed=0):
    """Principal components explaining `variance` of the z-scores (or the z-scores themselves
    when variance >= 1). Clustering on fewer, uncorrelated dimensions stops groups of stats that
    move together (kicks, disposals, uncontested possessions, ...) counting several times over."""
    if variance >= 1:
        return z.to_numpy()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    components = pca.fit_transform(z.to_numpy())
    print(f"PCA: {pca.n_components_} components explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance")
    return components


def outputDirectory(method, args):
    first, last = args.years
    name = f"{method}_{first}" + (f"-{last}" if last != first else "")
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def playerLabel(info, index):
    year, _ = index
    row = info.loc[index]
    return f"{row['name']} ({row['team']}{f' {year}' if info.index.get_level_values('year').nunique() > 1 else ''})"


def describeClusters(info, z, labels, space, out_dir, examples=6, top=6):
    """Print each cluster's size, its most distinctive stats (mean z-score), its most typical
    players (closest to the cluster centre), and the listed positions in it. Saves the cluster
    profiles and assignments as CSV. Label -1 is treated as noise (unclustered players)."""
    labels = np.asarray(labels)
    profiles = z.groupby(labels).mean()
    profiles.index.name = "cluster"

    for cluster in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster)
        title = "Unclustered (noise)" if cluster == -1 else f"Cluster {cluster}"
        print(f"\n=== {title}: {len(members)} player-seasons ===")
        profile = profiles.loc[cluster].sort_values()
        high = ", ".join(f"{c} {v:+.1f}" for c, v in profile[::-1][:top].items())
        low = ", ".join(f"{c} {v:+.1f}" for c, v in profile[: top // 2].items())
        print(f"  high: {high}")
        print(f"  low:  {low}")

        centre = space[members].mean(axis=0)
        typical = members[np.argsort(((space[members] - centre) ** 2).sum(axis=1))[:examples]]
        print("  typical: " + "; ".join(playerLabel(info, info.index[i]) for i in typical))

        positions = info.iloc[members]["listed_position"].fillna("unknown").value_counts()
        print("  listed:  " + ", ".join(f"{p} {n}" for p, n in positions.items()))

    profiles.to_csv(os.path.join(out_dir, "cluster_profiles_zscores.csv"))
    assignments = info[["name", "team", "games", "listed_position"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(os.path.join(out_dir, "assignments.csv"))
    return assignments


def compareToListedPositions(info, labels, out_dir):
    """How well the clusters line up with afl.com.au's listed positions. These are each player's
    current listed position, not necessarily the one they had that season, and disagreement is
    expected: finding where listed positions don't match the stats is part of the point."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    listed = info["listed_position"]
    known = listed.notna().to_numpy() & (np.asarray(labels) != -1)
    if not known.any():
        print("\nNo listed positions to compare against")
        return
    table = pd.crosstab(np.asarray(labels)[known], listed[known].to_numpy(),
                        rownames=["cluster"], colnames=["listed"])
    table.to_csv(os.path.join(out_dir, "clusters_vs_listed_positions.csv"))
    print("\nClusters vs listed positions:")
    print(table.to_string())
    print(f"Adjusted Rand index {adjusted_rand_score(listed[known], np.asarray(labels)[known]):.3f}, "
          f"normalised mutual information "
          f"{normalized_mutual_info_score(listed[known], np.asarray(labels)[known]):.3f} "
          "(1 = identical to listed positions, 0 = no better than chance)")


def newParser(description):
    return addCommonArguments(argparse.ArgumentParser(description=description))
