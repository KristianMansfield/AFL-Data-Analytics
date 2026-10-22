#!/usr/bin/env python3

# Do any umpires favour (or disfavour) particular teams, beyond what random chance would produce?
#
# What it does: for every home and away game from --first-year on, works out how each team did
# compared with what was expected, then checks whether those differences line up with which field
# umpires were on the game.
#   1. Expected values. A team's free kick differential, margin and chance of winning depend mostly
#      on how good it (and its opponent) were that season and on home ground advantage. These are
#      fitted by least squares over every game: one strength per team per season, and one home
#      advantage per team (travelling interstate matters more for some teams than others). Free
#      kick differential and margin each get their own fit; the chance of winning comes from the
#      expected margin and the spread of margins around it. What's left over for each game, the
#      residual, is how much better or worse the home side did than expected (the away side's is
#      the same with the sign flipped).
#   2. Umpire x team pairs. For every field umpire and team with at least --min-games games
#      together, the team's residuals in those games are added up and turned into a z-score (how
#      many standard errors from zero), with a two-sided p-value.
#   3. Many tests at once. With over a thousand pairs and three measures, about 5% of pairs will
#      look "significant" at p < 0.05 by chance alone. So the script also reports:
#        - false discovery rate (Benjamini-Hochberg) q-values: a pair with q < 0.05 is flagged
#          with at most 5% of flagged pairs expected to be false alarms
#        - family-wise adjusted p-values by permutation (Westfall-Young max-T): the crews of the
#          games in each season are shuffled between games --permutations times, and each pair's
#          |z| is compared with the largest |z| of any pair in each shuffle. A pair with adjusted
#          p < 0.05 stands out even allowing for every pair that was tested.
#        - an overall test: whether the pairs' z-scores as a whole spread out more than in the
#          shuffles (sum of z squared), i.e. whether there's any umpire-team effect at all, even if
#          no single pair can be singled out
#      Shuffling crews within a season keeps everything else fixed (the games, the results, how
#      often each umpire officiates), and it automatically allows for each game having three or
#      four umpires, which makes pairs that share games move together.
#
# Why it matters for the analysis: a team's record under one umpire will always differ from its
# record overall, simply because of which games that umpire happened to get. The question is
# whether any difference is bigger than chance can explain, once the strength of the teams and home
# ground are allowed for and once the number of comparisons is taken into account.
#
# What it can't show: umpires aren't assigned to games at random. If, say, senior umpires get the
# big games, an effect that looks like an umpire bias could come from the kind of game instead.
# Finals are left out for that reason. A pair flagged here is a reason to look closer, not proof of
# bias.
#
# Outputs (in output/): pairs_<measure>.csv with every umpire-team pair tested.
#
# Usage: ../venv/bin/python umpireTeamBias.py [--first-year 2012] [--min-games 10]
#        [--permutations 1000] [--seed 0] [--top 10]

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbConnection import createReaderConnection, finalRound  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

GAMES_SQL = f"""
SELECT g.id AS game, s.year, g.home_team, g.away_team, g.winning_team,
       g.final_home_score, g.final_away_score,
       home.frees_for AS home_frees_for, home.frees_against AS home_frees_against
FROM games g
JOIN rounds r ON r.id = g.round
JOIN seasons s ON s.id = r.season
JOIN team_game_stats home ON home.game = g.id AND home.team = g.home_team
WHERE s.year >= ? AND NOT {finalRound()} AND home.frees_for IS NOT NULL
ORDER BY g.start_datetime, g.id
"""

CREWS_SQL = """
SELECT ug.game, ug.umpire, u.first_name || ' ' || u.last_name AS umpire_name
FROM umpire_games ug
JOIN umpires u ON u.id = ug.umpire
WHERE ug.umpire_type = 'FIELD'
"""

TEAMS_SQL = "SELECT id, COALESCE(aflcom_abbreviation, short_name, location) FROM teams"

MEASURES = {
    "frees": "free kick differential",
    "margin": "margin (points)",
    "win": "wins",
}


def load(first_year):
    db = createReaderConnection()

    def query(sql, params=()):
        cursor = db.execute(sql, params)
        return pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])

    games = query(GAMES_SQL, (first_year,))
    crews = query(CREWS_SQL)
    teams = dict(db.fetchall(TEAMS_SQL))
    db.close()
    for column in games.columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")
    crews = crews[crews["game"].isin(games["game"])]
    games = games[games["game"].isin(crews["game"])].reset_index(drop=True)
    return games, crews, teams


def expectedValues(games):
    """Home-side residuals for free kick differential, margin and wins, after team-season
    strengths and per-team home advantage. Returns a frame of residuals and per-game standard
    deviations."""
    games = games.copy()
    games["home_ts"] = games["home_team"].astype(int).astype(str) + "_" + games["year"].astype(int).astype(str)
    games["away_ts"] = games["away_team"].astype(int).astype(str) + "_" + games["year"].astype(int).astype(str)
    team_seasons = sorted(set(games["home_ts"]) | set(games["away_ts"]))
    home_teams = sorted(games["home_team"].unique())
    ts_index = {ts: i for i, ts in enumerate(team_seasons)}
    ht_index = {t: len(team_seasons) + i for i, t in enumerate(home_teams)}

    rows = np.arange(len(games))
    design = np.zeros((len(games), len(team_seasons) + len(home_teams)))
    design[rows, games["home_ts"].map(ts_index)] += 1
    design[rows, games["away_ts"].map(ts_index)] -= 1
    design[rows, games["home_team"].map(ht_index)] = 1

    def fit(y):
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return design @ coefficients

    out = pd.DataFrame(index=games.index)
    frees = (games["home_frees_for"] - games["home_frees_against"]).to_numpy(dtype=float)
    expected_frees = fit(frees)
    out["frees"] = frees - expected_frees
    out["frees_sd"] = out["frees"].std()

    margin = (games["final_home_score"] - games["final_away_score"]).to_numpy(dtype=float)
    expected_margin = fit(margin)
    out["margin"] = margin - expected_margin
    out["margin_sd"] = out["margin"].std()

    # Chance of a home win from the expected margin and the spread of margins around it
    p_home = norm.cdf(expected_margin / out["margin_sd"].iloc[0])
    won = np.where(games["winning_team"].isna(), 0.5,
                   (games["winning_team"] == games["home_team"]).astype(float))
    out["win"] = won - p_home
    out["win_sd"] = np.sqrt(p_home * (1 - p_home))
    return out


def pairMatrices(games, crews):
    """Sparse games x umpires (1 if on the game) and games x teams (+1 home, -1 away) matrices."""
    umpires = np.sort(crews["umpire"].unique())
    teams = np.sort(np.union1d(games["home_team"].unique(), games["away_team"].unique()))
    game_row = pd.Series(np.arange(len(games)), index=games["game"])
    umpire_col = pd.Series(np.arange(len(umpires)), index=umpires)
    team_col = pd.Series(np.arange(len(teams)), index=teams)
    on_game = sparse.csr_matrix(
        (np.ones(len(crews)), (game_row[crews["game"]].to_numpy(), umpire_col[crews["umpire"]].to_numpy())),
        shape=(len(games), len(umpires)))
    rows = np.concatenate([np.arange(len(games))] * 2)
    columns = np.concatenate([team_col[games["home_team"]].to_numpy(), team_col[games["away_team"]].to_numpy()])
    signs = np.concatenate([np.ones(len(games)), -np.ones(len(games))])
    side = sparse.csr_matrix((signs, (rows, columns)), shape=(len(games), len(teams)))
    return on_game, side, umpires, teams


def pairZ(on_game, side, residual, variance):
    """z-score for every umpire x team pair: the team's summed residuals in the umpire's games over
    their standard error, and the number of games together."""
    totals = (on_game.T @ side.multiply(residual[:, None])).toarray()
    variances = (on_game.T @ abs(side).multiply(variance[:, None])).toarray()
    counts = (on_game.T @ abs(side)).toarray()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = totals / np.sqrt(variances)
    return z, totals, counts


def benjaminiHochberg(p):
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def shuffleCrews(on_game, years, rng):
    """The games x umpires matrix with whole crews moved between games of the same season."""
    order = np.arange(len(years))
    for year in np.unique(years):
        positions = np.flatnonzero(years == year)
        order[positions] = rng.permutation(positions)
    return on_game[order]


def main():
    parser = argparse.ArgumentParser(description="Test umpire x team pairs for bias beyond chance")
    parser.add_argument("--first-year", type=int, default=2012,
                        help="First season to include (default 2012)")
    parser.add_argument("--min-games", type=int, default=10,
                        help="Only test umpire-team pairs with at least this many games together "
                        "(default 10)")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Crew shuffles for the adjusted p-values (default 1000)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    parser.add_argument("--top", type=int, default=10, help="Pairs to list per measure")
    args = parser.parse_args()

    games, crews, team_names = load(args.first_year)
    residuals = expectedValues(games)
    on_game, side, umpires, teams = pairMatrices(games, crews)
    umpire_names = crews.drop_duplicates("umpire").set_index("umpire")["umpire_name"]
    years = games["year"].to_numpy()
    print(f"{len(games)} home and away games {int(years.min())}-{int(years.max())}, "
          f"{len(umpires)} field umpires, {len(teams)} teams")
    print(f"Residual spread per game: free kick differential {residuals['frees_sd'].iloc[0]:.1f}, "
          f"margin {residuals['margin_sd'].iloc[0]:.1f} points")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for measure, label in MEASURES.items():
        residual = residuals[measure].to_numpy()
        variance = (residuals[f"{measure}_sd"] ** 2).to_numpy()
        z, totals, counts = pairZ(on_game, side, residual, variance)
        tested = counts >= args.min_games
        z_tested = z[tested]
        p = 2 * norm.sf(np.abs(z_tested))
        q = benjaminiHochberg(p)

        # Shuffled crews: the largest |z| and the sum of z squared in each shuffle
        max_null, spread_null = [], []
        for _ in range(args.permutations):
            z_null, _, counts_null = pairZ(shuffleCrews(on_game, years, rng), side, residual, variance)
            keep = counts_null >= args.min_games
            max_null.append(np.nanmax(np.abs(z_null[keep])))
            spread_null.append(np.nansum(z_null[keep] ** 2) / keep.sum())
        max_null, spread_null = np.array(max_null), np.array(spread_null)
        adjusted = (1 + (max_null[None, :] >= np.abs(z_tested)[:, None]).sum(axis=1)) / (1 + len(max_null))
        spread = np.sum(z_tested ** 2) / len(z_tested)
        spread_p = (1 + (spread_null >= spread).sum()) / (1 + len(spread_null))

        u, t = np.nonzero(tested)
        pairs = pd.DataFrame({
            "umpire": umpire_names.reindex(umpires[u]).to_numpy(),
            "team": [team_names.get(int(team), team) for team in teams[t]],
            "games": counts[tested].astype(int),
            "mean_deviation": totals[tested] / counts[tested],
            "z": z_tested, "p": p, "fdr_q": q, "adjusted_p": adjusted,
        }).sort_values("p")
        pairs.to_csv(os.path.join(OUTPUT_DIR, f"pairs_{measure}.csv"), index=False, float_format="%.4f")

        unit = {"frees": "frees per game", "margin": "points per game", "win": "wins per game"}[measure]
        print(f"\n=== {label} ===")
        print(f"{len(pairs)} umpire-team pairs with {args.min_games}+ games together")
        print(f"  p < 0.05 before correction: {(p < 0.05).sum()} (about {0.05 * len(p):.0f} expected by chance)")
        print(f"  false discovery rate q < 0.05: {(q < 0.05).sum()}")
        print(f"  family-wise adjusted p < 0.05: {(adjusted < 0.05).sum()}")
        print(f"  overall spread of z-scores: {spread:.2f} (shuffled crews {spread_null.mean():.2f} "
              f"+/- {spread_null.std():.2f}), p = {spread_p:.3f}")
        print(f"\n  {'Umpire':<24} {'Team':<6} {'Games':>5} {'Deviation':>10} {'z':>6} {'p':>7} "
              f"{'FDR q':>6} {'Adj. p':>6}")
        for _, row in pairs.head(args.top).iterrows():
            print(f"  {row['umpire']:<24} {row['team']:<6} {row['games']:>5} "
                  f"{row['mean_deviation']:>+10.2f} {row['z']:>+6.2f} {row['p']:>7.4f} "
                  f"{row['fdr_q']:>6.2f} {row['adjusted_p']:>6.3f}")
        print(f"  (deviation: {unit} above expected for that team with that umpire)")

    print(f"\nWritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
