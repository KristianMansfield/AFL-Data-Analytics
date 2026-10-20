# Plackett-Luce Model

Predicts Brownlow Medal votes game by game with a ranked (Plackett-Luce) logit, then adds them up
into a season leaderboard with each player's range of likely totals and chance of winning. No
simulation: every probability is worked out exactly, so the same inputs always give the same
answer.

This folder also holds the data preparation and reporting code the other models in
`Brownlow Prediction` use (`gameData.py`, `rankedLogit.py`, `seasonTally.py`,
`predictionReport.py`).

## Scripts

| Script | What it does |
|---|---|
| [gameData.py](gameData.py) | Loads every player's home and away games from the database (basic stats from `player_game_stats`, advanced stats from `aflcom_player_game_stats`) and builds the features: game result, within-game ranks, scoring progression (including a small win probability model fitted on `score_progressions`), season form and reputation. Defines the feature groups the other scripts switch on and off. Shared by every model. |
| [rankedLogit.py](rankedLogit.py) | The vote model. Fits a ranked (Plackett-Luce) logit on the votes actually given, and for each game gives every player their chance of 3, 2 and 1 votes, their expected votes (a share of the game's 6 votes) and the model's single most likely 3-2-1. Also turns any model's Plackett-Luce impacts into those chances (used by the gradient-boosted ranker). |
| [brownlowModel.py](brownlowModel.py) | The full model: the ranked logit plus the umpire appeal feature, which needs a two-stage fit (see below). |
| [seasonTally.py](seasonTally.py) | Turns the game predictions into a season leaderboard: each player's distribution of season votes (combined exactly from their games), an 80% range, chance of winning, and round-by-round running totals for the leaders. Allows for a player's season-long "umpire appeal" spread (`--player-sd`). |
| [predictionReport.py](predictionReport.py) | Command line options, output files and the printed leaderboard shared by every model's predict script. |
| [ensembleModel.py](ensembleModel.py) | The optional blend with the gradient-boosted ranker (`--ensemble`): each player's impact is 70% this model and 30% the LightGBM ListMLE ranker. |
| [predictBrownlow.py](predictBrownlow.py) | Command line: fits the model on the training seasons and predicts one season. If that season's votes are known, shows them alongside. |
| [backtest.py](backtest.py) | Command line: predicts each season from earlier seasons only, for several versions of the model with feature groups switched on and off, and compares them. Also checks how honest the season ranges are for different `--player-sd` values. |

## Usage

```
cd "Brownlow Prediction/Plackett-Luce Model"
../../venv/bin/python predictBrownlow.py                       # latest season, trained on 2012 to the season before
../../venv/bin/python predictBrownlow.py --year 2025 --train-years 2012-2024
../../venv/bin/python predictBrownlow.py --groups basic advanced rating rank result   # pick feature groups
../../venv/bin/python predictBrownlow.py --basic-stats         # without afl.com.au stats, trained from 2003
../../venv/bin/python predictBrownlow.py --ensemble            # blend 70/30 with the gradient-boosted ranker
../../venv/bin/python backtest.py                              # ~5 minutes, tests 2015-2026
../../venv/bin/python backtest.py --basic-stats                # tests 2006-2026 (21 seasons)
../../venv/bin/python backtest.py --loso                       # train on every other season instead
../../venv/bin/python backtest.py --variants full "full - result"  # only some variants
```

`--basic-stats` leaves out the 15 features afl.com.au only records from 2012 (the advanced stats,
rating points, rating rank and rating form) and uses seasons from 2003, when every basic stat is
first recorded. That roughly doubles the number of test seasons. This matters for judging the
winner and top 10, because each season gives only one result: with 12 test seasons, one more
correctly picked winner is easily luck.

`--ensemble` blends each player's impact with the gradient-boosted ranker's (70% Plackett-Luce,
30% ranker; `--ensemble-weight` changes the share) and writes to `output/<year>-ensemble/`. In the
backtests the blend predicted the actual top 10's votes game by game better than this model alone
in 12 of 12 seasons (2015–2026) and 19 of 21 (2006–2026), but it wasn't clearly better at picking
the winner (mean rank 2.08 against 2.25 for 2015–2026, 2.62 against 2.43 for 2006–2026) or the
top 10 order (0.678 against 0.691, 0.618 against 0.609), so it's off by default. It needs LightGBM.

`predictBrownlow.py` writes to `output/<year>/`:

- `leaderboard.csv`: expected votes, 3-2-1 votes, 80% range, chance of winning and actual votes for every player
- `game_predictions.csv`: every player in every game, with P(3), P(2), P(1) and expected votes
- `round_by_round.csv`: running totals for the top 10 after each round
- `coefficients.csv`: what the model learned (the weight on each feature)

`backtest.py` writes `game_metrics.csv`, `season_metrics.csv` and `calibration.csv` to
`output/backtest/` (or `output/backtest_basic_stats/`).

## How it works

1. **Impact per player.** Each player in a game gets an impact score: a weighted sum of their
   features. The weights are learned from the votes actually given, so the data decides how much a
   kick, a win or a late goal is worth.
2. **Votes per game.** The 3 votes are modelled as going to one player in the game with
   probability proportional to exp(impact), the 2 votes to one of the remaining players, then the
   1 vote. Each player's chance of 3, 2 and 1 votes comes from adding up every possible top-three
   order. Only differences between players in the same game matter, which is how umpires vote:
   against the other players on the ground, not a fixed standard.
3. **Umpire appeal (two stages).** Some players poll more than their stats suggest year after
   year, and some less. A first model is fitted without this feature and used to predict the
   earlier seasons. Each player's actual minus expected votes over the previous three seasons,
   divided by (games + 10), becomes a feature for the final model. Only seasons before the one
   being predicted are ever used.
4. **Two ways to count.** *Expected votes* gives each player a share of the 6 votes in each game.
   *3-2-1* gives the model's top three in each game 3, 2 and 1. Expected votes is more accurate for
   season totals and gets the top 10 order right more often. 3-2-1 overstates the leaders.
5. **Season totals.** Each player's per-game chances are combined exactly into the distribution of
   their season total. Their chance of winning is P(they poll t) × P(everyone else polls t or
   fewer), added up over every t (ties count as wins, since joint winners get the medal).
6. **Ranges.** Umpires rate some players higher or lower than the model does all season, so its
   misses for a player lean the same way week after week. A normal shift to each player's impact
   (standard deviation `--player-sd`, default 0.3) widens the season ranges to allow for this. It's
   averaged over exactly with Gauss-Hermite quadrature rather than random draws.

## Features

By default 49 features in eight groups (`basic`, `advanced`, `rating`, `rank`, `result`,
`progression`, `form`, `appeal`), chosen by the backtests below for picking the winner and the top
10 in order.

| Group | Features |
|---|---|
| `basic` (21) | kicks, handballs, marks, goals, behinds, hitouts, tackles, rebound 50s, inside 50s, clearances, clangers, frees for, frees against, contested possessions, uncontested possessions, contested marks, marks inside 50, one percenters, bounces, goal assists, time on ground % |
| `advanced` (12) | metres gained, score involvements, intercepts, centre clearances, turnovers, effective disposals, pressure acts, ground ball gets, spoils, score launches, hitouts to advantage, contested possession rate |
| `rating` (1) | afl.com.au rating points |
| `rank` (3) | 1/rank of disposals in the game, 1/rank of rating points in the game, top goalkicker (most goals in the game, and at least 3) |
| `result` (2) | won (1, 0.5 for a draw, 0), margin for the player's side (in goals, capped at ±10) |
| `progression` (4) | win probability added by the player's scores, last-quarter goals, goals while the margin was within 2 goals, scored the goal or behind that put the winner in front for good |
| `form` (5) | the player's average disposals, goals, contested possessions, clearances and rating points over the rest of the season (every game but this one) |
| `appeal` (1) | umpire appeal: actual minus expected votes over the previous 3 seasons, per game (see How it works) |
| `reputation` (1, off by default) | votes per game over the previous 3 seasons, scaled down for players with few games. Umpire appeal replaced it: in the 21-season backtest, the model without reputation placed the winner higher (mean rank 2.43 against 2.52) and got the top 10 order slightly better (0.61 against 0.60). |
| `baseline` (yardstick only) | disposals, goals (the two stats `brownlowStats.py` plots) |

The advanced stats start in 2012, so training defaults to 2012 onwards. Scoring progression data
starts in 2001. Season form uses only stats, which are all known by count night, so a season's
votes never leak into its own predictions.

**Win probability added.** A small model gives the home side's chance of winning from the margin
and the share of the game left (the margin is scaled by the square root of the time left, so a
12-point lead with a quarter to go counts about the same as a 24-point lead at the start). Each
goal or behind is worth how much it changed the scoring side's chance of winning. A goal that puts a
side in front late is worth far more than one at three-quarter time of a 60-point win.

## Most impactful features

From the 2026 model (trained on 2012–2025). The multiplier is how much a one standard deviation
increase (a typical-sized difference between players) multiplies a player's chance of polling.

| Feature | ×odds per SD | 1 SD is about | Per unit |
|---|---|---|---|
| goals | **2.11** | 1 goal | ~×2.2 per goal |
| margin_for | **2.01** | 6 goals of margin | ~×1.12 per goal of margin |
| kicks | **1.99** | 4.7 kicks | ~×1.16 per kick |
| handballs | **1.81** | 4.3 handballs | ~×1.15 per handball |
| uncontested possessions | 1.44 | 5.3 | |
| won | 1.42 | (win vs loss) | ×2.0 for a win vs a loss |
| form: rating points | 1.28 | 3.1 season-average rating points | |
| tackles | 1.25 | 2.3 | |
| spoils | 1.24 | 2.1 | |
| marks | 1.21 | 2.6 | |
| contested possessions | 1.19 | 3.7 | |
| metres gained | 1.18 | 148 m | |
| hitouts | 1.18 | 6.6 | |
| scoring_wpa | 1.17 | 0.09 of win probability | |
| clearances | 1.17 | 2.2 | |
| top goalkicker | 1.17 | | |
| umpire appeal | 1.13 | 0.05 votes per game above expected | |
| form: goals | 0.85 | 0.6 season-average goals | |

Almost no effect (between ×0.95 and ×1.05): effective disposals, behinds, the sealing score, goal
assists, score launches, frees against, rating rank, inside 50s, turnovers, clangers, rebound 50s,
one percenters, form contested possessions, and the raw counts of late and close goals.

How to read these weights:

- **Overlapping stats split the credit.** Kicks + handballs = disposals, and contested +
  uncontested possessions measure roughly the same thing, so judge them as a group. Together,
  getting lots of the ball is the strongest signal in the model.
- **Winning counts twice.** Winning and the margin stack: a 6-goal win against a 6-goal loss is
  roughly ×1.4 × 2.0² ≈ ×5.7 on a player's odds.
- **Season form.** A player having a strong season overall (season-average rating points) polls
  more for the same game. Season-average goals and disposals come out negative once rating is in:
  for the same game and the same overall rating, a player who usually kicks a lot of goals or gets
  a lot of the ball polls a little less. One reading is that umpires reward a big game more when it
  stands out from the player's usual output, but the form features overlap with each other, so
  treat the group as a whole: together they improved the winner's rank and the top 10 order in the
  backtests.
- **Late goals.** Once win probability added is in, raw last-quarter and close-game goal counts
  turn slightly negative. That's the overlap, not umpires penalising late goals: the goals that
  count extra are the ones that swing the game.
- **Ruck stats are modest.** Midfield possessions count for much more than hitouts, which matches
  umpires' known preference for midfielders.

The weights are refitted each run, so they shift a little with the training seasons.

## Backtest results

Each season predicted from earlier seasons only. "Top 10 order" is the Spearman correlation
between the actual and predicted positions of the actual top 10 (1 = exactly the right order).
Season measures use expected votes unless stated.

### 2015–2026 (`backtest.py`)

| Version | Log-lik per game | 3-vote player picked | Winners picked | Winner's rank (mean / median) | Top 10 overlap | Top 10 order |
|---|---|---|---|---|---|---|
| disposals + goals | -6.217 | 48.7% | 5/12 | 6.42 / 2 | 5.67 | 0.46 |
| basic stats | -5.668 | 52.4% | 5/12 | 4.17 / 2 | 6.92 | 0.52 |
| + advanced, rating | -5.565 | 52.7% | 5/12 | 3.17 / 2 | 7.08 | 0.53 |
| + rank | -5.525 | 52.9% | 5/12 | 2.92 / 2 | 7.00 | 0.52 |
| + result | -5.003 | 58.4% | 6/12 | 2.58 / 1.5 | 7.25 | 0.63 |
| + progression | -4.998 | 59.0% | 6/12 | 2.50 / 1.5 | 7.33 | 0.63 |
| + form | -4.950 | 58.6% | 6/12 | 2.33 / 1.5 | 7.42 | 0.66 |
| **full** (+ umpire appeal) | **-4.906** | **59.2%** | 7/12 | 2.25 / **1** | 7.08 | **0.69** |
| full without result | -5.414 | 53.6% | 7/12 | 2.08 / 1 | 7.08 | 0.57 |
| full without progression | -4.910 | 58.3% | 8/12 | 2.17 / 1 | 7.08 | 0.69 |
| full without form | -4.943 | 59.0% | 6/12 | 2.42 / 1.5 | 6.92 | 0.69 |
| full + reputation | -4.907 | 59.2% | 7/12 | 2.17 / 1 | 7.00 | 0.69 |

Log-likelihood is the average log probability the model gave the 3-2-1 that actually happened
(higher is better; picking at random from about 44 players scores about -11.2).

### 2006–2026 (`backtest.py --basic-stats`)

Without the afl.com.au features (34 features), training from 2003.

| Version | Log-lik per game | 3-vote player picked | Winners picked | Winner's rank (mean / median) | Top 10 overlap | Top 10 order |
|---|---|---|---|---|---|---|
| disposals + goals | -6.506 | 45.4% | 7/21 | 6.24 / 3 | 5.81 | 0.37 |
| basic stats | -6.036 | 48.2% | 8/21 | 4.10 / 2 | 6.52 | 0.47 |
| + rank | -5.994 | 48.1% | 9/21 | 3.86 / 2 | 6.62 | 0.48 |
| + result | -5.470 | 54.5% | 9/21 | 3.67 / 3 | 6.57 | 0.56 |
| + progression | -5.465 | 54.6% | 9/21 | 3.62 / 3 | 6.62 | 0.55 |
| + form | -5.420 | 54.8% | 8/21 | 2.71 / 2 | 6.86 | 0.58 |
| **full** (+ umpire appeal) | **-5.372** | **55.4%** | 9/21 | **2.43 / 2** | 6.76 | **0.61** |
| full without result | -5.889 | 49.1% | 10/21 | 2.57 / 2 | 6.48 | 0.53 |
| full without progression | -5.377 | 55.4% | 9/21 | 2.43 / 2 | 6.76 | 0.61 |
| full without form | -5.401 | 55.3% | 8/21 | 2.76 / 2 | 6.62 | 0.59 |
| full + reputation | -5.374 | 55.3% | 9/21 | 2.52 / 2 | 6.90 | 0.60 |

What the backtests show:

- **The result features matter most for predicting votes.** Taking them out drops the share of
  3-vote players picked by about 6 points and the top 10 order from 0.69 to 0.57 (0.61 to 0.53
  over 21 seasons). Without them the model happens to pick one more winner, but over 12 or 21
  seasons that's within luck, and the order of the top 10 is clearly worse.
- **Season form and umpire appeal improve the top of the leaderboard.** Over 21 seasons they
  moved the winner's mean predicted rank from 3.62 to 2.43 and the top 10 order from 0.55 to 0.61.
  Season form alone moved the actual winner up the leaderboard in 7 of 21 seasons and never down.
- **Scoring progression adds little.** It slightly improves game-by-game predictions, but makes
  no difference to the winner or the top 10 over 21 seasons. It's kept because it does no harm.
- **The 2006–2014 seasons were harder to predict than 2015–2026.** Over all 21 seasons the model
  picks the winner about 43% of the time, so the 58% from 2015–2026 alone is probably flattering.

### Share of 6 votes or strict 3-2-1 (full model)

| Seasons | Tally | Season total error (votes per player) | Winners picked | Winner's mean rank | Top 10 order |
|---|---|---|---|---|---|
| 2015–2026 | Expected votes (share of 6) | **1.82** | 7/12 | 2.25 | **0.69** |
| 2015–2026 | 3-2-1 | 2.14 | 7/12 | 2.33 | 0.62 |
| 2006–2026 | Expected votes (share of 6) | **1.87** | 9/21 | 2.43 | **0.61** |
| 2006–2026 | 3-2-1 | 2.31 | 9/21 | 2.62 | 0.58 |

### Full model by season

| Season | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Winner's predicted rank (basic stats) | 3 | 4 | 4 | 1 | 5 | 1 | 2 | 1 | 4 | 1 | 1 |
| Top 10 order (basic stats) | 0.65 | 0.27 | 0.47 | 0.79 | 0.52 | 0.43 | 0.40 | 0.65 | 0.36 | 0.92 | 0.63 |
| Winner's predicted rank (all features) | | | | | | | | | | 1 | 1 |
| Top 10 order (all features) | | | | | | | | | | 0.88 | 0.55 |

| Season | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| Winner's predicted rank (basic stats) | 2 | 1 | 2 | 1 | 1 | 3 | 4 | 2 | 7 | 1 |
| Top 10 order (basic stats) | 0.72 | -0.03 | 0.69 | 0.96 | 0.62 | 0.87 | 0.75 | 0.90 | 0.51 | 0.73 |
| Winner's predicted rank (all features) | 2 | 1 | 2 | 1 | 1 | 3 | 6 | 1 | 7 | 1 |
| Top 10 order (all features) | 0.82 | -0.00 | 0.66 | 0.94 | 0.64 | 0.82 | 0.69 | 0.96 | 0.55 | 0.79 |

The worst miss is 2025: Matt Rowell polled 39 and the model ranked him 7th. In 2026 the model
predicted Nick Daicos 45.7 (actual 47) and Bailey Smith 34.1 (36), both in the right places.

### How honest the season ranges are (full model, 2015–2026)

| `--player-sd` | Actual totals inside the 80% range | Log probability of actual totals | Chance given to the actual winner |
|---|---|---|---|
| 0.0 | 82.9% | -2.216 | 0.579 |
| 0.2 | 83.8% | -2.202 | 0.569 |
| **0.3** (default) | 84.7% | -2.191 | 0.556 |
| 0.4 | 86.1% | -2.182 | 0.540 |
| 0.5 | 87.3% | -2.176 | 0.520 |

Larger values make outliers like Rowell less surprising, at the cost of ranges a little wider than
80%. 0.3 is a compromise between the two. This only affects the ranges and win chances, not the
order of the leaderboard.

## Limitations

- Suspended players can't win, but suspensions aren't in the database, so the model doesn't know
  who is ineligible.
- Win chances treat players as independent, though two players in the same game can't both get
  the 3 votes (the Monte Carlo model handles this).
- The model is linear: the same weights for every position, and no interactions between stats
  (the gradient-boosted ranker tests whether that matters).
- Training defaults to 2012 onwards, where the afl.com.au stats start.
