# Brownlow Prediction

Four models that predict Brownlow Medal votes from the stats in the database, and a script that
backtests them against each other on picking the winner and the top 10 in order.

| Folder | Model | In one line |
|---|---|---|
| [Plackett-Luce Model](Plackett-Luce%20Model/) | Ranked (Plackett-Luce) logit | A weighted sum of each player's stats gives their impact; the 3, 2 and 1 votes are drawn in turn from the players in the game in proportion to exp(impact). Exact probabilities, no simulation. **Recommended.** |
| [Gradient Boosted Ranking](Gradient%20Boosted%20Ranking/) | LightGBM decision trees with a ranking loss (ListMLE or LambdaRank) | Trees learn each player's impact instead of a weighted sum, so stats can interact; votes then follow as in the Plackett-Luce model. |
| [Vote Classifier](Vote%20Classifier/) | Per-player 0/1/2/3 classifier | Classifies each player-game on its own, then rescales each game so exactly one 3, one 2 and one 1 are given. |
| [Monte Carlo Simulation](Monte%20Carlo%20Simulation/) | Simulated seasons | The Plackett-Luce model's games played out 10,000 times, with uncertain weights and a shared umpire appeal per player; gives chances of a top 3 or top 10 finish and average finishing position. |

The Plackett-Luce folder also holds the code every model shares: loading the data and building
the features (`gameData.py`), turning impacts into vote chances (`rankedLogit.py`), the season
tally (`seasonTally.py`) and the output files and printed leaderboard (`predictionReport.py`). Its
[README](Plackett-Luce%20Model/README.md) describes the features, what the model learned and its
own backtests.

## Usage

Each model is run from its own folder and writes to `output/<year>/` there: `leaderboard.csv`,
`game_predictions.csv` and `round_by_round.csv`, in the same format for every model.

```
cd "Brownlow Prediction/Plackett-Luce Model";     ../../venv/bin/python predictBrownlow.py      # ~5 s
cd "Brownlow Prediction/Gradient Boosted Ranking"; ../../venv/bin/python predictRanking.py      # ~10 s
cd "Brownlow Prediction/Vote Classifier";          ../../venv/bin/python predictClassifier.py   # ~15 s
cd "Brownlow Prediction/Monte Carlo Simulation";   ../../venv/bin/python predictSimulation.py   # ~10 s
```

`predictBrownlow.py --ensemble` blends the Plackett-Luce model 70/30 with the gradient-boosted
ranker; see the [Plackett-Luce README](Plackett-Luce%20Model/README.md) for how it compared.

Every predict script takes `--year`, `--train-years`, `--basic-stats` (leave out the afl.com.au
stats and train from 2003) and `--top`; `--help` lists each model's own options.

To compare the models:

```
cd "Brownlow Prediction"
../venv/bin/python compareModels.py                   # tests 2015-2026, ~4 minutes
../venv/bin/python compareModels.py --basic-stats     # tests 2006-2026, ~10 minutes
../venv/bin/python compareModels.py --models pl simulation   # only some models
```

Results are written to `output/model_comparison.csv` (or `model_comparison_basic_stats.csv`).

## The models

### Plackett-Luce Model

Each player's impact in a game is a weighted sum of 49 features: their stats, within-game ranks,
the result and margin, scoring progression (win probability added by their goals), season form
and umpire appeal (how much they out-polled the model in the previous three seasons). The weights
are fitted by maximum likelihood on the votes actually given. Each player's chance of 3, 2 and 1
votes is worked out exactly by summing over every possible top-three order, and season totals and
win chances are combined exactly from those. See its [README](Plackett-Luce%20Model/README.md).

### Gradient Boosted Ranking

Replaces the weighted sum with an ensemble of small LightGBM decision trees (15 leaves each),
fitted one at a time with half the rows sampled for each tree. Two ranking objectives are
available (`--objective`):

- `listmle` (default): the same Plackett-Luce likelihood as the Plackett-Luce model, given to
  LightGBM as a custom objective. Its scores are Plackett-Luce impacts directly.
- `lambdarank`: LightGBM's standard learning-to-rank objective, which optimises the order of each
  game's top three (NDCG@3, with 3 votes more relevant than 2, and 2 than 1). Its scores only rank
  players, so one scale factor, fitted on a held-out season, turns them into Plackett-Luce impacts.

The number of trees is chosen by holding out the last training season. Vote chances and season
tallies are then worked out the same way as the Plackett-Luce model. It uses the same features
except umpire appeal (which needs a Plackett-Luce model of its own); reputation (votes per game in
the previous three seasons) stands in for it.

LightGBM needs the OpenMP runtime, which on macOS is `brew install libomp`.

### Vote Classifier

A gradient-boosted classifier (scikit-learn's `HistGradientBoostingClassifier`) gives each
player-game P(0), P(1), P(2) and P(3 votes), from the same features as the ranker plus key stats as
z-scores within the game, so it can see how the player compared with the rest of the game. The raw
chances don't add up across a game, so each game's grid of chances is rescaled with Sinkhorn
balancing until every player's row adds to 1 and each vote is given exactly once. Season tallies
treat games as independent.

### Monte Carlo Simulation

Fits the Plackett-Luce model, then simulates the season (10,000 times by default). Each simulated
season:

1. redraws the model's weights from their estimated uncertainty
2. gives each player one umpire appeal shift, shared by all their games
3. draws every game's 3, 2 and 1 from the Plackett-Luce model (Gumbel noise added to each impact,
   top three get the votes)

Counting across simulations gives average votes, chance of winning, chance of a top 3 and top 10
finish, average finishing position and a range of totals. Unlike the exact calculation, the
simulation never lets two players in the same game both get the 3 votes, and it carries the
uncertainty in the weights. `--order position` ranks the leaderboard by average finishing
position instead of average votes.

## Comparison

`compareModels.py`: each season predicted from earlier seasons only. Winner's rank is where the
actual winner sat on the predicted leaderboard (1 = picked). Top 10 order is the Spearman
correlation between the actual and predicted positions of the actual top 10 (1 = exactly the right
order). Contender log-lik scores the chances each model gave the votes the actual top 10 got in
every one of their games (about 230 player-games a season, so it's steadier than the winner;
higher is better).

### 2015–2026, all features

| Model | Winners picked | Winner's rank (mean / median) | Top 10 overlap | Top 10 order | Top 10 total error | Contender log-lik | 3-vote player picked |
|---|---|---|---|---|---|---|---|
| Plackett-Luce | 7/12 | 2.25 / 1 | 7.08 | 0.691 | **4.33** | **-174.9** | 59.2% |
| Gradient boosted ranking (ListMLE) | 7/12 | 2.33 / 1 | 7.08 | 0.667 | 4.43 | -175.8 | 57.8% |
| Gradient boosted ranking (LambdaRank) | **8/12** | 2.42 / 1 | **7.25** | 0.660 | 4.61 | -177.7 | **59.5%** |
| Vote classifier | 7/12 | 2.58 / 1 | 7.17 | 0.656 | 4.74 | -182.5 | 58.2% |
| Monte Carlo | 7/12 | 2.08 / 1 | 7.08 | **0.701** | 4.39 | **-174.9** | 59.2% |
| Monte Carlo, by position | 7/12 | **2.00 / 1** | 7.08 | 0.699 | 4.39 | **-174.9** | 59.2% |

### 2006–2026, basic stats only (`--basic-stats`)

| Model | Winners picked | Winner's rank (mean / median) | Top 10 overlap | Top 10 order | Top 10 total error | Contender log-lik | 3-vote player picked |
|---|---|---|---|---|---|---|---|
| Plackett-Luce | **9/21** | **2.43 / 2** | 6.76 | **0.609** | **4.76** | **-180.2** | **55.4%** |
| Gradient boosted ranking (ListMLE) | **9/21** | 2.95 / 2 | 6.76 | 0.563 | 4.87 | -182.4 | 54.1% |
| Gradient boosted ranking (LambdaRank) | 8/21 | 2.71 / 2 | 6.71 | 0.544 | **4.76** | -184.9 | 53.9% |
| Vote classifier | 8/21 | 2.95 / 2 | **6.90** | 0.579 | 5.29 | -189.7 | 55.1% |
| Monte Carlo | **9/21** | 2.48 / 2 | 6.86 | **0.609** | 4.81 | **-180.2** | **55.4%** |
| Monte Carlo, by position | **9/21** | 2.52 / 2 | **6.90** | 0.606 | 4.81 | **-180.2** | **55.4%** |

Against the Plackett-Luce model, season by season over the 21 seasons:

| Model | Winner placed higher / same / lower | Top 10 order better / worse |
|---|---|---|
| Gradient boosted ranking (ListMLE) | 4 / 11 / 6 | 8 / 12 |
| Gradient boosted ranking (LambdaRank) | 2 / 12 / 7 | 10 / 11 |
| Vote classifier | 2 / 12 / 7 | 11 / 10 |
| Monte Carlo | 0 / 20 / 1 | 2 / 2 |
| Monte Carlo, by position | 0 / 19 / 2 | 6 / 5 |

### What the comparison shows

- **The Plackett-Luce model is the best choice.** It's the most accurate game by game (the
  highest contender log-likelihood, and the highest share of 3-vote players picked over 21
  seasons), the best on the winner's rank and the top 10 order over 21 seasons (Monte Carlo edges
  it over 2015–2026 alone, see below), and the fastest.
- **Monte Carlo is effectively the same model.** It uses the same game predictions, so it only
  differs in how the season is added up. Over 12 seasons it looked slightly better, mostly because
  of one season (2025, where it ranked Rowell 5th instead of 7th). Over 21 seasons it was identical
  in 20 of them. Its value is the extra outputs (chances of a top 3 or top 10 finish, average
  position) and handling the same-game dependence properly, not better predictions.
  Ordering by average position instead of average votes makes no real difference.
- **Trees don't beat the weighted sum.** With the ListMLE objective the gradient-boosted ranker
  comes close over 2015–2026 (top 10 order 0.667 against 0.691), but over 21 seasons it places the
  winner lower in more seasons than higher (6 against 4) and gets the top 10 order worse in 12 of
  21. LambdaRank picked one more winner over 2015–2026, which is within luck, and is worse on the
  top 10 order and game-by-game accuracy in both periods. Umpires' votes look close to a weighted
  sum of the stats, and with 12–21 seasons of votes the extra flexibility of trees costs more in
  noise than it gains. Of the two objectives, ListMLE is the better default: its scores are vote
  probabilities directly, and it predicts the contenders' votes better.
- **Classifying players one at a time is the weakest approach.** The vote classifier can't see
  that one player's 3 votes take them away from everyone else in the game, and rescaling
  afterwards only partly recovers it. It's the least accurate game by game for the contenders
  and, with the ListMLE ranker, places the winner lowest over 21 seasons.
- **Differences in winners picked are mostly luck.** One more or fewer winner out of 12 or 21 is
  well within chance. The contender log-likelihood and top 10 order are steadier guides.

## Limitations

- Suspended players can't win, but suspensions aren't in the database, so no model knows who is
  ineligible.
- The Monte Carlo results vary slightly from run to run (fixed by `--seed`); 10,000 simulations
  keep chances accurate to about ±0.5 percentage points.
- The gradient-boosted ranker and vote classifier have hyperparameters (leaves per tree, learning
  rate, leaf size) set to sensible defaults, not tuned. Tuning might narrow their gap, but would need
  care not to tune to the test seasons.
