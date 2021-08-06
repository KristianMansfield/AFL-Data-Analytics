#!/usr/bin/env python

# TODO Account for draws

import matplotlib.pyplot as plt
import pymysql
import sys
import numpy as np
from winSizeVsLadderPositionCorrelation import genCorrolationScore
from dbConnection import createReaderConnection

dbhandler = createReaderConnection()

cursor = dbhandler.cursor()

years = []
evenness_scores = []
num_upsets = []
margins = []
table_difference_ratio = []
sum_upsets = []
correlation_factors = []

for YEAR in range(1990, 2020 + 1):

	years.append(YEAR)



	# Number of times lower beats upper
	number_of_upsets_sql = "SELECT COUNT(*) FROM (SELECT CAST(ltp.position AS SIGNED) - CAST(wtp.position AS SIGNED) AS Total FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season INNER JOIN teams wt ON games.winning_team = wt.id INNER JOIN teams lt ON games.losing_team = lt.id INNER JOIN ladder_entries wtp ON wtp.team = games.winning_team INNER JOIN rounds wr ON wr.id = wtp.round INNER JOIN seasons ws ON ws.id = wr.season INNER JOIN ladder_entries ltp ON ltp.team = games.losing_team INNER JOIN rounds lr ON lr.id = ltp.round INNER JOIN seasons ls ON ls.id = lr.season WHERE s.id = ws.id AND ws.id = ls.id AND s.year = " + str(YEAR) + " AND wtp.position > ltp.position ORDER BY games.start_datetime ASC) x;"
	cursor.execute(number_of_upsets_sql)
	results = cursor.fetchall()
	number_of_upsets = results[0][0]

	number_of_games_sql = "SELECT COUNT(*) FROM (SELECT games.id FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season WHERE s.year = " + str(YEAR) + ") x"
	cursor.execute(number_of_games_sql)
	results = cursor.fetchall()
	number_of_games = results[0][0]

	upset_ratio = (number_of_upsets)/(number_of_games)

	num_upsets.append(upset_ratio)



	# Average margin
	margin_sql = "SELECT AVG(CAST(games.winning_score AS SIGNED)/CAST(games.losing_score AS SIGNED)) AS Margin FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season INNER JOIN teams wt ON games.winning_team = wt.id INNER JOIN teams lt ON games.losing_team = lt.id INNER JOIN ladder_entries wtp ON wtp.team = games.winning_team INNER JOIN rounds wr ON wr.id = wtp.round INNER JOIN seasons ws ON ws.id = wr.season INNER JOIN ladder_entries ltp ON ltp.team = games.losing_team INNER JOIN rounds lr ON lr.id = ltp.round INNER JOIN seasons ls ON ls.id = lr.season WHERE s.id = ws.id AND ws.id = ls.id AND s.year = " + str(YEAR) + " ORDER BY games.start_datetime ASC;"
	cursor.execute(margin_sql)
	results = cursor.fetchall()
	margin = float(results[0][0])
	margins.append(margin)



	# Correlation factor
	cor_fact = genCorrolationScore(YEAR)
	correlation_factors.append(cor_fact)



	# Evenness
	evenness = upset_ratio/(margin * cor_fact)
	evenness_scores.append(evenness)



plt.plot(years, num_upsets, 'bo')
plt.plot(years, margins, 'yo')
plt.plot(years, correlation_factors, 'ro')
plt.plot(years, evenness_scores, 'go')

plt.title("How Uneven is the Competition?")
plt.ylabel("Uneven Factor")
plt.xlabel("Year")

plt.show()



# # Sum of table difference where lower beats upper
# sum_upset_sql = "SELECT sum(CAST(ltp.position AS SIGNED) - CAST(wtp.position AS SIGNED)) AS Total FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season INNER JOIN teams wt ON games.winning_team = wt.id INNER JOIN teams lt ON games.losing_team = lt.id INNER JOIN ladder_entries wtp ON wtp.team = games.winning_team INNER JOIN rounds wr ON wr.id = wtp.round INNER JOIN seasons ws ON ws.id = wr.season INNER JOIN ladder_entries ltp ON ltp.team = games.losing_team INNER JOIN rounds lr ON lr.id = ltp.round INNER JOIN seasons ls ON ls.id = lr.season WHERE s.id = ws.id AND ws.id = ls.id AND s.year = " + str(YEAR) + " AND wtp.position > ltp.position ORDER BY games.start_datetime ASC;"
# cursor.execute(sum_upset_sql)
# results = cursor.fetchall()
# sum_upsets_result = results[0][0]

# # Sum of table difference where upper beats lower
# sum_standard_sql = "SELECT sum(CAST(ltp.position AS SIGNED) - CAST(wtp.position AS SIGNED)) AS Total FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season INNER JOIN teams wt ON games.winning_team = wt.id INNER JOIN teams lt ON games.losing_team = lt.id INNER JOIN ladder_entries wtp ON wtp.team = games.winning_team INNER JOIN rounds wr ON wr.id = wtp.round INNER JOIN seasons ws ON ws.id = wr.season INNER JOIN ladder_entries ltp ON ltp.team = games.losing_team INNER JOIN rounds lr ON lr.id = ltp.round INNER JOIN seasons ls ON ls.id = lr.season WHERE s.id = ws.id AND ws.id = ls.id AND s.year = " + str(YEAR) + " AND wtp.position < ltp.position ORDER BY games.start_datetime ASC;"
# cursor.execute(sum_standard_sql)
# results = cursor.fetchall()
# sum_standard_result = results[0][0]

# sum_upsets.append(sum_upsets_result)
# table_difference_ratio.append((-sum_upsets_result/(sum_standard_result - sum_upsets_result)))
# print(sum_upsets_result, sum_standard_result)