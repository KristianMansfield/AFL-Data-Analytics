#!/usr/bin/env python

# TODO Account for draws

import matplotlib.pyplot as plt
import pymysql
import sys
import numpy as np
from scipy import stats
from dbConnection import createReaderConnection

def genCorrolationScore(year, show_graph=False, output=False):
    dbhandler = createReaderConnection()

    cursor = dbhandler.cursor()

    margins = []
    position_differences = []

    sql = "SELECT games.winning_score, games.losing_score, wtp.position, ltp.position FROM games INNER JOIN rounds r ON games.round = r.id INNER JOIN seasons s ON s.id = r.season INNER JOIN teams wt ON games.winning_team = wt.id INNER JOIN teams lt ON games.losing_team = lt.id INNER JOIN ladder_entries wtp ON wtp.team = games.winning_team INNER JOIN rounds wr ON wr.id = wtp.round INNER JOIN seasons ws ON ws.id = wr.season INNER JOIN ladder_entries ltp ON ltp.team = games.losing_team INNER JOIN rounds lr ON lr.id = ltp.round INNER JOIN seasons ls ON ls.id = lr.season WHERE s.id = ws.id AND ws.id = ls.id AND s.year = " + str(year) + " ORDER BY games.start_datetime ASC;"
    cursor.execute(sql)
    results = cursor.fetchall()

    for res in results:
        winning_score, losing_score, winning_position, losing_position = res[0], res[1], res[2], res[3]
        margins.append(winning_score - losing_score)
        position_differences.append(losing_position - winning_position)

    x = np.array(position_differences)
    y = np.array(margins)

    scp_res = stats.linregress(x, y)
    
    if output: 
        print("Scipy r value:", scp_res.rvalue)
        print("Line of best fit: ", str(scp_res.slope) + "*x + " + str(scp_res.intercept))

        if show_graph: 
            plt.plot(x, y, 'bo')
            plt.plot(x, scp_res.slope*x + scp_res.intercept, 'y-')

            plt.title("Margins by Position Differences")
            plt.ylabel("Margin")
            plt.xlabel("Difference of Position between Winning Team vs Losing Team")

            plt.show()

    return float(scp_res.rvalue)

if __name__ == '__main__':
    genCorrolationScore(sys.argv[1], show_graph=True, output=True)