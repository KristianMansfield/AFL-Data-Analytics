#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import numpy as np
from dbConnection import createReaderConnection

db = createReaderConnection()

polled_stat = []
polls = []
unpolled_stat = []
unpolls = []

sql = "SELECT id, disposals, goals FROM player_game_stats WHERE brownlow_votes > 0;"

results = db.fetchall(sql)

for res in results:
    polled_stat.append(res[1])
    polls.append(res[2])


sql = "SELECT id, disposals, goals FROM player_game_stats WHERE brownlow_votes = 0;"

results = db.fetchall(sql)

for res in results:
    unpolled_stat.append(res[1])
    unpolls.append(res[2])

db.close()

plt.plot(unpolled_stat, unpolls, 'bo')
plt.plot(polled_stat, polls, 'ro')
plt.title("Brownlow by stat distribution")

plt.ylabel("Goals")
plt.xlabel("Disposals")

plt.show()
