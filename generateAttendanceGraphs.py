#!/usr/bin/env python

import matplotlib.pyplot as plt
import pymysql
import sys
import numpy as np
from dbConnection import createReaderConnection

dbhandler = createReaderConnection()

cursor = dbhandler.cursor()

# sql = "SELECT id, final_home_score, final_away_score, attendance, final_home_score - final_away_score as Margin FROM games WHERE home_team = 14;"
sql = sys.argv[1]

cursor.execute(sql)

results = cursor.fetchall()

attendances = []
margins = []

for res in results:
    if res[3] != None:
        attendances.append(res[3])
    else:
        attendances.append(0)
    margins.append(res[4])

print(len(attendances), len(margins))
x = np.array(attendances)
y = np.array(margins)
print(len(x), len(y))

m, b = np.polyfit(x, y, 1)
print("Line of best fit: ", str(m) + "*x + " + str(b))

plt.plot(x, y, 'bo')
plt.plot(x, m*x + b, 'y-')
plt.axhline(y=0, color='r', linestyle='-')
plt.title(sys.argv[2])

plt.ylabel("Margin")
plt.xlabel("Attendance")

plt.show()
