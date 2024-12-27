import os
import sys
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

all_files = []
for root, dirs, files in os.walk("./SKAB"):
    for file in files:
        if file.endswith(".csv"):
             all_files.append(f'{root}/{file}')

anomaly_free = pd.read_csv('./SKAB/anomaly-free/anomaly-free.csv', sep=';')
valve1 = [pd.read_csv(file, sep=';', index_col='datetime', parse_dates=True) for file in all_files if 'valve1' in file]

anomaly_free.plot(x="datetime", y="Current")
for exp in valve1:
    print(exp.anomaly.value_counts())
    print(exp.changepoint.value_counts())

plt.show()
