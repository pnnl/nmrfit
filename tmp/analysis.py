import numpy as np
import glob
import os
import pandas
from scipy import stats
from itertools import combinations


def parse(txtfile):
    attempts = []
    ps = []
    ssds = []
    fits = []
    with open(txtfile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if 'Failed attempts:' in l:
                fa = int(l.split()[-1])
                attempts.append(fa)
            elif 'Percent:' in l:
                percent = float(l.split()[-1])
                ps.append(percent)
            elif 'SSD:' in l:
                ssd = float(l.split()[-1])
                ssds.append(ssd)
            elif 'Fit params:' in l:
                fit = [float(x) for x in l.split()[2:]]
                fits.append(fit)
    return attempts, ps, ssds, fits


inDir = '../Data/blindedData/results/'

files = glob.glob(os.path.join(inDir, '*.fid.txt'))

data = {'4a': [],
        '4b': [],
        '4c': [],
        '4d': [],
        '4e': [],
        '4f': [],
        '4h': []}

for f in files:
    parts = f.split('_')
    dataset = parts[2]
    sample = parts[-2]

    a, p, s, fits = parse(f)
    data[dataset].extend(p)

for k in data:
    data[k] = np.array(data[k])

df = pandas.DataFrame.from_dict(data, orient='index')

n = df.count(axis=1)
avg = np.mean(df, axis=1)
stdev = np.std(df, axis=1)

df['n'] = n
df['avg'] = avg
df['std'] = stdev

df.sort(columns='avg', inplace=True)

for i in range(len(df.index) - 1):
    t, p = stats.ttest_ind(data[df.index[i]], data[df.index[i + 1]], equal_var=False)
    print(df.index[i], df.index[i + 1], p / 2)

df.to_csv('../stats.csv')
