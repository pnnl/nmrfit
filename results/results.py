import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./all_working.csv')

# df = df[df['method'] == 'TNC']
tmp = pd.DataFrame()
for key, grp in df.groupby(by='sample'):
    col = grp['fraction'].values
    print(len(col))
    avg = np.mean(col)
    std = col.std()

    col = np.append(col, np.zeros(5 - len(col)) + np.nan)

    # col[abs(col - avg) > 1.5 * std] = np.nan
    tmp[key] = col

print(tmp)
tmp.boxplot()
plt.ylim((0, 0.02))
plt.ylabel('Area Fraction')
plt.xlabel('Sample')
plt.show()
# plt.boxplot(tmp.values, labels=list(tmp.columns))
# plt.show()