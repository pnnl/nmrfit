import nmrfit
import os
import matplotlib.pyplot as plt
import numpy as np


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def compress(x, y, ythresh=1.0, xthresh=1.0):

    dy = np.stack((np.abs(np.roll(y, 1) - y), np.abs(np.roll(y, -1) - y)), axis=1).max(axis=1)
    rdy = dy / y

    dy = normalize(dy)
    rdy = normalize(rdy)

    dy = np.stack((dy, rdy), axis=1).max(axis=1)

    idx1 = np.where(dy > ythresh * max(dy) / 2)[0]

    if xthresh == 0:
        idx2 = np.array([0, len(x) - 1])
    else:
        idx2 = np.arange(0, len(x), int(1 / xthresh))

    idx = np.union1d(idx1, idx2)

    print('Before:', len(x))
    print('After:', len(idx))
    print('Compression ratio:', len(x) / len(idx))

    return x[idx], y[idx]


# input directory
inDir = "examples/data/blindedData/dc_4d_cdcl3_kilimanjaro_25c_1d_1H_1_072316.fid"

# read in data
data = nmrfit.load(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.23, high=3.58)

# phase shift
data.shift_phase(method='brute', plot=False)


x, y = compress(data.w, data.V, ythresh=0.2, xthresh=0.2)

plt.scatter(x, y, s=3, color='r', alpha=1)
# plt.scatter(data.w, data.V, s=3, color='b', alpha=0.5)
plt.show()
