import nmrfit
import peakutils
import matplotlib.pyplot as plt
import os

import scipy as sp
from scipy import integrate
import numpy as np


# input directory
inDir = "examples/data/blindedData/dc_4d_cdcl3_kilimanjaro_25c_1d_1H_1_072316.fid"

# read in data
data = nmrfit.load(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.23, high=3.58)

# phase shift
data.shift_phase(method='brute', plot=False)

# select peaks and satellites
data.select_peaks(method='auto', thresh=0.005, plot=False)

bl = peakutils.baseline(data.V, 0)

areas1 = []
areas2 = []
for p in data.peaks:
    idx = np.where((data.w >= p.bounds[0]) & (data.w <= p.bounds[1]))

    pw = data.w[idx]
    pV = data.V[idx]

    gbl = bl[idx]
    lbl = peakutils.baseline(pV, 0)

    areas1.append(sp.integrate.simps(pV - gbl, pw))
    areas2.append(sp.integrate.simps(pV - lbl, pw))

areas1 = np.array(areas1)
areas2 = np.array(areas2)

m1 = areas1.mean()
m2 = areas2.mean()

peaks1 = areas1[areas1 > m1].sum()
sats1 = areas1[areas1 < m1].sum()

peaks2 = areas2[areas2 > m2].sum()
sats2 = areas2[areas2 < m2].sum()

print('global', sats1 / (sats1 + peaks1))
print('local', sats2 / (sats2 + peaks2))
