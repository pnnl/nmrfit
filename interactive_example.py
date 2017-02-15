import NMRfit as nmrft
import os
import matplotlib.pyplot as plt
import numpy as np


# input directory
inDir = "./data/blindedData/dc_4c_cdcl3_kilimanjaro_25c_1d_1H_1_072116.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.20, high=3.6)

# select peaks and satellites
peaks = data.select_peaks(method='auto', n=6, plot=True)

# generate bounds and initial conditions
lb, ub = data.generate_initial_conditions()

# fit data
fit = nmrft.FitUtility(data, lb, ub)

# generate result
fit.calculate_area_fraction()
fit.generate_result(scale=1)

# # summary
fit.summary()
