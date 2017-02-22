import NMRfit as nmrft
import os


# input directory
inDir = "./data/toluene/toluene2_cdcl3_long2_070115.fid"

# read in data
data = nmrft.load(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.30, high=3.7)

data.shift_phase(method='brute')

# select peaks and satellites
data.select_peaks(method='auto', thresh=0.005, piecewise_baseline=False, plot=False)

# generate bounds and initial conditions
lb, ub = data.generate_solution_bounds()

# fit data
fit = nmrft.FitUtility(data, lb, ub, fitIm=False, expon=0.8)

# generate result
fit.generate_result(scale=10)

# summary
fit.summary()
