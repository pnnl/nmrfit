import NMRfit as nmrft
import os


# input directory
inDir = "./Data/blindedData/dc_4b_cdcl3_kilimanjaro_25c_1d_1H_1_043016.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.23, high=3.6)

# interactively select peaks and satellites
peaks = data.select_peaks(method='manual', n=6, plot=True)

# generate bounds and initial conditions
x0, bounds = data.generate_initial_conditions()

# fit data
fit = nmrft.FitUtility(data, x0, method='Powell')

# generate result
res = fit.generate_result(scale=1)

# summary
fit.summary()

print('\nMoving on to TNC fit:\n')

# Now we will pass global results onto TNC
x0[:3] = res.params[:3]

# fit data
fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)

# generate result
res = fit.generate_result(scale=1)

# summary
fit.summary()
