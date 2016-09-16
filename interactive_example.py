import NMRfit as nmrft
import os


# input directory
inDir = "./Data/blindedData/dc_4d_cdcl3_kilimanjaro_25c_1d_1H_2_050116.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.25, high=3.6)

# interactively select peaks and satellites
peaks = data.select_peaks(method='auto', n=6)

# generate bounds and initial conditions
x0, bounds = data.generate_initial_conditions()

# fit data
fit = nmrft.FitUtility(data, x0, method='Powell')

# generate result
res = fit.generate_result(scale=1)

# summary
fit.print_summary()
fit.summary_plot()

print ('\nMoving onto TNC fit:\n')

# Now we will pass global results onto TNC
x0[:3] = res.params[:3]

# fit data
fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)

# generate result
res = fit.generate_result(scale=1)

# summary
fit.print_summary()
fit.summary_plot()
