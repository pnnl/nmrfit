import NMRfit as nmrft
import os
import matplotlib.pyplot as plt
import numpy as np


# input directory
inDir = "./Data/blindedData/dc_4a_cdcl3_kilimanjaro_25c_1d_1H_1_043016.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.23, high=3.6)

# select peaks and satellites
peaks = data.select_peaks(method='manual', n=6, plot=True)
quit()

# generate bounds and initial conditions
x0, bounds = data.generate_initial_conditions()

# fit data
fit = nmrft.FitUtility(data, x0, method='Powell')

# generate result
res = fit.generate_result(scale=1)

# summary
fit.summary()

# print(fit.calculate_area_fraction())

print('\nMoving on to TNC fit:\n')

# Now we will pass global results onto TNC
x0[:3] = res.params[:3]
x0[1]=max(0.,min(1.,x0[1]))

# fit data
#fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)
fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options={'maxCGit':1000,'maxiter':1000})

# generate result
res = fit.generate_result(scale=1)

# # summary
fit.summary()

plt.figure(figsize=(5, 5))
plt.plot(fit.data.w, fit.data.V, linewidth=2, alpha=0.5, color='b', label='Data')
plt.plot(fit.result.w, fit.result.V, linewidth=2, alpha=0.5, color='r', label='Fit')
plt.legend()
plt.xlabel('Frequency', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(fit.data.w, fit.data.I, linewidth=2, alpha=0.5, color='b', label='Data')
plt.plot(fit.result.w, fit.result.I, linewidth=2, alpha=0.5, color='r', label='Fit')
plt.legend()
plt.xlabel('Frequency', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.show()
