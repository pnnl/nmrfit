import NMRfit as nmrft
import os
import matplotlib.pyplot as plt
import numpy as np


# input directory
inDir = "./data/toluene/toluene2_cdcl3_long2_070115.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.30, high=3.7)

# select peaks and satellites
peaks = data.select_peaks(method='auto', n=3, plot=True)

# generate bounds and initial conditions
lb, ub = data.generate_initial_conditions()

# fit data
fit = nmrft.FitUtility(data, lb, ub)

# generate result
fit.calculate_area_fraction()
fit.generate_result(scale=1)

# # summary
plt.plot(fit.data.w, fit.data.V, linewidth=2, alpha=0.5, color='blue', label='Data')
plt.plot(fit.result.w, fit.result.V, linewidth=2, alpha=0.5, color='red', label='fit')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
# plt.savefig('./results/%s_pso.png' % ID, bbox_inches='tight', dpi=200)
plt.show()
plt.close()
