import NMRfit as nmrft
import os
import matplotlib.pyplot as plt
import numpy as np


# input directory
inDir = "./Data/blindedData/dc_4f_cdcl3_kilimanjaro_25c_1d_1H_2_050216.fid"

# read in data
data = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# bound the data
data.select_bounds(low=3.23, high=3.6)


# select peaks and satellites
peaks = data.select_peaks(method='auto', n=6, plot=True)

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

# create x0_adj, adjusted initial condition vector that has smaller sigmas
mult = np.ones_like(x0)
for i, item in enumerate(mult):
    if (i>2 and (i%3==0)):
        mult[i]=0.75

x0_adj=x0*mult

print("x0=",x0)
print("x0_adj=",x0_adj)

# fit data
#fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)
fit = nmrft.FitUtility(data, x0_adj, method='TNC', bounds=bounds, options=None)

# generate result
res = fit.generate_result(scale=1)

# # summary
fit.summary()

plt.plot(fit.data.w, fit.data.V, linewidth=2, alpha=0.5)
plt.plot(fit.result.w, fit.result.V, linewidth=2, alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

plt.plot(fit.data.w, fit.data.I, linewidth=2, alpha=0.5)
plt.plot(fit.result.w, fit.result.I, linewidth=2, alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()
