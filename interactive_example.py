import NMRfit as nmrft
import os
import matplotlib.pyplot as plt


def lblpars(i):
    if (i == 0):
        print ('Global Parameters:')
    elif (i == 3):
        print ('Major Peak Parameters:')
    elif (i == 9):
        print ('Minor Peak Parameters:')


def print_summary(x0, res):
    # print summary
    print('\nSEED PARAMETER VALUES:\n')

    for i in range(0, len(x0), 3):
        lblpars(i)
        print(x0[i:i + 3])

    print('\nCONVERGED PARAMETER VALUES:\n')

    for i in range(0, len(res.params), 3):
        lblpars(i)
        print(res.params[i:i + 3])

    print("\nFINAL ERROR:  ", res.error)

# Powell alone works
# inDir = "./Data/blindedData/dc_4a_cdcl3_kilimanjaro_25c_1d_1H_1_043016.fid"

# TNC adds value
# inDir = "./Data/blindedData/dc_4e_cdcl3_kilimanjaro_25c_1d_1H_1_050116.fid"

# Even TNC can't get them all
# inDir = "./Data/blindedData/dc_4h_cdcl3_kilimanjaro_25c_1d_1H_2_050616.fid"

# Another random
inDir = "./Data/blindedData/dc_4b_cdcl3_kilimanjaro_25c_1d_1H_5_071116.fid"

w, u, v, theta0 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# create data object
data = nmrft.Data(w, u, v, theta0)

# bound the data
data.select_bounds(low=3.2, high=3.6)

# interactively select peaks and satellites
peaks = data.select_peaks('manual',6)

# initial conditions of the form [theta, r, yOff, sigma_n, mu_n, a_n,...]
x0 = [data.theta, 1., 0.]
bounds = [(None, None), (0., 1.), (None, None)]

for p in peaks:
    x0.extend([p.sigma, p.loc, p.area])
    bounds.extend([(None, None), (p.loc - 0.05, p.loc + 0.05), (None, None)])

# fit data
fit = nmrft.FitUtility(data, x0, method='Powell', options=None)

# generate result
res = fit.generate_result(scale=1)

# plot summary
print_summary(x0, res)

# fit.summary_plot(p, s)  # currently broken.  need to split peaks/satellites
plt.plot(data.w, data.V)
plt.plot(res.w, res.V)
plt.show()


print ('\nMoving onto TNC fit:\n')

# Now we will pass global results onto TNC
x0[:3] = res.params[:3]

# fit data
fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)

# generate result
res = fit.generate_result(scale=1)

# plot summary
print_summary(x0, res)

# fit.summary_plot(p, s)
plt.plot(data.w, data.V)
plt.plot(res.w, res.V)
plt.show()
