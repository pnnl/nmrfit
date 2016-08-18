import NMRfit as nmrft
import os


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
inDir = "./Data/blindedData/dc_4h_cdcl3_kilimanjaro_25c_1d_1H_2_050616.fid"

w, u, v, theta0 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# create data object
data = nmrft.Data(w, u, v, theta0)

# bound the data
data.select_bounds(low=3.2, high=3.6)

# interactively select peaks and satellites
p = data.select_peaks(2)
s = data.select_satellites(4)

# initial conditions of the form [theta, r, yOff, sigma_n, mu_n, a_n,...]
x0 = [data.theta, 1., 0.,                                  # shared params
      p[0].width / 10., p[0].loc, p[0].area,       # p1 init
      p[1].width / 10., p[1].loc, p[1].area,       # p2 init
      s[0].width / 10., s[0].loc, s[0].area,       # s1 init
      s[1].width / 10., s[1].loc, s[1].area,       # s2 init
      s[2].width / 10., s[2].loc, s[2].area,       # s3 init
      s[3].width / 10., s[3].loc, s[3].area]       # s4 init

bounds1 = ((None, None), (0., 1.), (None, None),
           (None, None), (x0[4] - 0.05, x0[4] + 0.05), (None, None),
           (None, None), (x0[7] - 0.05, x0[7] + 0.05), (None, None),
           (None, None), (x0[10] - 0.05, x0[10] + 0.05), (None, None),
           (None, None), (x0[13] - 0.05, x0[13] + 0.05), (None, None),
           (None, None), (x0[16] - 0.05, x0[16] + 0.05), (None, None),
           (None, None), (x0[19] - 0.05, x0[19] + 0.05), (None, None))

# fit data
fit = nmrft.FitUtility(data, x0, method='Powell', options=None)

# generate result
res = fit.generate_result(scale=1)

# plot summary
fit.summary_plot(p, s)
print_summary(x0, res)

print ('\nMoving onto TNC fit:\n')

# Now we will pass global results onto TNC
x0[:3] = res.params[:3]

# fit data
fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds1, options=None)

# generate result
res = fit.generate_result(scale=1)

# plot summary
print_summary(x0, res)
fit.summary_plot(p, s)
