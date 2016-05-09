# -*- coding: utf-8 -*-
"""
@author: Ryan Renslow (ryan.renslow@pnnl.gov)
Copyright (c) 2015 Battelle Memorial Institute.
"""

# Import required modules and libraries
import NMRfit_module as nmrft
import os
import numpy as np


# Load and process data
inDir = "./Data/organophosphate/dc-0445_cdcl3_kilimanjaro_22c_1d_1H_1_031816.fid"
w, u, v, p0, p1 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

theta0 = p0

# initiate bounds selection
bs = nmrft.BoundsSelector(w, u, v)
w, u, v = bs.applyBounds()
w, u, v = nmrft.increaseResolution(w, u, v)

# get an approximation of phase shift
V, I = nmrft.shift_phase(u, v, theta0)

# get approximate initial conditions
p1 = nmrft.PeakSelector(w, V, I)
p2 = nmrft.PeakSelector(w, V, I)

s1 = nmrft.PeakSelector(w, V, I)
s2 = nmrft.PeakSelector(w, V, I)
s3 = nmrft.PeakSelector(w, V, I)
s4 = nmrft.PeakSelector(w, V, I)

weights = [['all', 1.0],
           ['all', 1.0],
           [s1.idx, p1.height / s1.height],
           [s2.idx, p1.height / s2.height],
           [s3.idx, p1.height / s3.height],
           [s4.idx, p1.height / s4.height]]

# initial conditions of the form [theta, r, yOff, sigma_n, mu_n, a_n,...]
x0 = [0, 0.1, 0,                            # shared params
      p1.width / 10, p1.loc, p1.area,       # main peak init
      p2.width / 10, p2.loc, p2.area,
      p1.width / 10, s1.loc, s1.area,       # s1 init
      p1.width / 10, s2.loc, s2.area,       # s2 init
      p1.width / 10, s3.loc, s3.area,
      p1.width / 10, s4.loc, s4.area]

# Fit peak and satellites
u_fit, v_fit, fitParams = nmrft.fit_peak(w, u, v, x0, method='Powell', options=None, weights=weights, fitIm=False)

print('theta, r, yOff')
print(fitParams[:3])
print('sigma, mu, a')
print(fitParams[3:6])       # p1 -- 5
print(fitParams[6:9])       # p2 -- 8
print(fitParams[9:12])      # s1 -- 11
print(fitParams[12:15])     # s2 -- 14
print(fitParams[15:18])     # s3 -- 17
print(fitParams[18:21])     # s4 -- 20
nmrft.plot(w, u, u_fit)
# nmrft.plot(w, v, v_fit)

percent = (fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20]) / (fitParams[5] + fitParams[8] + fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20])
print('\narea ratio:', percent)

# noiseS = 3.8
# noiseE = 4.0
# sigma = nmrft.sample_noise(w, u, noiseS, noiseE)

# # # Monte Carlo fitting
# MCsize = 10

# # # raw_input('Initialization done. Press Enter to start simulation...')
# fitstd = np.ones(MCsize)
# fitpercent = np.ones(MCsize)

# for i in range(MCsize):
#     print(i)
#     u_fit, v_fit, params = nmrft.fit_peak(w, nmrft.rnd_data(sigma, u), nmrft.rnd_data(sigma, v), fitParams, method='Powell', weights=weights)

#     fitpercent[i] = (params[8] + params[11]) / (params[8] + params[11] + params[5])
#     fitstd[i] = np.std(fitpercent[:i])
# nmrft.plot(range(2, MCsize), fitstd[2:])

# print('13C Composition: ' + '{0:.4f}'.format(round(percent * 100, 4)) + '%' +
#       ' (std: {0:.5f}'.format(round(fitstd[-1] * 100, 5)) + '%)')
# print('    max observed: ' +
#       '{0:.4f}'.format(round(np.max(fitpercent) * 100, 4)) + '%')
# print('    min observed: ' +
#       '{0:.4f}'.format(round(np.min(fitpercent) * 100, 4)) + '%')
