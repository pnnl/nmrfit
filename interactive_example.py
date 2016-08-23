import NMRfit as nmrft
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.signal
import numpy as np

import scipy.interpolate 


class Peak:
    def __init__(self):
        pass

    def __repr__(self):
        return '''\
               Location: %s
               Height: %s
               Bounds: [%s, %s]
               Sigma: %s
               Area: %s\
               ''' % (self.loc, self.height, self.bounds[0], self.bounds[1], self.sigma, self.area)


class AutoPeakSelector:
    def __init__(self, w, u, v):

        f = sp.interpolate.interp1d(w, u)

        self.w = np.linspace(w.min(), w.max(), int(len(w) * 100))
        self.u = f(self.w)
        self.v = v

        self.u_smoothed = sp.signal.savgol_filter(self.u, 11, 4)

        self.peaks = []

    def findMaxima(self):

        x_spacing = self.w[1] - self.w[0]
        window = int(0.02 / x_spacing)

        idx = sp.signal.argrelmax(self.u_smoothed, order=window)

        u_peaks = self.u[idx]
        w_peaks = self.w[idx]

        for y, x in zip(u_peaks, w_peaks):
            p = Peak()
            p.loc = x
            p.height = y
            self.peaks.append(p)

    def findSigma(self):
        for p in self.peaks:
            d = np.sign(p.height / 2. - self.u[0:-1]) - np.sign(p.height / 2. - self.u[1:])
            rightIdx = np.where(d < 0)[0]  # right
            leftIdx = np.where(d > 0)[0]  # left

            x_right = self.w[rightIdx[np.argmin(np.abs(self.w[rightIdx] - p.loc))]]
            x_left = self.w[leftIdx[np.argmin(np.abs(self.w[leftIdx] - p.loc))]]

            p.bounds = [p.loc - (3 * (p.loc - x_left)), p.loc + (3 * (x_right - p.loc))]
            p.sigma = x_right - x_left

            p.idx = np.where((self.w >= p.bounds[0]) & (self.w <= p.bounds[1]))

            p.area = sp.integrate.simps(self.u[p.idx], self.w[p.idx])

    def findPeaks(self):
        self.findMaxima()
        self.findSigma()

        return self.peaks

inDir = "./Data/blindedData/dc_4h_cdcl3_kilimanjaro_25c_1d_1H_2_050616.fid"

w, u, v, theta0 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# create data object
data = nmrft.Data(w, u, v, theta0)

# bound the data
data.select_bounds(low=3.2, high=3.6)

aps = AutoPeakSelector(data.w, data.V, data.I)
peaks = aps.findPeaks()

plt.plot(aps.w, aps.u_smoothed, color='g')
for p in peaks:
    plt.axvline(x=p.bounds[0], color='b')
    plt.axvline(x=p.bounds[1], color='b')
    plt.scatter(p.loc, p.height, color='r')


# plt.scatter(x, y)
plt.show()
