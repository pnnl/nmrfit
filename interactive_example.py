import NMRfit as nmrft
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.signal

class AutoPeakSelector:
    def __init__(self, w, u, v):
        self.w = w
        self.u = u
        self.v = v

        self.u_smoothed = sp.signal.savgol_filter(self.u, 11, 4)

    def findMaxima(self):

        idx = sp.signal.argrelmax(self.u_smoothed, order=10)

        u_peaks = self.u[idx]
        w_peaks = self.w[idx]

        plt.plot(self.w, smoothed)
        plt.scatter(w_peaks, u_peaks)
        plt.show()

        return w_peaks, u_peaks


inDir = "./Data/blindedData/dc_4h_cdcl3_kilimanjaro_25c_1d_1H_2_050616.fid"

w, u, v, theta0 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# create data object
data = nmrft.Data(w, u, v, theta0)

# bound the data
data.select_bounds(low=3.2, high=3.6)

aps = AutoPeakSelector(data.w, data.V, data.I)
x, y = aps.findMaxima()


