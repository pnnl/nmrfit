# -*- coding: utf-8 -*-
import numpy as np
import nmrglue as ng
import proc_autophase
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import scipy.integrate
import scipy.interpolate


"""NMR Peak Fitting Module

This module...
attempt at weighted fitting.
"""

__author__ = 'Ryan Renslow (ryan.renslow@pnnl.gov)'
__copyright__ = 'Copyright (c) 2015 PNNL'
__license__ = 'Battelle Memorial Institute BSD-like license'
__version__ = '2.0.0'


def plot(X, Y_re, Y_im=None):
    """Plot function
    """
    # plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(X, Y_re)
    if Y_im is not None:
        plt.plot(X, Y_im)
        Y_min = min(np.min(Y_re), np.min(Y_im))
        Y_max = max(np.max(Y_re), np.max(Y_im))
    else:
        Y_min = min(Y_re)
        Y_max = max(Y_re)
    plt.axis([X[-1], X[0], Y_min - Y_max * 0.05, Y_max * 1.1])
    plt.gca().invert_xaxis()
    plt.show()
    return


def voigt_1D(x, r, yOff, sigma, mu, a):
    '''Fit a Voigt body.
            sigma - modulates the "width" of the peak
            mu    - center of the peak
            a     - area of peak
            r     - Gaussian/Lorentzian ratio
            yOff  - y offset
    '''
    L = (2 / (np.pi * sigma)) * 1 / (1 + ((x - mu) / (0.5 * sigma))**2)
    G = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((x - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)
    V = yOff + a * (r * L + (1 - r) * G)
    return V


def kk_equation(x, r, yOff, sigma, mu, a, w):
    '''The equation inside the integral in the Kramers-Kronig relation.
       Used to evaluate the V->I transform.
    '''
    L1 = (2 / (np.pi * sigma)) * 1 / (1 + ((x + w - mu) / (0.5 * sigma))**2)
    G1 = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((x + w - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)
    V1 = yOff + a * (r * L1 + (1 - r) * G1)

    L2 = (2 / (np.pi * sigma)) * 1 / (1 + ((-x + w - mu) / (0.5 * sigma))**2)
    G2 = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((-x + w - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)
    V2 = yOff + a * (r * L2 + (1 - r) * G2)

    V = 1 / x * (V2 - V1)
    return V


def kk_relation(w, r, yOff, sigma, mu, a):
    '''Performs the integral required of the Kramers-Kronig relation using the kk_equation function
       for a given w.
    '''
    res, err = sp.integrate.quad(kk_equation, 0, np.inf, args=(r, yOff, sigma, mu, a, w))   # , weight='cauchy', wvar=0)
    return res / np.pi


def objective(x0, w, u, v, kk, weights, fitIm):
    '''Fits the data using a least-squares approach.
    '''
    theta, r, yOff = x0[:3]
    x0 = x0[3:]

    V_data = u * np.cos(theta) - v * np.sin(theta)
    V_fit = np.zeros_like(V_data)

    V_residual = 0

    if fitIm is True:
        I_data = u * np.sin(theta) + v * np.cos(theta)
        I_fit = np.zeros_like(V_data)

        I_residual = 0

    for i in range(0, len(x0), 3):
        sigma = x0[i]
        mu = x0[i + 1]
        a = x0[i + 2]

        V_fit = V_fit + voigt_1D(w, r, yOff, sigma, mu, a)
        if fitIm is True:
            I_fit = I_fit + kk(w, r, yOff, sigma, mu, a)

    if weights is None:
        V_residual = np.square(V_data - V_fit).sum(axis=None)
        if fitIm is True:
            I_residual = np.square(I_data - I_fit).sum(axis=None)
    else:
        for q in weights:
            bounds, weight = q
            if bounds == 'all':
                V_residual = V_residual + np.square(V_data - V_fit).sum(axis=0) * weight
                if fitIm is True:
                    I_residual = I_residual + np.square(I_data - I_fit).sum(axis=0) * weight
            else:
                idx = np.where((w > bounds[0]) & (w < bounds[1]))
                V_residual = V_residual + np.square(V_data[idx] - V_fit[idx]).sum(axis=0) * weight
                if fitIm is True:
                    I_residual = I_residual + np.square(I_data[idx] - I_fit[idx]).sum(axis=0) * weight
    if fitIm is True:
        return (V_residual + I_residual) / 2.0
    else:
        return V_residual


def fit_peak(w, u, v, x0, method='Powell', options=None, weights=None, fitIm=False):
    '''Fit a Voigt body using both real and complex data.
            Positional arguments:
                w - the frequency vector
                u - the real component of the power spectrum
                v - the imaginary component of the power spectrum
            Keyword arguments:

    '''
    result = sp.optimize.minimize(objective, x0, args=(w, u, v, kk_relation_vectorized, weights, fitIm), method=method, options=options)

    return result.x, result.fun


def generate_fit(w, u, v, result, scale=4):
    # w, u, v = increase_resolution(w, u, v, f=scale)
    V_fit = np.zeros_like(u)
    I_fit = np.zeros_like(v)

    theta, r, yOff = result[:3]
    res = result[3:]

    for i in range(0, len(res), 3):
        sigma = res[i]
        mu = res[i + 1]
        a = res[i + 2]

        V_fit = V_fit + voigt_1D(w, r, yOff, sigma, mu, a)
        I_fit = I_fit + kk_relation_vectorized(w, r, yOff, sigma, mu, a)

    u_fit = V_fit * np.cos(theta) + I_fit * np.sin(theta)
    v_fit = -V_fit * np.sin(theta) + I_fit * np.cos(theta)

    return w, u, v, u_fit, v_fit


def find_peak(X, re, est, searchwidth=0.5):
    """Find peak within tolerance
    """
    idx = np.where((X <= est + searchwidth) & (X >= est - searchwidth))
    peakestX = X[idx]
    peakestY = re[idx]
    peakindex = np.argmax(peakestY)
    peakloc = peakestX[peakindex]

    return peakestY[peakindex], peakloc


def rnd_data(sigma, origdata):
    """Add random noise
    """
    synthnoise = sigma * np.random.randn(origdata.size)
    synthdata = origdata + synthnoise
    return synthdata


def varian_process(fidfile, procfile):
    """?
    """
    dic, data = ng.varian.read_fid(fidfile)
    procs = ng.varian.read_procpar(procfile)
    offset = [float(i) for i in procs['tof']['values']][0]
    magfreq = [float(i) for i in procs['sfrq']['values']][0]
    rangeHz = [float(i) for i in procs['sw']['values']][0]
    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq
    ppm = np.linspace(rangeppm - offsetppm, -offsetppm, data.size)
    # plt.figure(); plt.plot(data[0,:])
    # process the spectrum
    # data = ng.proc_base.zf_size(data, 32768)    # zero fill to 32768 points
    data = ng.proc_base.fft(data)               # Fourier transform
    data = data / np.max(data)

    # phase correct then manually offset for testing
    p0, p1 = proc_autophase.approximate_phase(data, 'acme')  # auto phase correct
    # data = ng.proc_base.ps(data, p0=48)
    return ppm, data[0, :].real, data[0, :].imag, p0, p1


def sample_noise(X, Y, xstart, xstop):
    """Calculate standard deviation of sample noise.
    """
    noiseY = Y[np.where((X <= xstop) & (X >= xstart))]
    noiseX = X[np.where((X <= xstop) & (X >= xstart))]
    baselinefit = np.poly1d(np.polyfit(noiseX, noiseY, 2))
    noise = noiseY - baselinefit(noiseX)
    return np.std(noise)


class BoundsSelector:
    def __init__(self, w, u, v, supress=False):
        self.u = u
        self.v = v
        self.w = w
        self.supress = supress

        if not self.supress:
            self.fig = plt.figure()  # figsize=(9, 5), dpi=300
            plt.plot(w, u)
            plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
            plt.gca().invert_xaxis()
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
            self.bounds = []
            plt.show()

    def __call__(self, event):
        self.bounds.append(event.xdata)
        if len(self.bounds) == 2:
            plt.close()

    def apply_bounds(self, low=None, high=None):
        if not self.supress:
            low = min(self.bounds)
            high = max(self.bounds)

        idx = np.where((self.w > low) & (self.w < high))

        self.u = self.u[idx]
        self.v = self.v[idx]
        self.w = self.w[idx]
        return self.w, self.u, self.v


class PeakSelector:
    def __init__(self, w, u, v):
        self.u = u
        self.v = v
        self.w = w
        self.fig = plt.figure()  # figsize=(9, 5), dpi=300
        plt.plot(w, u)
        plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
        # plt.gca().invert_xaxis()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        self.points = []
        plt.show()

    def __call__(self, event):
        self.points.append([event.xdata, event.ydata])
        if len(self.points) == 3:
            self.parse_points()
            plt.close()

    def parse_points(self):
        self.points.sort()
        wMin = self.points[0][0]
        wMax = self.points[2][0]
        self.width = wMax - wMin

        peakest = self.points[1][0]

        self.height, self.loc = find_peak(
            self.w, self.u, peakest, searchwidth=self.width / 2.)

        self.idx = np.where((self.w > wMin) & (self.w < wMax))
        self.bounds = [wMin, wMax]

        self.area = sp.integrate.simps(self.u[self.idx], self.w[self.idx])


def increase_resolution(w, u, v, f=4, kind='linear'):
    n = int(len(w) * f)

    new_w = np.linspace(w.min(), w.max(), n)
    new_u = sp.interpolate.interp1d(w, u, kind=kind)(new_w)
    new_v = sp.interpolate.interp1d(w, v, kind=kind)(new_w)

    return new_w, new_u, new_v


def shift_phase(u, v, theta):
    V = u * np.cos(theta) - v * np.sin(theta)
    I = u * np.sin(theta) + v * np.cos(theta)
    return V, I

# the vectorized form can compute the integral for all w
kk_relation_vectorized = np.vectorize(kk_relation, otypes=[np.float])

if __name__ == '__main__':
    print('NMR Peak Fitting Module')
    print('Author: ' + __author__ + '\n')
