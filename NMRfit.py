import numpy as np
import scipy as sp
import scipy.optimize
import nmrglue as ng
import matplotlib.pyplot as plt

from package import proc_autophase
from package.equations import *
from package.utility import *
from package.containers import *


class FitUtility:
    '''
    Interface used to perform a fit of the data.

    Parameters
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I) of the data.
    x0 : list(float)
        Initial conditions for the minimizer.
    method: string, optional
        Determines optimization algorithm to be used for minimization.  Default is "Powell"
    options: dict, optional
        Used to pass additional options to the minimizer.

    Returns
    -------
    None.
    '''

    def __init__(self, data, x0, method='Powell', bounds=None, options=None):
        self.result = Result()
        self.data = data

        # initial condition vector
        self.x0 = x0

        # method used in the minimization step
        self.method = method

        # bounds
        self.bounds = bounds

        # any additional options for the minimization step
        self.options = options

        # call to the fit method
        self.fit()

    def fit(self):
        '''
        Fit a number of Voigt functions to the input data by objective function minimization.  By default, only the real
        component of the data is used when performing the fit.  The imaginary data can be used, but at a severe performance
        penalty (often with little to no gains in goodness of fit).

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        # call to the minimization function
        result = sp.optimize.minimize(objective, self.x0, args=(self.data.w, self.data.u, self.data.v, self.x0), method=self.method, bounds=self.bounds, options=self.options)

        # store the fit parameters and error in the result object
        self.result.params = result.x
        self.result.error = result.fun

    def generate_result(self, scale=10):
        '''
        Uses the output of the fit method to generate results.

        Parameters
        ----------
        scale : float, optional
            Upsample the resolution by this factor when calculating the fits.

        Returns
        -------
        result : instance of Result class
            Container for ndarrays relevant to the fitting process (w, u, v, V, I) of the fit.
        '''

        if scale == 1.0:
            # just use w vector as is
            w = self.data.w
        else:
            # upsample the w vector for plotting
            w = np.linspace(self.data.w.min(), self.data.w.max(), int(scale * self.data.w.shape[0]))

        # initialize arrays for the fit of V and I
        V_fit = np.zeros_like(w)
        I_fit = np.zeros_like(w)

        # extract global params from result object
        theta, r, yOff = self.result.params[:3]
        res = self.result.params[3:]

        # transform u and v to get V and I for the data
        V_data = self.data.u * np.cos(theta) - self.data.v * np.sin(theta)
        I_data = self.data.u * np.sin(theta) + self.data.v * np.cos(theta)

        # iteratively add the contribution of each peak to the fits for V and I
        for i in range(0, len(res), 3):
            sigma = res[i]
            mu = res[i + 1]
            a = res[i + 2]

            V_fit = V_fit + voigt(w, r, yOff, sigma, mu, a)
            I_fit = I_fit + kk_relation_vectorized(w, r, yOff, sigma, mu, a)

        # transform the fits for V and I to get fits for u and v
        u_fit = V_fit * np.cos(theta) + I_fit * np.sin(theta)
        v_fit = -V_fit * np.sin(theta) + I_fit * np.cos(theta)

        # populate the result object
        self.result.u = u_fit
        self.result.v = v_fit
        self.result.V = V_fit
        self.result.I = I_fit
        self.result.w = w

        # update the data object
        self.data.V = V_data
        self.data.I = I_data

        return self.result

    def summary_plot(self):
        peaks, satellites = self.data.peaks.split()

        peakBounds = []
        for p in peaks:
            low, high = p.bounds
            peakBounds.append(low)
            peakBounds.append(high)

        peakRange = [min(peakBounds), max(peakBounds)]

        set1Bounds = []
        set2Bounds = []
        for s in satellites:
            low, high = s.bounds
            if high < peakRange[0]:
                set1Bounds.append(low)
                set1Bounds.append(high)
            else:
                set2Bounds.append(low)
                set2Bounds.append(high)

        set1Range = [min(set1Bounds), max(set1Bounds)]
        set2Range = [min(set2Bounds), max(set2Bounds)]

        totalIdxData = np.where((self.data.w >= set1Range[0]) & (self.data.w <= set2Range[1]))
        totalIdxFit = np.where((self.result.w >= set1Range[0]) & (self.result.w <= set2Range[1]))

        peakIdxData = np.where((self.data.w >= peakRange[0]) & (self.data.w <= peakRange[1]))
        peakIdxFit = np.where((self.result.w >= peakRange[0]) & (self.result.w <= peakRange[1]))

        set1IdxData = np.where((self.data.w >= set1Range[0]) & (self.data.w <= set1Range[1]))
        set1IdxFit = np.where((self.result.w >= set1Range[0]) & (self.result.w <= set1Range[1]))

        set2IdxData = np.where((self.data.w >= set2Range[0]) & (self.data.w <= set2Range[1]))
        set2IdxFit = np.where((self.result.w >= set2Range[0]) & (self.result.w <= set2Range[1]))

        # set up figures
        # real
        fig_re = plt.figure(1)
        ax1_re = plt.subplot(211)
        ax2_re = plt.subplot(234)
        ax3_re = plt.subplot(235)
        ax4_re = plt.subplot(236)

        # plot everything
        ax1_re.plot(self.data.w[totalIdxData], self.data.V[totalIdxData])
        ax1_re.plot(self.result.w[totalIdxFit], self.result.V[totalIdxFit])

        # plot left sats
        ax2_re.plot(self.data.w[set1IdxData], self.data.V[set1IdxData])
        ax2_re.plot(self.result.w[set1IdxFit], self.result.V[set1IdxFit])

        # plot main peaks
        ax3_re.plot(self.data.w[peakIdxData], self.data.V[peakIdxData])
        ax3_re.plot(self.result.w[peakIdxFit], self.result.V[peakIdxFit])

        # plot right satellites
        ax4_re.plot(self.data.w[set2IdxData], self.data.V[set2IdxData])
        ax4_re.plot(self.result.w[set2IdxFit], self.result.V[set2IdxFit])

        # imag
        fig_im = plt.figure(2)
        ax1_im = plt.subplot(211)
        ax2_im = plt.subplot(234)
        ax3_im = plt.subplot(235)
        ax4_im = plt.subplot(236)

        # plot everything
        ax1_im.plot(self.data.w[totalIdxData], self.data.I[totalIdxData])
        ax1_im.plot(self.result.w[totalIdxFit], self.result.I[totalIdxFit])

        # plot left sats
        ax2_im.plot(self.data.w[set1IdxData], self.data.I[set1IdxData])
        ax2_im.plot(self.result.w[set1IdxFit], self.result.I[set1IdxFit])

        # plot main peaks
        ax3_im.plot(self.data.w[peakIdxData], self.data.I[peakIdxData])
        ax3_im.plot(self.result.w[peakIdxFit], self.result.I[peakIdxFit])

        # plot right satellites
        ax4_im.plot(self.data.w[set2IdxData], self.data.I[set2IdxData])
        ax4_im.plot(self.result.w[set2IdxFit], self.result.I[set2IdxFit])

        # display
        plt.tight_layout()
        plt.show()

    def print_summary(self):
        # print summary
        print()
        print('SEED PARAMETER VALUES:')
        print('----------------------')

        for i in range(0, len(self.x0), 3):
            lblpars(i)
            print(self.x0[i:i + 3])

        print()
        print('CONVERGED PARAMETER VALUES:')
        print('---------------------------')

        for i in range(0, len(self.result.params), 3):
            lblpars(i)
            print(self.result.params[i:i + 3])

        print("Error:  ", self.result.error)


def lblpars(i):
    if (i == 0):
        print ('Global Parameters:')
    elif (i == 3):
        print ('Major Peak Parameters:')
    elif (i == 9):
        print ('Minor Peak Parameters:')


def varian_process(fidfile, procfile):
    """
    Parameters
    ----------
    fidfile : string
        Path to the fid file.
    procfile : string
        Path to the procpar file.

    Returns
    -------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    p0 : float
        Zero order phase in radians.
    """

    dic, data = ng.varian.read_fid(fidfile)
    procs = ng.varian.read_procpar(procfile)

    offset = [float(i) for i in procs['tof']['values']][0]
    magfreq = [float(i) for i in procs['sfrq']['values']][0]
    rangeHz = [float(i) for i in procs['sw']['values']][0]

    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq

    w = np.linspace(rangeppm - offsetppm, -offsetppm, data.size)

    data = ng.proc_base.fft(data)               # Fourier transform
    data = data / np.max(data)

    # phase correct then manually offset for testing
    p0, p1 = proc_autophase.approximate_phase(data, 'acme')  # auto phase correct

    u = data[0, :].real
    v = data[0, :].imag

    return Data(w[::-1], u[::-1], v[::-1], p0)
