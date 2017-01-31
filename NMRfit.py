import numpy as np
import scipy as sp
import scipy.optimize
import nmrglue as ng
import matplotlib.pyplot as plt

from package import proc_autophase
from package import equations
from package import containers


class FitUtility:
    """
    Interface used to perform a fit of the data.

    Attributes
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    result : instance of Result class
        Container for ndarrays (w, u, v, V, I) of the fit result.
    x0 : list of floats
        Initial conditions for the minimizer.
    method: string
        Determines optimization algorithm to be used for minimization.  Default is "Powell".
    bounds : list of 2-tuples
        Min, max bounds for each parameter in x0.
    options : dict
        Additional options for the minimizer.
    weights : ndarray
        Array giving frequency-dependent weighting of error.

    """

    def __init__(self, data, x0, method='Powell', bounds=None, options=None):
        """
        FitUtility constructor.

        Parameters
        ----------
        data : instance of Data class
            Container for ndarrays relevant to the fitting process (w, u, v, V, I).
        x0 : list of floats
            Initial conditions for the minimizer.
        method : string, optional
            Determines optimization algorithm to be used for minimization.  Default is "Powell".
        bounds : list of 2-tuples
            Min, max bounds for each parameter in x0.
        options : dict, optional
            Used to pass additional options to the minimizer.

        """
        self.result = containers.Result()
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
        """
        Fit a number of Voigt functions to the input data by objective function minimization.  By default, only the real
        component of the data is used when performing the fit.  The imaginary data can be used, but at a severe performance
        penalty (often with little to no gains in goodness of fit).

        """
        self.weights = self.compute_weights()

        # call to the minimization function
        result = sp.optimize.minimize(equations.objective, self.x0, args=(self.data.w, self.data.u, self.data.v, self.x0, self.weights, self.data.roibounds),
                                      method=self.method, bounds=self.bounds, options=self.options)

        # store the fit parameters and error in the result object
        self.result.params = result.x
        self.result.error = result.fun

    def compute_weights(self, expon=0.5):
        """
        Smoothly weights each peak based on its height relative to the largest peak.

        Parameters
        ----------
        expon : float
            Raise relative weighting to this power.

        Returns
        -------
        weights : ndarray
            Array giving frequency-dependent weighting of error.

        """
        lIdx = np.zeros(len(self.data.peaks), dtype=np.int)
        rIdx = np.zeros(len(self.data.peaks), dtype=np.int)
        maxabs = np.zeros(len(self.data.peaks))

        for i, p in enumerate(self.data.peaks):
            lIdx[i] = np.argmin(np.abs(self.data.w - p.bounds[0]))
            rIdx[i] = np.argmin(np.abs(self.data.w - p.bounds[1]))
            if lIdx[i] > rIdx[i]:
                temp = lIdx[i]
                lIdx[i] = rIdx[i]
                rIdx[i] = temp

            maxabs[i] = np.abs(p.height)

        biggest = np.amax(maxabs)

        defaultweight = 0.1
        weights = np.ones(len(self.data.w)) * defaultweight

        for i in range(len(self.data.peaks)):
            weights[lIdx[i]:rIdx[i] + 1] = np.power(biggest / maxabs[i], expon)

        weights = equations.laplace1d(weights)
        return weights

    def generate_result(self, scale=10):
        """
        Uses the output of the fit method to generate results.

        Parameters
        ----------
        scale : float, optional
            Upsample the resolution by this factor when calculating the fits.

        Returns
        -------
        result : instance of Result class
            Container for ndarrays (w, u, v, V, I) of the fit result.

        """
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
            width = res[i]
            loc = res[i + 1]
            a = res[i + 2]

            V_fit = V_fit + equations.voigt(w, r, yOff, width, loc, a)
            I_fit = I_fit + equations.kk_relation_vectorized(w, r, yOff, width, loc, a)

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

        # calculate area fraction
        self.result.area_fraction = self.calculate_area_fraction()

        return self.result

    def calculate_area_fraction(self):
        """
        Calculates the relative fraction of the satellite peaks to the total peak area from the fit.

        Returns
        -------
        area_fraction : float
            Area fraction of satellite peaks.

        """
        areas = np.array([self.result.params[i] for i in range(5, len(self.result.params), 3)])
        m = np.mean(areas)
        peaks = areas[areas >= m].sum()
        sats = areas[areas < m].sum()

        area_fraction = (sats / (peaks + sats))

        return area_fraction

    def summary_plot(self):
        """
        Generates a summary plot of the calculated fit alongside the input data.

        """
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

        # set up figures
        # real
        fig_re = plt.figure(1)
        ax1_re = plt.subplot(211)
        ax2_re = plt.subplot(234)
        ax3_re = plt.subplot(235)
        ax4_re = plt.subplot(236)

        # plot everything
        ax1_re.plot(self.data.w, self.data.V)
        ax1_re.plot(self.result.w, self.result.V)
        ax1_re.autoscale_view()

        # plot left sats
        ax2_re.plot(self.data.w, self.data.V)
        ax2_re.plot(self.result.w, self.result.V)
        ax2_re.autoscale_view()
        ax2_re.set_xlim(set1Range)

        # plot main peaks
        ax3_re.plot(self.data.w, self.data.V)
        ax3_re.plot(self.result.w, self.result.V)
        ax3_re.autoscale_view()
        ax3_re.set_xlim(peakRange)

        # plot right satellites
        ax4_re.plot(self.data.w, self.data.V)
        ax4_re.plot(self.result.w, self.result.V)
        ax4_re.autoscale_view()
        ax4_re.set_xlim(set2Range)

        # imag
        fig_im = plt.figure(2)
        ax1_im = plt.subplot(211)
        ax2_im = plt.subplot(234)
        ax3_im = plt.subplot(235)
        ax4_im = plt.subplot(236)

        # plot everything
        ax1_im.plot(self.data.w, self.data.I)
        ax1_im.plot(self.result.w, self.result.I)
        ax1_im.autoscale_view(tight=True)

        # plot left sats
        ax2_im.plot(self.data.w, self.data.I)
        ax2_im.plot(self.result.w, self.result.I)
        ax2_im.autoscale_view(tight=True)
        ax2_im.set_xlim(set1Range)

        # plot main peaks
        ax3_im.plot(self.data.w, self.data.I)
        ax3_im.plot(self.result.w, self.result.I)
        ax3_im.autoscale_view(tight=True)
        ax3_im.set_xlim(peakRange)

        # plot right satellites
        ax4_im.plot(self.data.w, self.data.I)
        ax4_im.plot(self.result.w, self.result.I)
        ax4_im.autoscale_view(tight=True)
        ax4_im.set_xlim(set2Range)

        # display
        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """
        Generates and prints a summary of the fitting process.

        """
        x0 = np.array(self.x0).reshape((-1, 3))
        x0_globals = x0[0, :]
        x0 = x0[1:, :]
        res = np.array(self.result.params).reshape((-1, 3))
        res_globals = res[0, :]
        res = res[1:, :]

        idx = x0[:, 1].argsort()[::-1]
        x0 = x0[idx]
        res = res[idx]

        # print summary
        print()
        print('SEED PARAMETER VALUES:')
        print('----------------------')
        print('Global parameters')
        print(x0_globals)
        print('Peak parameters')
        for i in range(x0.shape[0]):
            print(x0[i, :])

        print()
        print('CONVERGED PARAMETER VALUES:')
        print('---------------------------')
        print('Global parameters')
        print(res_globals)
        print('Peak parameters')
        for i in range(res.shape[0]):
            print(res[i, :])

        print("Error:  ", self.result.error)
        print("Area fraction:  ", self.result.area_fraction)

    def summary(self, plot=True):
        """
        Convenience function to print a summary as well as display summary plots.

        Parameters
        ----------
        plot : bool, optional
            Signals whether a plot of the resulting fit will be generated.

        """
        self.print_summary()
        if plot is True:
            self.summary_plot()


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
    result : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).

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

    result = containers.Data(w[::-1], u[::-1], v[::-1], p0)
    return result
