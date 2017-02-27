import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate
import peakutils
import pyswarm
import multiprocessing as mp
import pandas as pd

from . import containers
from . import equations
from . import proc_autophase


class FitUtility:
    """
    Interface used to perform a fit of the data.

    Attributes
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    lower, upper : list of floats
            Min, max bounds for each parameter in the optimization.
    expon : float
        Raise relative weighting to this power.
    fit_im : bool
        Specify whether the imaginary part of the spectrum will be fit. Computationally expensive.
    options : dict, optional
        Used to pass additional options to the minimizer.
    weights : ndarray
        Array giving frequency-dependent weighting of error.
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    V, I : ndarray
        Arrays of the phase corrected real and imaginary components of the frequency response.
    real_contribs, imag_contribs : list of ndarrays
        List containing the individual contributions of each peak to the real and imaginary components, respectively.
    params : ndarray
        Solution vector.
    error : float
        Weighted sum of squared error between the data and fit.

    """

    def __init__(self, data, lower, upper, expon=0.5, fit_im=False, summary=True, options={}):
        """
        FitUtility constructor.

        Parameters
        ----------
        data : instance of Data class
            Container for ndarrays relevant to the fitting process (w, u, v, V, I).
        lower, upper : list of floats
            Min, max bounds for each parameter in the optimization.
        expon : float
            Raise relative weighting to this power.
        fit_im : bool
            Specify whether the imaginary part of the spectrum will be fit. Computationally expensive.
        summary : bool
            Flag to display a summary of the fit.
        options : dict, optional
            Used to pass additional options to the minimizer.

        """
        # store init attributes
        self.data = data
        self.lower = lower
        self.upper = upper
        self.fit_im = fit_im
        self.expon = expon
        self.summary = summary
        self.options = options

    def fit(self):
        """
        Fit a number of Voigt functions to the input data by objective function minimization.  By default, only the real
        component of the data is used when performing the fit.  The imaginary data can be used, but at a severe performance
        penalty (often with little to no gains in goodness of fit).

        """
        self.weights = self._compute_weights()

        # call to the minimization function
        xopt, fopt = pyswarm.pso(equations.objective, self.lower, self.upper, args=(self.data.w, self.data.u, self.data.v, self.weights, self.fit_im),
                                 swarmsize=self.options.get('swarmsize', 204),
                                 maxiter=self.options.get('maxiter', 2000),
                                 omega=self.options.get('omega', -0.2134),
                                 phip=self.options.get('phip', -0.3344),
                                 phig=self.options.get('phig', 2.3259),
                                 processes=self.options.get('processes', mp.cpu_count() - 1))

        # store the fit parameters and error in the result object
        self.params = xopt
        self.error = fopt

        if self.summary is True:
            self._print_summary()

    def _compute_weights(self):
        """
        Smoothly weights each peak based on its height relative to the largest peak.

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
            weights[lIdx[i]:rIdx[i] + 1] = np.power(biggest / maxabs[i], self.expon)

        weights = equations.laplace1d(weights)
        return weights

    def generate_result(self, scale=10):
        """
        Uses the output of the fit method to generate results.

        Parameters
        ----------
        scale : float, optional
            Upsample the resolution by this factor when calculating the fits.

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
        p0, p1, r, yoff = self.params[:4]
        res = self.params[4:]

        # phase shift data by fit theta
        self.data.shift_phase(method='manual', p0=p0, p1=p1)

        # iteratively add the contribution of each peak to the fits for V and I
        real_contribs = []
        imag_contribs = []
        for i in range(0, len(res), 3):
            width = res[i]
            loc = res[i + 1]
            a = res[i + 2]

            real = equations.voigt(w, r, yoff, width, loc, a)
            imag = equations.kk_relation_parallel(w, r, yoff, width, loc, a)

            real_contribs.append(real)
            imag_contribs.append(imag)

            V_fit = V_fit + real
            I_fit = I_fit + imag

        # transform the fits for V and I to get fits for u and v
        u_fit, v_fit = proc_autophase.ps2(V_fit, I_fit, inv=True, p0=p0, p1=p1)
        # u_fit = V_fit * np.cos(theta) + I_fit * np.sin(theta)
        # v_fit = -V_fit * np.sin(theta) + I_fit * np.cos(theta)

        # populate result attributes
        self.u = u_fit
        self.v = v_fit
        self.V = V_fit
        self.I = I_fit
        self.w = w
        self.real_contribs = real_contribs
        self.imag_contribs = imag_contribs

    def calculate_area_fraction(self):
        """
        Calculates the relative fraction of the satellite peaks to the total peak area from the fit.

        """
        areas = np.array([self.params[i] for i in range(6, len(self.params), 3)])
        m = np.mean(areas)
        peaks = areas[areas >= m].sum()
        sats = areas[areas < m].sum()

        area_fraction = (sats / (peaks + sats))

        # calculate area fraction
        return area_fraction

    def _print_summary(self):
        """
        Generates and prints a summary of the fitting process.

        """
        res = np.array(self.params)
        res_globals = pd.DataFrame(res[:4].reshape((1, -1)), columns=['p0', 'p1', 'g-l ratio', 'y-off'])
        res = pd.DataFrame(res[4:].reshape((-1, 3)), columns=['width', 'location', 'area'])

        print()
        print('CONVERGED PARAMETER VALUES:')
        print('---------------------------')
        print('Global parameters')
        print(res_globals.to_string(index=False))
        print('Peak parameters')
        print(res.to_string(index=False))
        print("Error:\t", self.error)


class BoundsSelector:
    """
    Interactive utility used to bound the spectroscopy data.  The user clicks twice on a plot to
    indicate the lower and upper bounds in the frequency domain.

    Attributes
    ----------
    w : ndarray
            Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    supress : bool, optional
        Flag to specify whether the interactive portion will be invoked.
    bounds : list of floats of length 2
        Upper and lower bound of the region of interest.
    fig : PyPlot figure
    cid : figure canvas socket

    """

    def __init__(self, w, u, v, supress=False):
        """
        Constructor for BoundsSelector class.

        Parameters
        ----------
        w : ndarray
            Array of frequency data.
        u, v : ndarray
            Arrays of the real and imaginary components of the frequency response.
        supress : bool, optional
            Flag to specify whether the interactive portion will be invoked.

        """
        self.u = u
        self.v = v
        self.w = w
        self.supress = supress

        if not self.supress:
            self.fig = plt.figure()  # figsize=(9, 5), dpi=300
            plt.plot(w, u, linewidth=2)
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
            self.bounds = []
            plt.show()

    def __call__(self, event):
        """
        Called on mouse click events to capture location.

        Parameters
        ----------
        event : mouse click event
            Contains mouse click information (e.g., x, y location)

        """
        self.bounds.append(event.xdata)
        if len(self.bounds) == 2:
            plt.close()

    def apply_bounds(self, low=None, high=None):
        """
        Applies boundaries determined interactively, or passed via low and high.

        Parameters
        ----------
        low : float, optional
            Lower bound.
        high : float, optional
            Upper bound.

        Returns
        -------
        w, u, v : ndarray
            Arrays containing the bounded data.

        """
        if not self.supress:
            low = min(self.bounds)
            high = max(self.bounds)

        idx = np.where((self.w > low) & (self.w < high))

        self.u = self.u[idx]
        self.v = self.v[idx]
        self.w = self.w[idx]
        return self.w, self.u, self.v


class PeakSelector:
    """
    Interactive utility used to identify peaks and calculate approximations to peak height, width, and area.

    Attributes
    ----------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    n : int
        Number of peaks to select.
    peaks : instance of Peaks object
    points : list of lists
        List containing x, y points for each mouse click.
    fig : PyPlot figure
    cid : figure canvas socket
    baseline : ndarray
        Array that defines a piecewise polynomial baseline of the data.

    """

    def __init__(self, w, u, n, piecewise_baseline=False, one_click=False):
        """
        PeakSelector constructor.

        Parameters
        ----------
        w : ndarray
            Array of frequency data.
        u, v : ndarray
            Arrays of the real and imaginary components of the frequency response.
        n : int
            Number of peaks to select.
        piecewise_baseline : bool, optional
            Specify whether baseline correction is performed.
        one_click : bool, optional
            Specify whether a single click will be used to select peaks (as opposed to two)

        """
        self.u = u
        self.w = w
        self.n = n
        self.one_click = one_click

        # peak container
        self.peaks = containers.Peaks()

        # empty list to store point information from clicks
        self.points = []

        # initialize plot
        self.fig = plt.figure()
        plt.plot(w, u, linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')

        # start event listener
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

        if piecewise_baseline is True:
            self.baseline = _piecewise_baseline(self.w, self.u)
        else:
            self.baseline = peakutils.baseline(self.u, 0)

        # display the plot
        plt.show()

    def __call__(self, event):
        """
        Called whenever the user clicks on the plot.  Stores x and y location of the cursor.
        After 2 clicks, the plot is closed as the peak has been "defined."

        Parameters
        ----------
        event : mouse click event
            Contains mouse click information (e.g., x, y location)

        """
        # add x,y location of click
        self.points.append([event.xdata, event.ydata])

        if self.one_click is True:
            if len(self.points) >= self.n:
                plt.close()
                self.parse_points2()
                self.find_width()

        else:
            if (len(self.points) % 2 == 0) and (len(self.peaks) < self.n):
                # add the peak
                self.parse_points()

                # clear the list of points
                self.points = []

                if len(self.peaks) >= self.n:
                    plt.close()

    def parse_points2(self):
        """
        Determines approximate peak height and location based on single-click selection method.

        """
        for x, y in self.points:
            p = containers.Peak()
            p.loc = x
            p.i = np.argmin(np.abs(self.w - p.loc))
            p.height = self.u[p.i] - self.baseline[p.i]

            self.peaks.append(p)

    def find_width(self):
        """
        Using peak information, finds FWHM.

        """
        screened_peaks = containers.Peaks()
        for p in self.peaks:
            d = np.sign(p.height / 2. - (self.u[0:-1] - self.baseline[0:-1])) - np.sign(p.height / 2. - (self.u[1:] - self.baseline[1:]))
            rightIdx = np.where(d < 0)[0]  # right
            leftIdx = np.where(d > 0)[0]  # left

            x_right = self.w[rightIdx[np.argmin(np.abs(self.w[rightIdx] - p.loc))]]
            x_left = self.w[leftIdx[np.argmin(np.abs(self.w[leftIdx] - p.loc))]]

            if x_left < x_right:
                # width equals FWHM
                p.width = x_right - x_left

                # bounds are +/- 2 widths
                p.bounds = [p.loc - 2 * p.width, p.loc + 2 * p.width]

                # peak indices
                p.idx = np.where((self.w >= p.bounds[0]) & (self.w <= p.bounds[1]))

                # area over peak indices
                p.area = sp.integrate.simps(self.u[p.idx] - self.baseline[p.idx], self.w[p.idx])

                screened_peaks.append(p)

        self.peaks = screened_peaks

    def parse_points(self):
        """
        Called after 2 clicks on the plot.  Sorts the stored points in terms of frequency (w) to define
        low, middle, and high.  Subsequently determines approximate peak height, width, and area.

        """
        peak = containers.Peak()

        # sort points in frequency
        self.points.sort()

        # determine minimum and maximum
        wMin = self.points[0][0]
        wMax = self.points[1][0]

        # determine width from min and max
        # user captures +/- 3 FWHMs with clicks
        peak.width = (wMax - wMin) / 4

        # determine peak height and location of peak by searching over an interval
        peak.height, peak.loc, peak.i = find_peak(self.w, self.u, wMin, wMax)

        # bounds are +/- 3 widths
        peak.bounds = [peak.loc - 2 * peak.width, peak.loc + 2 * peak.width]

        peak.height = peak.height - self.baseline[peak.i]

        # determine indices within the peak width
        peak.idx = np.where((self.w > peak.bounds[0]) & (self.w < peak.bounds[1]))

        # calculate AUC over the width of the peak numerically
        peak.area = sp.integrate.simps(self.u[peak.idx] - self.baseline[peak.idx], self.w[peak.idx])

        self.peaks.append(peak)

    def plot(self):
        """
        Plots the result of the peak selection process to indicate detected peak locations and bounds.

        """
        plt.figure(figsize=(9, 5))
        plt.plot(self.w, self.u, color='b', linewidth=2)
        for p in self.peaks:
            plt.scatter(p.loc, p.height + self.baseline[p.i], color='r')
            plt.axvline(p.bounds[0], color='g')
            plt.axvline(p.bounds[1], color='g')

        plt.xlabel('Frequency', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.show()


class AutoPeakSelector:
    """
    Automatic utility used to identify peaks and calculate approximations to peak height, width, and area.
    Uses local non-maxima supression to find relative maxima (peaks) and FWHM analysis to determine an
    approximation of width/width.

    Attributes
    ----------
    w : ndarray
        Array of frequency data.
    u : ndarray
        Array of the real component of the frequency response.
    thresh : float
        Threshold for minimum amplitude cutoff in peak selection.
    window : float
        Window size for local non-maximum supression.
    piecewise_baseline : bool, optional
            Specify whether baseline correction is performed.
    peaks : instance of Peaks object
    u_smoothed : ndarray
        Smoothed version of u.
    baseline : ndarray
        Array that defines a piecewise polynomial baseline of the data.

    """

    def __init__(self, w, u, thresh, window, piecewise_baseline=False):
        """
        AutoPeakSelector constructor.

        Parameters
        ----------
        w : ndarray
            Array of frequency data.
        u : ndarray
            Array of the real component of the frequency response.
        thresh : float
            Threshold for minimum amplitude cutoff in peak selection.
        window : float
            Window size for local non-maximum supression.
        piecewise_baseline : bool, optional
            Specify whether baseline correction is performed.

        """
        self.thresh = thresh
        self.window = window
        f = sp.interpolate.interp1d(w, u)

        self.w = np.linspace(w.min(), w.max(), int(len(w) * 100))  # arbitrary upsampling
        self.u = f(self.w)

        self.u_smoothed = sp.signal.savgol_filter(self.u, 11, 4)

        if piecewise_baseline is True:
            self.baseline = _piecewise_baseline(self.w, self.u_smoothed)
        else:
            self.baseline = peakutils.baseline(self.u_smoothed, 0)

        self.peaks = containers.Peaks()

    def find_maxima(self):
        """
        Local non-maxima supression to find peaks.

        """
        x_spacing = self.w[1] - self.w[0]
        window = int(self.window / x_spacing)  # arbitrary spacing (0.02)

        idx = sp.signal.argrelmax(self.u_smoothed, order=window)[0]

        for i in idx:
            p = containers.Peak()
            p.loc = self.w[i]
            p.i = i
            p.height = self.u[i] - self.baseline[i]
            if p.height > self.thresh:
                self.peaks.append(p)

    def find_width(self):
        """
        Using peak information, finds FWHM.

        """
        screened_peaks = containers.Peaks()
        for p in self.peaks:
            d = np.sign(p.height / 2. - (self.u[0:-1] - self.baseline[0:-1])) - np.sign(p.height / 2. - (self.u[1:] - self.baseline[1:]))
            rightIdx = np.where(d < 0)[0]  # right
            leftIdx = np.where(d > 0)[0]  # left

            x_right = self.w[rightIdx[np.argmin(np.abs(self.w[rightIdx] - p.loc))]]
            x_left = self.w[leftIdx[np.argmin(np.abs(self.w[leftIdx] - p.loc))]]

            if x_left < x_right:
                # width equals FWHM
                p.width = x_right - x_left

                # bounds are +/- 2 widths
                p.bounds = [p.loc - 2 * p.width, p.loc + 2 * p.width]

                # peak indices
                p.idx = np.where((self.w >= p.bounds[0]) & (self.w <= p.bounds[1]))

                # area over peak indices
                p.area = sp.integrate.simps(self.u[p.idx] - self.baseline[p.idx], self.w[p.idx])

                screened_peaks.append(p)

        self.peaks = screened_peaks

    def find_peaks(self):
        """
        Convenience function to call both peak detection and FWHM analysis methods
        in appropriate order.

        """
        self.find_maxima()
        self.find_width()

    def plot(self):
        """
        Plots the result of the peak selection process to indicate detected peak locations and bounds.

        """
        plt.figure(figsize=(9, 5))
        plt.plot(self.w, self.u, color='b', linewidth=2)
        for p in self.peaks:
            plt.scatter(p.loc, p.height + self.baseline[p.i], color='r')
            plt.axvline(p.bounds[0], color='g')
            plt.axvline(p.bounds[1], color='g')

        plt.xlabel('Frequency', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)

        plt.show()


def find_peak(x, y, low, high):
    """
    Find peak within tolerance, as well as maximum value.

    Parameters
    ----------
    x, y : ndarray
        x and y components of the data being searched.
    low, high : float
        Upper and lower bounds of the search window, respectively.

    Returns
    -------
    peakheight : float
        Height of the located peak.
    peakloc : float
        Location of the peak in terms of x.
    peakindex : int
        Index of the peak location.

    """
    # indices of frequency values around the estimate within the tolerance
    idx = np.where((x <= high) & (x >= low))

    # x and y for these indices
    peakestX = x[idx]
    peakestY = y[idx]

    # determine peak index and location
    peakindex = np.argmax(peakestY)

    peakloc = peakestX[peakindex]
    peakheight = peakestY[peakindex]

    return peakheight, peakloc, peakindex


def rnd_data(width, origdata):
    """
    Add normally distributed noise.

    Parameters
    ----------
    width : float
        Magnitude of noise distribution.
    origdata : ndarray
        Data to which noise will be added.

    Returns
    -------
    synthdata : ndarray
        Input data plus random noise.

    """
    synthnoise = width * np.random.randn(origdata.size)
    synthdata = origdata + synthnoise
    return synthdata


def sample_noise(X, Y, xstart, xstop):
    """
    Calculate standard deviation of sample noise.

    Parameters
    ----------
    X, Y : ndarray
        Arrays containing the x and y components of a signal, respectively.
    xstart, xstop: float
        Start and stop points that define the sample range in terms of X.

    Returns
    -------
    noise : float
        Magnitude of the noise associated with the sampled signal.

    """
    noiseY = Y[np.where((X <= xstop) & (X >= xstart))]
    noiseX = X[np.where((X <= xstop) & (X >= xstart))]

    baselinefit = np.poly1d(np.polyfit(noiseX, noiseY, 2))

    noise = noiseY - baselinefit(noiseX)

    return np.std(noise)


def _piecewise_baseline(x, y, plot=False):
    """
    Calculates a piecewise baseline from the x/y data.  Splits the data into thirds and fits a baseline
    to each section.  Used to correct for baseline offsest during initial condition selection.

    Parameters
    ----------
    x, y : ndarray
        x and y components of the data.

    Returns
    -------
    baseline : ndarray
        Array of y values representing the baseline.  Same shape as x, y.

    """
    third = int(x.shape[0] / 3)

    y1 = y[0:third]
    y2 = y[third:2 * third]
    y3 = y[2 * third:]

    base1 = peakutils.baseline(y1, 2)
    # base2 = np.ones(y2.shape) * np.median(y2)

    base3 = peakutils.baseline(y3, 2)

    base2 = np.linspace(base1[-1], base3[0], y2.size)

    baseline = np.concatenate((base1, base2, base3))

    if plot is True:
        plt.close()
        plt.plot(x, y)
        plt.plot(x, baseline)
        # plt.plot(x, y - baseline)
        plt.show()
        plt.close()

    return baseline
