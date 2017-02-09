import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate
import peakutils


class Peaks(list):
    """
    Extension of list object that stores a number of Peak instances.

    """

    def average_height(self):
        """
        Calculate average height of all peaks stored.

        Returns
        -------
        average : float
            Average height of peaks.

        """
        h = 0.
        for p in self:
            h += abs(p.height)
        return h / len(self)

    def split(self):
        """
        Split peaks into two sublists (peaks, satellites) based on relative heights.

        Returns
        -------
        peaks, sats : Peak instances
            Peak lists containing peaks and satellites, respectively.

        """
        h = self.average_height()
        sats = Peaks()
        peaks = Peaks()

        for p in self:
            if abs(p.height) >= h:
                peaks.append(p)
            else:
                sats.append(p)

        return peaks, sats


class Peak:
    """
    Contains metadata for 'peaks' observed in NMR spectroscopy.

    Attributes
    ----------
    loc : float
        Location of the peak.
    height : float
        Height of the peak in terms of signal intensity at its center.
    bounds : list of floats.
        Upper and lower bounds of the peak.  Captures 4 FWHMs.
    width : float
        The width of the peak in terms of FWHM.
    area : float
        Approximation of area of the peak.

    """

    def __repr__(self):
        """
        Overrides __repr__ to print peak-relevant information.

        Returns
        -------
        repr : string
            Formatted string to display peak information.

        """
        return """\
               Location: %s
               Height: %s
               Bounds: [%s, %s]
               Width: %s
               Area: %s\
               """ % (self.loc, self.height, self.bounds[0], self.bounds[1], self.width, self.area)


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

    def __init__(self, w, u, n):
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

        """
        self.u = u
        self.w = w
        self.n = n

        # peak container
        self.peaks = Peaks()

        # empty list to store point information from clicks
        self.points = []

        # initialize plot
        self.fig = plt.figure()
        plt.plot(w, u, linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')

        # start event listener
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

        self.baseline = piecewise_baseline(self.w, self.u)

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

        if (len(self.points) % 2 == 0) and (len(self.peaks) < self.n):
            # add the peak
            self.parse_points()

            # clear the list of points
            self.points = []

            if len(self.peaks) >= self.n:
                plt.close()

    def parse_points(self):
        """
        Called after 2 clicks on the plot.  Sorts the stored points in terms of frequency (w) to define
        low, middle, and high.  Subsequently determines approximate peak height, width, and area.

        """
        peak = Peak()

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
    peaks : instance of Peaks object
    u_smoothed : ndarray
        Smoothed version of u.
    baseline : ndarray
        Array that defines a piecewise polynomial baseline of the data.

    """

    def __init__(self, w, u, thresh, window):
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

        """
        self.thresh = thresh
        self.window = window
        f = sp.interpolate.interp1d(w, u)

        self.w = np.linspace(w.min(), w.max(), int(len(w) * 100))  # arbitrary upsampling
        self.u = f(self.w)

        self.u_smoothed = sp.signal.savgol_filter(self.u, 11, 4)

        self.baseline = piecewise_baseline(self.w, self.u_smoothed)

        self.peaks = Peaks()

    def find_maxima(self):
        """
        Local non-maxima supression to find peaks.

        """
        x_spacing = self.w[1] - self.w[0]
        window = int(self.window / x_spacing)  # arbitrary spacing (0.02)

        idx = sp.signal.argrelmax(self.u_smoothed, order=window)[0]

        for i in idx:
            p = Peak()
            p.loc = self.w[i]
            p.i = i
            p.height = self.u[i] - self.baseline[i]
            if p.height > self.thresh:
                self.peaks.append(p)

    def find_width(self):
        """
        Using peak information, finds FWHM and performs a conversion to get width.

        """
        screened_peaks = Peaks()
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


def piecewise_baseline(x, y, plot=True):
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
