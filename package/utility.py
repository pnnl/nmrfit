import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate
import peakutils


class Peaks(list):
    def average_height(self):
        h = 0.
        for p in self:
            h += abs(p.height)
        return h / len(self)

    def split(self):
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


class BoundsSelector:
    '''
    Interactive utility used to bound the spectroscopy data.  The user clicks twice on a plot to
    indicate the lower and upper bounds in the frequency domain.

    Parameters
    ----------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    supress : bool, optional
        Flag to specify whether the interactive portion will be invoked.

    Returns
    -------
    None.
    '''
    def __init__(self, w, u, v, supress=False):
        self.u = u
        self.v = v
        self.w = w
        self.supress = supress

        if not self.supress:
            self.fig = plt.figure()  # figsize=(9, 5), dpi=300
            plt.plot(w, u)
            # plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
            # plt.gca().invert_xaxis()
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
            self.bounds = []
            plt.show()

    def __call__(self, event):
        self.bounds.append(event.xdata)
        if len(self.bounds) == 2:
            plt.close()

    def apply_bounds(self, low=None, high=None):
        '''
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
        '''
        if not self.supress:
            low = min(self.bounds)
            high = max(self.bounds)

        idx = np.where((self.w > low) & (self.w < high))

        self.u = self.u[idx]
        self.v = self.v[idx]
        self.w = self.w[idx]
        return self.w, self.u, self.v


class PeakSelector:
    '''
    Interactive utility used to identify peaks and calculate approximations to peak height, width, and area.

    Parameters
    ----------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.

    Returns
    -------
    None.
    '''
    def __init__(self, w, u):
        self.u = u
        self.w = w

        # empty list to store point information from clicks
        self.points = []

        # initialize plot
        self.fig = plt.figure()
        plt.plot(w, u)
        # plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
        # plt.gca().invert_xaxis()

        # start event listener
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

        # display the plot
        plt.show()

    def __call__(self, event):
        '''
        Called whenever the user clicks on the plot.  Stores x and y location of the cursor.
        After 2 clicks, the plot is closed as the peak has been "defined."
        '''

        # add x,y location of click
        self.points.append([event.xdata, event.ydata])

        if len(self.points) == 2:
            self.parse_points()
            plt.close()

    def parse_points(self):
        '''
        Called after 2 clicks on the plot.  Sorts the stored points in terms of frequency (w) to define
        low, middle, and high.  Subsequently determines approximate peak height, width, and area.

        Parameters
        ----------
        None.

        Returns
        -------
        Instance of Peak class
        '''
        peak = Peak()

        # sort points in frequency
        self.points.sort()

        # determine minimum and maximum
        wMin = self.points[0][0]
        wMax = self.points[1][0]

        # determine width from min and max
        peak.sigma = (wMax - wMin) / 3.

        # determine peak height and location of peak by searching over an interval
        peak.height, peak.loc = find_peak(self.w, self.u, wMin, wMax)

        # determine indices within the peak width
        peak.idx = np.where((self.w > wMin) & (self.w < wMax))

        # store min/max bounds
        peak.bounds = [wMin, wMax]

        # calculate AUC over the width of the peak numerically
        peak.area = sp.integrate.simps(self.u[peak.idx], self.w[peak.idx])

        self.peak = peak

    def get_peak(self):
        return self.peak


class AutoPeakSelector:
    def __init__(self, w, u):

        f = sp.interpolate.interp1d(w, u)

        self.w = np.linspace(w.min(), w.max(), int(len(w) * 100))  # arbitrary upsampling
        self.u = f(self.w)

        self.u_smoothed = sp.signal.savgol_filter(self.u, 11, 4)

        self.baseline = piecewise_baseline(self.w, self.u_smoothed)

        self.peaks = Peaks()

    def find_maxima(self):
        x_spacing = self.w[1] - self.w[0]
        window = int(0.02 / x_spacing)  # arbitrary spacing (0.02)

        idx = sp.signal.argrelmax(self.u_smoothed, order=window)[0]

        for i in idx:
            p = Peak()
            p.loc = self.w[i]
            p.height = self.u[i] - self.baseline[i]
            self.peaks.append(p)

    def find_sigma(self):
        screened_peaks = Peaks()
        for p in self.peaks:
            d = np.sign(p.height / 2. - (self.u[0:-1] - self.baseline[0:-1])) - np.sign(p.height / 2. - (self.u[1:] - self.baseline[1:]))
            rightIdx = np.where(d < 0)[0]  # right
            leftIdx = np.where(d > 0)[0]  # left

            x_right = self.w[rightIdx[np.argmin(np.abs(self.w[rightIdx] - p.loc))]]
            x_left = self.w[leftIdx[np.argmin(np.abs(self.w[leftIdx] - p.loc))]]

            if x_left < x_right:
                p.sigma = (x_right - x_left) / 2.3548
                p.bounds = [p.loc - (3 * p.sigma), p.loc + 3 * p.sigma]

                p.idx = np.where((self.w >= p.bounds[0]) & (self.w <= p.bounds[1]))

                p.area = sp.integrate.simps(self.u[p.idx], self.w[p.idx])

                screened_peaks.append(p)

        self.peaks = screened_peaks

    def find_peaks(self):
        self.find_maxima()
        self.find_sigma()

        return self.peaks

    def plot(self):
        plt.plot(self.w, self.u, color='b')
        for p in self.peaks:
            plt.scatter(p.loc, p.height, color='r')
            plt.axvline(p.bounds[0], color='g')
            plt.axvline(p.bounds[1], color='g')

        plt.show()


def find_peak(x, y, low, high):
    """
    Find peak within tolerance, as well as maximum value.

    Parameters
    ----------
    x, y : ndarray
        x and y components of the data being searched.
    est : float
        Estimated x location of the peak.
    searchwidth : float, optional
        Width on either side of est to search for a peak.
    Returns
    -------
    peakheight : float
        Height of the located peak.
    peakloc : float
        Location of the peak in terms of x.
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

    return peakheight, peakloc


def rnd_data(sigma, origdata):
    """
    Add normally distributed noise.

    Parameters
    ----------

    Returns
    -------
    """

    synthnoise = sigma * np.random.randn(origdata.size)
    synthdata = origdata + synthnoise
    return synthdata


def sample_noise(X, Y, xstart, xstop):
    """
    Calculate standard deviation of sample noise.

    Parameters
    ----------

    Returns
    -------
    """

    noiseY = Y[np.where((X <= xstop) & (X >= xstart))]
    noiseX = X[np.where((X <= xstop) & (X >= xstart))]

    baselinefit = np.poly1d(np.polyfit(noiseX, noiseY, 2))

    noise = noiseY - baselinefit(noiseX)

    return np.std(noise)


def piecewise_baseline(x, y):
    third = x.shape[0] / 3

    y1 = y[0:third]
    y2 = y[third:2 * third]
    y3 = y[2 * third:]

    base1 = peakutils.baseline(y1, 2)
    base2 = np.ones(y2.shape) * np.median(y2)
    base3 = peakutils.baseline(y3, 2)

    baseline = np.concatenate((base1, base2, base3))

    return baseline
