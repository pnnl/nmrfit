import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate


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
            plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
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
    def __init__(self, w, u, v):
        self.u = u
        self.v = v
        self.w = w

        # empty list to store point information from clicks
        self.points = []

        # initialize plot
        self.fig = plt.figure()
        plt.plot(w, u)
        plt.axis([w[-1], w[0], min(u) - max(u) * 0.05, max(u) * 1.1])
        # plt.gca().invert_xaxis()

        # start event listener
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

        # display the plot
        plt.show()

    def __call__(self, event):
        '''
        Called whenever the user clicks on the plot.  Stores x and y location of the cursor.
        After 3 clicks, the plot is closed as the peak has been "defined."
        '''

        # add x,y location of click
        self.points.append([event.xdata, event.ydata])

        if len(self.points) == 3:
            self.parse_points()
            plt.close()

    def parse_points(self):
        '''
        Called after 3 clicks on the plot.  Sorts the stored points in terms of frequency (w) to define
        low, middle, and high.  Subsequently determines approximate peak height, width, and area.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        # sort points in frequency
        self.points.sort()

        # determine minimum and maximum
        wMin = self.points[0][0]
        wMax = self.points[2][0]

        # determine width from min and max
        self.width = wMax - wMin

        # initial prediction for peak center
        peakest = self.points[1][0]

        # determine peak height and location of peak by searching over an interval
        self.height, self.loc = find_peak(self.w, self.u, peakest, searchwidth=self.width / 2.)

        # determine indices within the peak width
        self.idx = np.where((self.w > wMin) & (self.w < wMax))

        # store min/max bounds
        self.bounds = [wMin, wMax]

        # calculate AUC over the width of the peak numerically
        self.area = sp.integrate.simps(self.u[self.idx], self.w[self.idx])


def find_peak(x, y, est, searchwidth=0.5):
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
    idx = np.where((x <= est + searchwidth) & (x >= est - searchwidth))

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
