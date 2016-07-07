import numpy as np
import utility


class Result:
    '''
    Used to store results of the fit.  Similar to the Data class, but without the methods.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    '''
    def __init__(self):
        self.params = None
        self.error = None

        self.w = None
        self.u = None
        self.v = None
        self.V = None
        self.I = None


class Data:
    '''
    Stores data relevant to the NMRfit module and provides methods to interface with a number
    of utility classes.

    Parameters
    ----------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    thetaEst : float
        Estimate of zero order phase in radians.

    Returns
    -------
    None.
    '''
    def __init__(self, w, u, v, thetaEst):
        self.w = w
        self.u = u
        self.v = v

        self.V = None
        self.I = None

        self.theta = thetaEst

    def shift_phase(self, theta):
        '''
        Phase shift u and v by theta to generate V and I.

        Parameters
        ----------
        theta : float
            Phase correction in radians.

        Returns
        -------
        None.
        '''

        # calculate V and I from u, v, and theta
        V = self.u * np.cos(theta) - self.v * np.sin(theta)
        I = self.u * np.sin(theta) + self.v * np.cos(theta)

        # store as attributes
        self.V = V
        self.I = I

    def select_bounds(self, low=None, high=None):
        '''
        Method to interface with the utility class BoundsSelector.  If low and high are supplied, the interactive
        aspect will be overridden.

        Parameters
        ----------
        low : float, optional
            Lower bound.
        high : float, optional
            Upper bound.

        Returns
        -------
        None.
        '''

        if low is not None and high is not None:
            bs = utility.BoundsSelector(self.w, self.u, self.v, supress=True)
            self.w, self.u, self.v = bs.apply_bounds(low=low, high=high)
        else:
            bs = utility.BoundsSelector(self.w, self.u, self.v)
            self.w, self.u, self.v = bs.apply_bounds()

        self.shift_phase(self.theta)

    def select_peaks(self, n):
        '''
        Method to interface with the utility class PeakSelector.  Will open an interactive utility used to select
        peaks n times.

        Parameters
        ----------
        n : int
            Number of peaks to select.

        Returns
        -------
        peaks : list(PeakSelector)
            List containing instances of PeakSelector objects, which contain information about each peak.
        '''

        peaks = []
        for i in range(n):
            ps = utility.PeakSelector(self.w, self.V, self.I)
            peaks.append(ps)

        return peaks

    def select_satellites(self, n):
        '''
        Method to interface with the utility class PeakSelector.  Will open an interactive utility used to select
        satellites n times.

        Parameters
        ----------
        n : int
            Number of satellites to select.

        Returns
        -------
        satellites : list(PeakSelector)
            List containing instances of PeakSelector objects, which contain information about each satellite.

        Note: functionally the same as the select_peaks method, but allows user to distinguish between peaks and
        satellites conceptually (and keep them seperate).
        '''
        return self.select_peaks(n)
