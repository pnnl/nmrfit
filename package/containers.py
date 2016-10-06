import numpy as np
from .utility import AutoPeakSelector
from .utility import PeakSelector
from .utility import BoundsSelector
from .utility import Peaks


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
            bs = BoundsSelector(self.w, self.u, self.v, supress=True)
            self.w, self.u, self.v = bs.apply_bounds(low=low, high=high)
        else:
            bs = BoundsSelector(self.w, self.u, self.v)
            self.w, self.u, self.v = bs.apply_bounds()

        self.shift_phase(self.theta)

    def select_peaks(self, method='auto', n=None, plot=False):
        '''
        Method to interface with the utility class PeakSelector.  Will open an interactive utility used to select
        peaks n times.

        Parameters
        ----------
        method : str, optional
            Flag to determine whether peaks will be selected automatically ('auto') or manually ('manual')
        n : int, optional
            Number of peaks to select.  Only required if 'manual' is selected.

        Returns
        -------
        peaks : list(Peak)
            List containing instances of Peak objects, which contain information about each peak.
        '''
        if method.lower() == 'manual':
            if isinstance(n, int) and n > 0:
                ps = PeakSelector(self.w, self.V, n)
            else:
                raise ValueError("Number of peaks must be specified when using 'manual' flag")

        elif method.lower() == 'auto':
            ps = AutoPeakSelector(self.w, self.V)
            ps.find_peaks()

        else:
            raise ValueError("Method must be 'auto' or 'manual'.")

        if plot is True:
            ps.plot()

        return ps.peaks

    def generate_initial_conditions(self, tol=0.05):
        x0 = [self.theta, 1., 0.]
        bounds = [(None, None), (0., 1.), (None, None)]

        for p in self.peaks:
            x0.extend([p.sigma, p.loc, p.area])
            bounds.extend([(None, None), (p.loc - tol, p.loc + tol), (None, None)])

        return x0, bounds
