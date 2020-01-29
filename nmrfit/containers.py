import numpy as np
import matplotlib.pyplot as plt

from . import utils
from . import proc_autophase


class Data:
    """
    Stores data relevant to the NMRfit module and provides methods to interface with a number
    of utility classes.

    Attributes
    ----------
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    V, I : ndarray
        Arrays of the phase corrected real and imaginary components of the frequency response.
    p0, p1 : float
        Estimate of zeroth and first order phase in radians.
    peaks : list of peak instances
        List containing instances of Peak objects, which contain information about each peak.
    roibounds : list of 2 tuples
        Frequency bounds of each peak.
    area_fraction : float
        Area fraction of satellite peaks.

    """

    def __init__(self, w, u, v):
        """
        Constructor for the Data class.

        Parameters
        ----------
        w : ndarray
            Array of frequency data.
        u, v : ndarray
            Arrays of the real and imaginary components of the frequency response.

        """
        self.w = w
        self.u = u
        self.v = v

        self.V = self.u[:]
        self.I = self.v[:]

    def shift_phase(self, method='auto', p0=0.0, p1=0.0, step=np.pi / 360, plot=False):
        """
        Phase shift u and v by theta to generate V and I.

        Parameters
        ----------
        method : string, optional
            Valid selections include 'auto', 'brute', and 'manual'.
        p0, p1 : float, optional
            Zeroth and first order phase correction in radians.
        step : float, optional
            Step size for brute force phase correction.
        plot : bool, optional
            Specify whether a plot of the peak selection is shown.

        """
        # calculate V and I from u, v, and theta
        if method.lower() == 'manual':
            self.p0 = p0
            self.p1 = p1
        elif method.lower() == 'auto':
            self.p0, self.p1 = proc_autophase.approximate_phase(self.u + 1j * self.v, 'acme')
        elif method.lower() == 'brute':
            self.p0, self.p1 = self._brute_phase(step=step)
        else:
            raise ValueError("Method must be 'auto', 'brute', or 'manual'.")

        self.V, self.I = proc_autophase.ps2(self.u, self.v, self.p0, self.p1)

        if plot is True:
            fig = plt.figure(1, figsize=(10, 8), dpi=150)
            ax = plt.subplot('111')

            plt.plot(self.w, self.V, linewidth=2, color='silver')

            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_yticklabels([])
            ax.tick_params(top='off', left='off', right='off')

            ax.set_xlabel('ppm', fontsize=16, fontweight='bold')
            ax.set_xlim((self.w.max(), self.w.min()))
            fig.tight_layout()

            plt.show()

    def _brute_phase(self, step=np.pi / 360):
        p0_best = 0
        bestError = np.inf
        n = max(1, int(len(self.V) / 5000))
        for p0 in np.arange(-np.pi, np.pi, step):
            self.V, self.I = proc_autophase.ps2(self.u, self.v, p0, 0.0)

            error = np.sqrt((self.V[:n].mean() - self.V[-n:].mean())**2)
            if error < bestError and np.max(self.V) > abs(np.min(self.V)):
                bestError = error
                p0_best = p0

        return p0_best, 0.0

    def select_bounds(self, low=None, high=None):
        """
        Method to interface with the utility class BoundsSelector.  If low and high are supplied, the interactive
        aspect will be overridden.

        Parameters
        ----------
        low : float, optional
            Lower bound.
        high : float, optional
            Upper bound.

        """
        if low is not None and high is not None:
            bs = utils.BoundsSelector(self.w, self.u, self.v, supress=True)
            self.w, self.u, self.v = bs.apply_bounds(low=low, high=high)
        else:
            bs = utils.BoundsSelector(self.w, self.u, self.v)
            self.w, self.u, self.v = bs.apply_bounds()

    def select_peaks(self, method='auto', n=None, one_click=False, thresh=0.0, window=0.02, plot=False):
        """
        Method to interface with the utility class PeakSelector.  Will open an interactive utility used to select
        peaks n times.

        Parameters
        ----------
        method : str, optional
            Flag to determine whether peaks will be selected automatically ('auto') or manually ('manual')
        n : int
            Number of peaks to select.  Only required if 'manual' is selected.
        one_click : bool, optional
            Enables single-click peak selection. Only used if 'manual' is selected.
        thresh : float, optional
            Threshold for peak detection. Only used if 'auto' is selected.
        window : float, optional
            Window for local non-maximum supression. Only used if 'auto' is selected.
        plot : bool, optional
            Specify whether a plot of the peak selection is shown.

        """
        if method.lower() == 'manual':
            if isinstance(n, int) and n > 0:
                ps = utils.PeakSelector(self.w, self.V, n, one_click=one_click)
            else:
                raise ValueError("Number of peaks must be specified when using 'manual' flag")

        elif method.lower() == 'auto':
            ps = utils.AutoPeakSelector(self.w, self.V, thresh=thresh, window=window)
            ps.find_peaks()

        else:
            raise ValueError("Method must be 'auto' or 'manual'.")

        if plot is True:
            ps.plot()

        self.peaks = ps.peaks

        self.roibounds = []
        for p in self.peaks:
            self.roibounds.append(p.bounds)

    def generate_solution_bounds(self, force_p0=False, force_p1=False):
        """
        Uses initial theta approximation as well as initial per-peak parameters (width, location, area)
        to construct a set of parameter bounds.

        Parameters
        ----------
        force_p0 : bool, optional
            Flag to use initial phase approximation for zeroth-order phase.
        force_p1 : bool, optional
            Flag to use initial phase approximation for first-order phase.

        Returns
        -------
        lower, upper : list of 2-tuples
            Min, max bounds for each parameter in optimization.

        """
        lower = []
        upper = []
        if force_p0 is True:
            upper.append(self.p0 + 0.001)
            lower.append(self.p0 - 0.001)
        else:
            upper.append(np.pi)
            lower.append(-np.pi)

        if force_p1 is True:
            upper.append(self.p1 + 0.001)
            lower.append(self.p1 - 0.001)
        else:
            upper.append(np.pi)
            lower.append(-np.pi)

        upper.extend([1.0, 0.01])
        lower.extend([0.0, -0.01])

        for p in self.peaks:
            lower.extend([p.width * 0.5, p.loc - 0.1 * (p.loc - p.bounds[0]), p.area * 0.5])
            # lower.extend([p.width * 0.01, p.loc - 0.1 * (p.loc - p.bounds[0]), p.area * 0.01])
            upper.extend([p.width * 1.5, p.loc - 0.1 * (p.loc - p.bounds[1]), p.area * 1.5])

        return lower, upper

    def approximate_areas(self):
        """
        Extracts the area attribute from each Peak instance and returns as a list.

        Returns
        -------
        areas : list of floats
            A list containing the peak area of each Peak instance.

        """
        areas = []
        for p in self.peaks:
            areas.append(p.area)
        return areas

    def approximate_area_fraction(self):
        """
        Calculates the relative fraction of the satellite peaks to the total peak area from the data.

        Returns
        -------
        area_fraction : float
            Area fraction of satellite peaks.

        """
        areas = np.array(self.approximate_areas())

        m = np.mean(areas)
        peaks = areas[areas >= m].sum()
        sats = areas[areas < m].sum()

        area_fraction = sats / (peaks + sats)

        return area_fraction
