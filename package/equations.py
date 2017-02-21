import numpy as np
import scipy as sp
import scipy.integrate
import multiprocessing as mp
from functools import partial


def kk_equation(x, r, yOff, width, loc, a, w):
    """
    The equation inside the integral in the Kramers-Kronig relation. Used to evaluate the V->I transform.
    This specific implementation has been arranged such that the singularity at x==w is accounted for.

    Parameters
    ----------
    x : float
        Variable the integral will be evaluated over.
    r : float
        Ratio between the Guassian and Lorentzian functions
    yOff : float
        Y-offset of the Voigt function.
    width : float
        The width of the Voigt function.
    loc : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.
    w : float
        Frequncy value for which the integral is calculated.

    Returns
    -------
    V : ndarray
        Array defining the Voigt function with respect to x.

    """
    # first half of integral (Lorentzian, Gaussian, and Voigt, respectively)
    L1 = (2 / (np.pi * width)) * 1 / (1 + ((x + w - loc) / (0.5 * width))**2)
    G1 = (2 / width) * np.sqrt(np.log(2) / np.pi) * np.exp(-((x + w - loc) / (width / (2 * np.sqrt(np.log(2)))))**2)
    V1 = yOff + a * (r * L1 + (1 - r) * G1)

    # second half of integral
    L2 = (2 / (np.pi * width)) * 1 / (1 + ((-x + w - loc) / (0.5 * width))**2)
    G2 = (2 / width) * np.sqrt(np.log(2) / np.pi) * np.exp(-((-x + w - loc) / (width / (2 * np.sqrt(np.log(2)))))**2)
    V2 = yOff + a * (r * L2 + (1 - r) * G2)

    # combining both halves for the total integral
    V = 1 / x * (V2 - V1)
    return V


def kk_relation(w, r, yOff, width, loc, a):
    """
    Performs the integral required of the Kramers-Kronig relation using the kk_equation function
    for a given w.  Note that this integral is only evaluated for a single w.  The vectorized form
    (kk_relation_vectorized) may be used to calulate the Kramers-Kronig relation for all w.

    Parameters
    ----------
    w : float
        Frequncy value for which the integral is calculated
    r : float
        Ratio between the Guassian and Lorentzian functions
    yOff : float
        Y-offset of the Voigt function.
    width : float
        The width of the Voigt function.
    loc : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.

    Returns
    -------
    res : float
        Value of the integral evaluated at w.

    """
    res, err = sp.integrate.quad(kk_equation, 0, np.inf, args=(r, yOff, width, loc, a, w))
    return res / np.pi


def kk_relation_parallel(w, r, yOff, width, loc, a):
    """
    Performs the integral required of the Kramers-Kronig relation using the kk_equation function
    for an array w in parallel.

    Parameters
    ----------
    w : ndarray
        Frequncy values for which the integral is calculated
    r : float
        Ratio between the Guassian and Lorentzian functions
    yOff : float
        Y-offset of the Voigt function.
    width : float
        The width of the Voigt function.
    loc : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.

    Returns
    -------
    res : ndarray
        Values of the integral evaluated at each w.

    """

    pool = mp.Pool(mp.cpu_count() - 1)
    res = np.array(pool.map(partial(kk_relation, r=r, yOff=yOff, width=width, loc=loc, a=a), w))
    pool.join()
    return res


def voigt(w, r, yOff, width, loc, a):
    """
    Calculates a Voigt function over w based on the relevant properties of the distribution.

    Parameters
    ----------
    w : ndarray
        Array over which the Voigt function will be evaluated.
    r : float
        Ratio between the Guassian and Lorentzian functions
    yOff : float
        Y-offset of the Voigt function.
    width : float
        The width of the Voigt function.
    loc : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.

    Returns
    -------
    V : ndarray
        Array defining the Voigt function over w.

    """
    # Lorentzian component
    L = (2 / (np.pi * width)) * 1 / (1 + ((w - loc) / (0.5 * width))**2)

    # Gaussian component
    G = (2 / width) * np.sqrt(np.log(2) / np.pi) * np.exp(-((w - loc) / (width / (2 * np.sqrt(np.log(2)))))**2)

    # Voigt body
    V = yOff + a * (r * L + (1 - r) * G)

    return V


def objective(x, w, u, v, weights, fitIm=False):
    """
    The objective function used to fit supplied data.  Evaluates sum of squared differences
    between the fit and the data.

    Parameters
    ----------
    x : list of floats
        Parameter vector.
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    weights : ndarray
        Array giving frequency-dependent weighting of error.
    fitIm : bool, optional
        Specify whether the imaginary part of the spectrum will be fit.

    Returns
    -------
    residual : float
        The sum of squared differences between the data and fit.

    """
    # weights = np.ones_like(weights)

    # global parameters
    theta, r, yOff = x[:3]

    # transform u and v to get V for the data
    V_data = u * np.cos(theta) - v * np.sin(theta)
    V_fit = np.zeros_like(V_data)

    # optionally, also for I
    if fitIm is True:
        I_data = u * np.sin(theta) + v * np.cos(theta)
        I_fit = np.zeros_like(I_data)

    # iteratively add the contribution of each peak to the fits for V
    for i in range(3, len(x), 3):
        # current approximations
        width = x[i]
        loc = x[i + 1]
        a = x[i + 2]

        # calculate fit for V
        V_fit = V_fit + voigt(w, r, yOff, width, loc, a)

        # optionally calculate for I (costly)
        if fitIm is True:
            I_fit = kk_relation_parallel(w, r, yOff, width, loc, a)

    # real component residual
    residual = np.square(np.multiply(weights, (V_data - V_fit))).sum(axis=None)

    # optionally add imaginary residual
    if fitIm is True:
        residual += np.square(np.multiply(weights, (I_data - I_fit))).sum(axis=None)

        # divide by two to ensure equal magnitude error
        residual /= 2.0

    # return the total residual
    return residual


def laplace1d(x, n=10, omega=0.33333333):
    """
    Given an array x, performs 1D laplacian smoothing on the values in x.

    .. note:: The end values of x are constrained to not change.

    Parameters
    ----------
    x : ndarray
        Array of 1D values.
    n : int, optional
        Number of smoothing iterations.
    omega : float, optional
        Relaxation factor.

    Returns
    -------
    x : ndarray
        Smoothed 1D array.

    """
    for i in range(0, n):
        x[1:-1] = (1. - omega) * x[1:-1] + omega * 0.5 * (x[2:] + x[:-2])
    return x


# the vectorized form can compute the integral for all w
kk_relation_vectorized = np.vectorize(kk_relation, otypes=[np.float])
