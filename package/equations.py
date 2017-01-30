import numpy as np
import scipy as sp
import scipy.integrate


def kk_equation(x, r, yOff, width, loc, a, w):
    '''
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
    '''

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
    '''
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
    '''

    res, err = sp.integrate.quad(kk_equation, 0, np.inf, args=(r, yOff, width, loc, a, w))
    return res / np.pi


def voigt(w, r, yOff, width, loc, a):
    '''
    Calculates a Voigt function over the range w based on the relevant properties of the distribution.

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
    '''

    # Lorentzian component
    L = (2 / (np.pi * width)) * 1 / (1 + ((w - loc) / (0.5 * width))**2)

    # Gaussian component
    G = (2 / width) * np.sqrt(np.log(2) / np.pi) * np.exp(-((w - loc) / (width / (2 * np.sqrt(np.log(2)))))**2)

    # Voigt body
    V = yOff + a * (r * L + (1 - r) * G)

    return V


def objective(x, w, u, v, x0, weights, roibounds):
    '''
    The objective function used to fit supplied data.  Evaluates sum of squared differences
    between the fit and the data.

    Parameters
    ----------
    x : list(float)
        Parameter vector.
    w : ndarray
        Array of frequency data.
    u, v : ndarray
        Arrays of the real and imaginary components of the frequency response.
    weights : ndarray
        Array giving freq dependent weighting of error.  If array has zero length, weights will be
        computed by objective function on each call.
    roibounds : list of 2-tuples
        bounds used for dynamic weight computation.
    fitIm : bool
        Flag to determine whether the imaginary component of the data will be fit.
    x0 : list(float)
        Initial conditions.

    Returns
    -------
    residual : float
        The sum of squared differences between the data and fit.
    '''

    # global parameters
    theta, r, yOff = x[:3]

    # initial global parameters
    theta0, r0, yOff0 = x0[:3]

    # transform u and v to get V for the data
    V_data = u * np.cos(theta) - v * np.sin(theta)

    # initialize array for the fit of V
    V_fit = np.zeros_like(V_data)

    # initialize container for the V residual
    V_residual = 0

    # iteratively add the contribution of each peak to the fits for V
    for i in range(3, len(x), 3):
        # current approximations
        width = x[i]
        loc = x[i + 1]
        a = x[i + 2]

        V_fit = V_fit + voigt(w, r, yOff, width, loc, a)

    if len(weights) == 0:
        V_residual = np.square(np.multiply(wts(roibounds, V_data, w), (V_data - V_fit))).sum(axis=None)
    else:
        V_residual = np.square(np.multiply(weights, (V_data - V_fit))).sum(axis=None)

    # Potentially use higher exponents for TNC than for Powell
    # V_residual = np.square(np.multiply(wts(roibounds,V_data,w,0.75),(V_data - V_fit))).sum(axis=None)

    # return the total residual
    return V_residual


# the vectorized form can compute the integral for all w
kk_relation_vectorized = np.vectorize(kk_relation, otypes=[np.float])


def wts(roibounds, V_data, w, expon=0.5):
    """
    Given sequence ((LHB[0],RHB[0]),...,(LHB[n-1],RHB[n-1])) of bounds and V_data
    weights, we obtain maximums of |V_data| for each ROI (region of interest) and
    then choose weight 1 for largest-max-region and all non-ROI regions, whereas
    we choose weight (largest/max[I])^expon for all non-max ROI regions.
    """
    lIdx = np.zeros(len(roibounds), dtype=np.int)
    rIdx = np.zeros(len(roibounds), dtype=np.int)
    maxabs = np.zeros(len(roibounds))
    for i, bound in enumerate(roibounds):
        lIdx[i] = np.argmin(np.abs(w - bound[0]))
        rIdx[i] = np.argmin(np.abs(w - bound[1]))
        if lIdx[i] > rIdx[i]:
            temp = lIdx[i]
            lIdx[i] = rIdx[i]
            rIdx[i] = temp
        maxabs[i] = np.amax(np.abs(V_data[lIdx[i]:rIdx[i] + 1]))

    biggest = np.amax(maxabs)

    defaultweight = 0.1
    wts = np.ones(len(w)) * defaultweight

    for i, bound in enumerate(roibounds):
        wts[lIdx[i]:rIdx[i] + 1] = np.power(biggest / maxabs[i], expon)

    n = 10
    omega = 0.33333333
    laplace1d(wts, n, omega)

    return wts


def laplace1d(x, n=10, omega=0.33333333):
    """
    Given an array x, we perform 1d laplacian smoothing on the
    values in x for n iterations and with relaxation factor
    omega.  The end values of x are constrained to not change.
    """
    for i in range(0, n):
        x[1:-1] = (1. - omega) * x[1:-1] + omega * 0.5 * (x[2:] + x[:-2])
