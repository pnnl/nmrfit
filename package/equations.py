import numpy as np
import scipy as sp
import scipy.integrate


def kk_equation(x, r, yOff, sigma, mu, a, w):
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
    sigma : float
        The width of the Voigt function.
    mu : float
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
    L1 = (2 / (np.pi * sigma)) * 1 / (1 + ((x + w - mu) / (0.5 * sigma))**2)
    G1 = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((x + w - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)
    V1 = yOff + a * (r * L1 + (1 - r) * G1)

    # second half of integral
    L2 = (2 / (np.pi * sigma)) * 1 / (1 + ((-x + w - mu) / (0.5 * sigma))**2)
    G2 = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((-x + w - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)
    V2 = yOff + a * (r * L2 + (1 - r) * G2)

    # combining both halves for the total integral
    V = 1 / x * (V2 - V1)
    return V


def kk_relation(w, r, yOff, sigma, mu, a):
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
    sigma : float
        The width of the Voigt function.
    mu : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.

    Returns
    -------
    res : float
        Value of the integral evaluated at w.
    '''

    res, err = sp.integrate.quad(kk_equation, 0, np.inf, args=(r, yOff, sigma, mu, a, w))
    return res / np.pi


def voigt(w, r, yOff, sigma, mu, a):
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
    sigma : float
        The width of the Voigt function.
    mu : float
        Center of the Voigt function.
    a : float
        Area of the Voigt function.

    Returns
    -------
    V : ndarray
        Array defining the Voigt function over w.
    '''

    # Lorentzian component
    L = (2 / (np.pi * sigma)) * 1 / (1 + ((w - mu) / (0.5 * sigma))**2)

    # Gaussian component
    G = (2 / sigma) * np.sqrt(np.log(2) / np.pi) * np.exp(-((w - mu) / (sigma / (2 * np.sqrt(np.log(2)))))**2)

    # Voigt body
    V = yOff + a * (r * L + (1 - r) * G)

    return V


def objective(x, w, u, v, x0):
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
    weights : list(list(float), float)
        Range, weight pairs for intervals corresponding to each peak.
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
        sigma = x[i]
        mu = x[i + 1]
        a = x[i + 2]

        V_fit = V_fit + voigt(w, r, yOff, sigma, mu, a)

    roibounds = []
    for i in range(4, len(x0), 3):
        roibounds.append((x0[i] - 0.05, x0[i] + 0.05))
    V_residual = np.square(np.multiply(wts(roibounds, V_data, w, 0.5), (V_data - V_fit))).sum(axis=None)

    # Potentially use higher exponents for TNC than for Powell
    # V_residual = np.square(np.multiply(wts(roibounds,V_data,w,0.75),(V_data - V_fit))).sum(axis=None)

    # return the total residual
    return V_residual

# the vectorized form can compute the integral for all w
kk_relation_vectorized = np.vectorize(kk_relation, otypes=[np.float])


def wts(roibounds, V_data, w, expon):
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

    wts = np.ones(len(w))

    for i, bound in enumerate(roibounds):
        wts[lIdx[i]:rIdx[i] + 1] = np.power(biggest / maxabs[i], expon)

    return wts
