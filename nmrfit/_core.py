import numpy as _np
import nmrglue as _ng

from . import _proc_autophase
from . import _containers
from . import _utility
from . import plot


def load(fidfile, procfile):
    """
    Loads NMR spectra data from relevant input files.

    Parameters
    ----------
    fidfile : string
        Path to the fid file.
    procfile : string
        Path to the procpar file.

    Returns
    -------
    result : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).

    """
    dic, data = _ng.varian.read_fid(fidfile)
    procs = _ng.varian.read_procpar(procfile)

    offset = [float(i) for i in procs['tof']['values']][0]
    magfreq = [float(i) for i in procs['sfrq']['values']][0]
    rangeHz = [float(i) for i in procs['sw']['values']][0]

    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq

    w = _np.linspace(rangeppm - offsetppm, -offsetppm, data.size)

    # Fourier transform
    data = _ng.proc_base.fft(data)
    data = data / _np.max(data)

    u = data[0, :].real
    v = data[0, :].imag

    result = _containers.Data(w[::-1], u[::-1], v[::-1])
    return result


def fit(data, lower, upper, expon=0.5, dynamic_weighting=True, fit_im=False, pool=None, summary=True, options={}):
    '''
    Perform a fit of NMR spectroscopy data.

    Parameters
    ----------
    data : instance of Data class
            Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    lower, upper : list of floats
        Min, max bounds for each parameter in the optimization.
    expon : float, optional
        Raise relative weighting to this power.
    dynamic_weighting : bool, optional
        Specify whether dynamic weighting is used.
    fit_im : bool, optional
        Specify whether the imaginary part of the spectrum will be fit. Computationally expensive.
    pool : multiprocessing.Pool, optional
        An instance of a multiprocessing pool used to evaluate objective function and constraints.
    summary : bool, optional
        Flag to display a summary of the fit.
    options : dict, optional
        Used to pass additional options to the minimizer.

    Returns
    -------
    f : FitUtility
        Object containing result of the fit.

    '''
    f = _utility.FitUtility(data, lower, upper, expon, dynamic_weighting, fit_im, summary, pool, options)
    f.fit()
    return f
