import numpy as np
import nmrglue as ng

from . import proc_autophase
from . import containers
from . import utility


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
    dic, data = ng.varian.read_fid(fidfile)
    procs = ng.varian.read_procpar(procfile)

    offset = [float(i) for i in procs['tof']['values']][0]
    magfreq = [float(i) for i in procs['sfrq']['values']][0]
    rangeHz = [float(i) for i in procs['sw']['values']][0]

    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq

    w = np.linspace(rangeppm - offsetppm, -offsetppm, data.size)

    # Fourier transform
    data = ng.proc_base.fft(data)
    data = data / np.max(data)

    # phase correct
    p0, p1 = proc_autophase.approximate_phase(data, 'acme')

    u = data[0, :].real
    v = data[0, :].imag

    result = containers.Data(w[::-1], u[::-1], v[::-1], p0, p1)
    return result


def fit(data, lower, upper, expon=0.5, fit_im=False, summary=True, options={}):
    '''
    Perform a fit of NMR spectroscopy data.

    Parameters
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    lower, upper : list of floats
        Min, max bounds for each parameter in the optimization.
    expon : float
        Raise relative weighting to this power.
    fit_im : bool
        Specify whether the imaginary part of the spectrum will be fit. Computationally expensive.
    summary : bool
        Flag to display a summary of the fit.
    options : dict, optional
        Used to pass additional options to the minimizer.

    Returns
    -------
    f : FitUtility
        Object containing result of the fit.

    '''
    f = utility.FitUtility(data, lower, upper, expon, fit_im, summary, options)
    f.fit()
    return f
