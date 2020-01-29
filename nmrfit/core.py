import numpy as np
import nmrglue as ng
import os

from . import containers
from . import utils


def load(path, vendor='varian'):
    """
    Loads NMR spectra data from relevant input files.

    Parameters
    ----------
    path : string
        path to the data directory.
    vendor : string
        varian or bruker, based on spectrometer.

    Returns
    -------
    result : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).

    """
    if vendor == 'varian':
        dic, data = ng.varian.read_fid(os.path.join(path, 'fid'))
        procs = ng.varian.read_procpar(os.path.join(path, 'procpar'))

        offset = [float(i) for i in procs['tof']['values']][0]
        magfreq = [float(i) for i in procs['sfrq']['values']][0]
        rangeHz = [float(i) for i in procs['sw']['values']][0]

    elif vendor == 'bruker':
        # read in the bruker formatted data
        dic, data = ng.bruker.read(path)
        # remove the digital filter
        data = ng.bruker.remove_digital_filter(dic, data)
        # reshape to be common with the varian data
        data = np.reshape(data, (1, len(data)))
        offset = float(dic['acqus']['O1'])
        magfreq = float(dic['acqus']['SFO1'])
        rangeHz = float(dic['acqus']['SW_h'])

    else:
        raise ValueError('Format not defined or recognised')

    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq

    # Fourier transform
    data = ng.proc_base.fft(data)
    data = data / np.max(data)

    u = data.real.sum(axis=0)
    v = data.imag.sum(axis=0)

    w = np.linspace(rangeppm - offsetppm, -offsetppm, u.size)

    result = containers.Data(w[::-1], u[::-1], v[::-1])
    return result


def fit(data, lower, upper, expon=0.5, dynamic_weighting=True, fit_im=False, processes=1, summary=True, options={}):
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
    processes : int, optional
        Number of processes used to evaluate objective function and constraints.
    summary : bool, optional
        Flag to display a summary of the fit.
    options : dict, optional
        Used to pass additional options to the minimizer.

    Returns
    -------
    f : FitUtility
        Object containing result of the fit.

    '''
    f = utils.FitUtility(data, lower, upper, expon, dynamic_weighting, fit_im, processes, summary, options)
    f.fit()
    return f
