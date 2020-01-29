import numpy as _np
import nmrglue as _ng
import os

from . import _proc_autophase
from . import _containers
from . import _utility
from . import plot


def load(datafolder,vendor='varian'):
    """
    Loads NMR spectra data from relevant input files.

    Parameters
    ----------
	datafolder: string
        path for the data directory
	vendor: string
	    varian or bruker, based on spectrometer.

    Returns
    -------
    result : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).

    """
    if vendor=='varian':
        dic, data = _ng.varian.read_fid(os.path.join(datafolder,'fid'))
        procs = _ng.varian.read_procpar(os.path.join(datafolder,'procpar'))
        
        offset = [float(i) for i in procs['tof']['values']][0]
        magfreq = [float(i) for i in procs['sfrq']['values']][0]
        rangeHz = [float(i) for i in procs['sw']['values']][0]
    elif vendor=='bruker':
        # read in the bruker formatted data
        dic, data = _ng.bruker.read(datafolder)
        # remove the digital filter
        data = _ng.bruker.remove_digital_filter(dic, data)
        # reshape to be common with the varian data
        data = _np.reshape(data,(1,len(data))) 
        offset = float(dic['acqus']['O1'])
        magfreq = float(dic['acqus']['SFO1'])
        rangeHz = float(dic['acqus']['SW_h'])
    else:
        print('Format not defined or recognised')

    rangeppm = rangeHz / magfreq
    offsetppm = offset / magfreq

    # Fourier transform
    data = _ng.proc_base.fft(data)
    data = data / _np.max(data)

    u = data.real.sum(axis=0)
    v = data.imag.sum(axis=0)

    w = _np.linspace(rangeppm - offsetppm, -offsetppm, u.size)

    result = _containers.Data(w[::-1], u[::-1], v[::-1])
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
    f = _utility.FitUtility(data, lower, upper, expon, dynamic_weighting, fit_im, processes, summary, options)
    f.fit()
    return f
