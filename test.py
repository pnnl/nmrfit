import NMRfit_module as nmrft
import os
import numpy as np
import scipy as sp
import scipy.integrate
import scipy.optimize
import proc_autophase as pa


def approximate_theta(w, u, v):
    def objective(theta, w, u, v):
        V = u * np.cos(theta) - v * np.sin(theta)
        area = scipy.integrate.simps(V, x=w)
        return -area
    res = sp.optimize.minimize_scalar(objective, args=(w, u, v), method='Bounded', bounds=(-np.pi, np.pi))
    return res.x

# Load and process data
inDir = "./Data/organophosphate/dc-0445_cdcl3_kilimanjaro_22c_1d_1H_1_031816.fid"
w, u, v = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))


# initiate bounds selection
bs = nmrft.BoundsSelector(w, u, v)
w, u, v = bs.applyBounds()
w, u, v = nmrft.increaseResolution(w, u, v)

res = sp.optimize.minimize_scalar(objective, args=(w, u, v), method='Bounded', bounds=(-np.pi, np.pi))
theta = res.x

V = u * np.cos(theta) - v * np.sin(theta)

print('theta:', theta)

nmrft.plot(w, V)