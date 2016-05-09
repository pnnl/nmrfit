import NMRfit_module6 as nmrft
import os
import numpy as np


# Load and process data
inDir = "./Data/toluene/toluene2_cdcl3_long1_070115.fid"
w, u, v = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

theta = 1.61146988e-02

# initiate bounds selection
bs = nmrft.BoundsSelector(w, u, v)
w, u, v = bs.applyBounds()
w, u, v = nmrft.increaseResolution(w, u, v)

V_data = u * np.cos(theta) - v * np.sin(theta)
# V_data = u
nmrft.plot(w, V_data)
quit()

p1 = nmrft.PeakSelector(w, V_data, None)

s1 = nmrft.PeakSelector(w, V_data, None)
s2 = nmrft.PeakSelector(w, V_data, None)

ar = (s1.area + s2.area) / (s1.area + s2.area + p1.area)
print("area ratio:", ar)
