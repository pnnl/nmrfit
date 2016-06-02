import glob
import os
import NMRfit_module as nmrft
import numpy as np
import InitialConditions


# set input directory
inDir = './Data/organophosphate'

# initialize dataset lists
folders = {}
folders['919V'] = []
folders['0445'] = []
folders['070'] = []

# result lists
results = {}
results['919V'] = []
results['0445'] = []
results['070'] = []

# store data based on container list
for subdir, dirs, files in os.walk(inDir):
    if 'dc-919V' in subdir:
        folders['919V'].append(subdir)
    elif 'dc-0445' in subdir:
        folders['0445'].append(subdir)
    elif 'dc-070' in subdir:
        folders['070'].append(subdir)

for dataset in folders.keys():
    if dataset == '070':
        for fn in folders[dataset]:
            exp = InitialConditions.Experiment(os.path.split(fn)[-1])
            print(os.path.split(fn)[-1])
            w, u, v, p0, p1 = nmrft.varian_process(os.path.join(fn, 'fid'), os.path.join(fn, 'procpar'))
            bs = nmrft.BoundsSelector(w, u, v, supress=True)
            w, u, v = bs.applyBounds(low=3.20, high=3.65)
            w, u, v = nmrft.increaseResolution(w, u, v)

            u_fit, v_fit, fitParams = nmrft.fit_peak(w, u, v, exp.initialConditions, method='Powell', options=None, weights=exp.weights, fitIm=False)
            percent = (fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20]) / (fitParams[5] + fitParams[8] + fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20])
            results[dataset].append(percent)

            nmrft.plot(w, u, u_fit)
            print(percent)
            print()

for dataset in results.keys():
    print(dataset)
    print('mean:', np.mean(results[dataset]))
    print('stdev:', np.std(results[dataset]))
    print()
