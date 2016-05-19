import glob
import os
import NMRfit_module as nmrft
import numpy as np


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

# set initial conditions for each dataset
x0 = {}
x0['919V'] = [0, 0.4590594625714286, 6.163417964285714e-05,
              0.002418037142857143, 3.4175240657142862, 0.002947622020000001,
              0.0024598158557142856, 3.431544792857143, 0.0029532094971428573,
              0.0020131753114285716, 3.2727146642857146, 1.6479002557142858e-05,
              0.0018598413128571426, 3.3053552371428574, 1.659152872857143e-05,
              0.001184384566, 3.5403618157142853, 1.2311321042857144e-05,
              0.0013514716314285715, 3.573018551428571, 1.2527656285714286e-05]
x0['0445'] = [0, 0.2, 0,
              2.5E-3, 3.43587682, 3E-3,
              2.5E-3, 3.40314646, 3E-3,
              1.5E-3, 3.56802656, 1E-5,
              1.5E-3, 3.53536475, 1E-5,
              2E-3, 3.30033289, 2E-5,
              2E-3, 3.26766376, 2E-5]
x0['070'] = [0, 0.2, 0,
             2.5E-3, 3.43517567, 3E-3,
             2.5E-3, 3.40245255, 3E-3,
             1.5E-3, 3.56735054, 1E-5,
             1.5E-3, 3.53468939, 1E-5,
             1.5E-3, 3.29966377, 1.5E-5,
             1.5E-3, 3.26699272, 1.5E-5]

# set weights for each dataset
weights = {}
weights['919V'] = [['all', 1.0],
                   ['all', 1.0],
                   [np.array([832, 833, 834, 835, 836, 837, 838, 839]), 127.60972906738266],
                   [np.array([759, 760, 761, 762, 763, 764, 765, 766, 767]), 129.60454931408256],
                   [np.array([237, 238, 239, 240, 241, 242, 243, 244, 245]), 126.06391549344025],
                   [np.array([164, 165, 166, 167, 168, 169, 170, 171, 172, 173]), 136.58676347670723]]
weights['0445'] = [['all', 1.0],
                   ['all', 1.0],
                   [np.array([814, 815, 816, 817, 818, 819, 820, 821, 822]), 116.1151542286283],
                   [np.array([742, 743, 744, 745, 746, 747, 748, 749]), 112.42429373438348],
                   [np.array([219, 220, 221, 222, 223, 224, 225, 226]), 139.02399113789605],
                   [np.array([146, 147, 148, 149, 150, 151, 152, 153]), 123.90569660894793]]
weights['070'] = [['all', 1.0],
                  ['all', 1.0],
                  [np.array([814, 815, 816, 817, 818, 819, 820, 821]), 118.74934629417289],
                  [np.array([739, 740, 741, 742, 743, 744, 745, 746]), 120.89155955493625],
                  [np.array([217, 218, 219, 220, 221, 222, 223, 224]), 131.79848222144506],
                  [np.array([144, 145, 146, 147, 148, 149, 150, 151]), 133.13414498147577]]

for dataset in folders.keys():
    dataset = '919V'
    for exp in folders[dataset]:
        w, u, v, p0, p1 = nmrft.varian_process(os.path.join(exp, 'fid'), os.path.join(exp, 'procpar'))
        bs = nmrft.BoundsSelector(w, u, v, supress=True)
        w, u, v = bs.applyBounds(low=3.20, high=3.65)
        w, u, v = nmrft.increaseResolution(w, u, v)

        u_fit, v_fit, fitParams = nmrft.fit_peak(w, u, v, x0[dataset], method='Powell', options=None, weights=weights[dataset], fitIm=False)
        percent = (fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20]) / (fitParams[5] + fitParams[8] + fitParams[11] + fitParams[14] + fitParams[17] + fitParams[20])
        results[dataset].append(percent)

        nmrft.plot(w, u, u_fit)

for dataset in results.keys():
    print(dataset)
    print('mean:', np.mean(results[dataset]))
    print('stdev:', np.std(results[dataset]))
    print()
