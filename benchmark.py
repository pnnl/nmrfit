import NMRfit as nmrft
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# input directory
inDir = "./Data/blinded/"
experiments = glob.glob(os.path.join(inDir, '*.fid'))

c = ['id', 'sample', 'method', 'error', 'fraction',
     'theta', 'GLratio', 'yoff']
for i in range(1, 7):
    c.extend(['s' + str(i), 'mu' + str(i), 'a' + str(i)])

l = []
for exp in experiments:
    ID = os.path.basename(exp)
    sample = ID.split('_')[1]
    print(ID)

    # read in data
    data = nmrft.varian_process(os.path.join(exp, 'fid'), os.path.join(exp, 'procpar'))

    # bound the data
    data.select_bounds(low=3.23, high=3.6)

    # select peaks and satellites
    peaks = data.select_peaks(method='auto', n=6, plot=False)

    # generate bounds and initial conditions
    x0, bounds = data.generate_initial_conditions()

    # fit data
    fit = nmrft.FitUtility(data, x0, method='Powell')

    # generate result
    res = fit.generate_result(scale=1)

    row = [ID, sample, 'Powell', res.error, res.area_fraction]
    if len(row) + len(res.params) == len(c):
        row.extend(res.params)
    else:
        print('Powell fit error...')
        row.extend(['NA' for i in range(len(c) - len(row))])
    print(len(row))
    l.append(row)

    plt.close()
    plt.plot(fit.data.w, fit.data.V, linewidth=2, alpha=0.5)
    plt.plot(fit.result.w, fit.result.V, linewidth=2, alpha=0.5)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.savefig('./results/%s_powell.png' % ID)

    # Now we will pass global results onto TNC
    x0[:3] = res.params[:3]
    x0[1]=max(0.,min(1.,x0[1]))

    # fit data
    fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options={'maxCGit':1000,'maxiter':1000})

    # generate result
    res = fit.generate_result(scale=1)

    row = [ID, sample, 'TNC', res.error, res.area_fraction]
    if len(row) + len(res.params) == len(c):
        row.extend(res.params)
    else:
        print('TNC fit error...')
        row.extend(['NA' for i in range(len(c) - len(row))])

    print(len(row))
    l.append(row)

    plt.close()
    plt.plot(fit.data.w, fit.data.V, linewidth=2, alpha=0.5)
    plt.plot(fit.result.w, fit.result.V, linewidth=2, alpha=0.5)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.savefig('./results/%s_tnc.png' % ID)

df = pd.DataFrame(l, columns=c)
df.to_csv('./results/results.csv', index=False)
