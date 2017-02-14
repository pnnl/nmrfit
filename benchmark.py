import NMRfit as nmrft
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def run_fit(exp, method='auto'):
    ID = os.path.basename(exp)
    sample = ID.split('_')[1]
    trial = ID.split('_')[-2]
    end = ID.split('_')[-1].split('.')[0]
    print(ID)

    c = ['id', 'sample', 'method', 'error', 'fraction',
         'theta', 'GLratio', 'yoff']
    for i in range(1, 7):
        c.extend(['s' + str(i), 'loc' + str(i), 'a' + str(i)])

    # read in data
    data = nmrft.varian_process(os.path.join(exp, 'fid'), os.path.join(exp, 'procpar'))

    # bound the data
    if sample == '4b':
        if int(trial) in [3, 4, 5]:
            data.select_bounds(low=3.2, high=3.6)
        else:
            data.select_bounds(low=3.25, high=3.6)
    elif sample == '4d':
        if end == '072316':
            data.select_bounds(low=3.23, high=3.6)
        else:
            data.select_bounds(low=3.25, high=3.6)
    else:

        data.select_bounds(low=3.2, high=3.6)

    # select peaks and satellites
    data.select_peaks(method=method, n=6, plot=False, thresh=0.002)

    # generate bounds and initial conditions
    lb, ub = data.generate_initial_conditions()

    # fit data
    fit = nmrft.FitUtility(data, lb, ub)

    # generate result
    fit.calculate_area_fraction()

    fit.generate_result()

    row = [ID, sample, 'PSO', fit.result.error, fit.result.area_fraction]
    if len(row) + len(fit.result.params) == len(c):
        row.extend(fit.result.params)
    else:
        print('TNC fit error...')
        row.extend(['NA' for i in range(len(c) - len(row))])

    return row


if __name__ == '__main__':
    # input directory
    inDir = "./data/blindedData/"
    experiments = glob.glob(os.path.join(inDir, '*.fid'))

    l = []
    for exp in experiments:
        l.append(run_fit(exp, method='auto'))

    # construct column labels
    c = ['id', 'sample', 'method', 'error', 'fraction',
         'theta', 'GLratio', 'yoff']
    for i in range(1, 7):
        c.extend(['s' + str(i), 'loc' + str(i), 'a' + str(i)])

    # build dataframe
    df = pd.DataFrame(l, columns=c)
    df.to_csv('./results/results_pso.csv', index=False)
