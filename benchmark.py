import NMRfit as nmrft
import os
import glob


# input directory
inDir = "./Data/blindedData/"
experiments = glob.glob(os.path.join(inDir, '*.fid'))

with open('results.log', 'w') as flog:
    for exp in experiments:
        print(os.path.basename(exp))
        flog.write(os.path.basename(exp))
        flog.write('\n--------------------------------------------------------------------------------------------\n')

        # read in data
        data = nmrft.varian_process(os.path.join(exp, 'fid'), os.path.join(exp, 'procpar'))

        # bound the data
        data.select_bounds(low=3.23, high=3.6)

        # select peaks and satellites
        peaks = data.select_peaks(method='auto', n=6, plot=False)

        # generate bounds and initial conditions
        x0, bounds = data.generate_initial_conditions()

        flog.write(('\nInitial conditions: %s') % x0)

        # fit data
        fit = nmrft.FitUtility(data, x0, method='Powell')

        # generate result
        res = fit.generate_result(scale=1)

        flog.write('\nPowell:\n')
        flog.write(('Params: %s') % res.params)
        flog.write(('\nError:  %s') % res.error)
        flog.write(('\nArea fraction: %s') % res.area_fraction)

        # Now we will pass global results onto TNC
        x0[:3] = res.params[:3]

        # fit data
        fit = nmrft.FitUtility(data, x0, method='TNC', bounds=bounds, options=None)

        # generate result
        res = fit.generate_result(scale=1)

        flog.write('\nTNC:\n')
        flog.write(('Params: %s') % res.params)
        flog.write(('\nError:  %s') % res.error)
        flog.write(('\nArea fraction: %s\n\n') % res.area_fraction)
