import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def individual_contributions(data, fit, component='real'):
    """
    Generates a plot of the individual fit peaks alongside the input data.

    Parameters
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    fit : instance of FitUtility class
        Container for ndarrays (w, u, v, V, I) of the fit result.
    component : string, optional
        Flag to specify the real or imaginary component will be plotted.

    """
    x_data = data.w
    x_res = fit.w
    if component.lower == 'real':
        y_data = data.V
        y_res = fit.real_contribs
    elif component.lower() == 'imag':
        y_data = data.I
        y_res = fit.imag_contribs
    else:
        raise ValueError("Valid options for the component parameter are 'real' and 'imag'.")

    plt.plot(x_data, y_data, color='black')
    for peak in y_res:
        plt.plot(x_res, peak)

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()


def residual(data, fit, component='real', plot_data=True, plot_fit=True):
    """
    Generates a residual plot between calculated fit and the input data.

    Parameters
    ----------
    data : instance of Data class
        Container for ndarrays relevant to the fitting process (w, u, v, V, I).
    fit : instance of FitUtility class
        Container for ndarrays (w, u, v, V, I) of the fit result.
    component : string, optional
        Flag to specify the real or imaginary component will be plotted.
    plot_data, plot_fit : bool, optional
        Flags to specify whether the data and/or fit will be plotted alongside the residual.

    """
    x_data = data.w
    x_res = fit.w
    if component.lower() == 'real':
        y_data = data.V
        y_res = fit.V
    elif component.lower() == 'imag':
        y_data = data.I
        y_res = fit.I
    else:
        raise ValueError("Valid options for the component parameter are 'real' and 'imag'.")

    if len(x_data) != len(x_res):
        raise IndexError("Dimension mismatch.  Regenerate result with scale=1.")

    resid = np.abs(y_res - y_data)

    fig, ax = plt.subplots()

    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx()]

    axes[0].plot(x_data, resid, color='black')

    if plot_data is True:
        axes[1].plot(x_data, y_data, color='black', alpha=0.5)

    if plot_fit is True:
        axes[1].plot(x_res, y_res, color='black', alpha=0.5)

    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Residual')
    axes[0].set_ylim((0, resid.max() * 5))
    axes[1].set_ylim((-0.5 * (y_res.max() - y_res.min()), y_res.max() * 1.05))
    axes[1].set_ylabel('Amplitude')
    plt.show()


def isotope_ratio(data, fit, area_fraction=None):
        """
        Generates a summary plot of the calculated fit alongside the input data.

        Parameters
        ----------
        data : instance of Data class
            Container for ndarrays relevant to the fitting process (w, u, v, V, I).
        fit : instance of FitUtility class
            Container for ndarrays (w, u, v, V, I) of the fit result.

        """
        peaks, satellites = data.peaks.split()

        peakBounds = []
        for p in peaks:
            low, high = p.bounds
            peakBounds.append(low)
            peakBounds.append(high)

        peakRange = [min(peakBounds) - 0.005, max(peakBounds) + 0.005]

        set1Bounds = []
        set2Bounds = []
        satHeights = []
        for s in satellites:
            low, high = s.bounds
            satHeights.append(s.height)
            if high < peakRange[0]:
                set1Bounds.append(low)
                set1Bounds.append(high)
            else:
                set2Bounds.append(low)
                set2Bounds.append(high)

        set1Range = [min(set1Bounds) - 0.02, max(set1Bounds) + 0.02]
        set2Range = [min(set2Bounds) - 0.02, max(set2Bounds) + 0.02]
        ht = max(satHeights)

        # set up figures
        fig_re = plt.figure(1, figsize=(10, 8))
        gs_re = gridspec.GridSpec(3, 3)
        ax1_re = fig_re.add_subplot(gs_re[0:2, :])
        ax2_re = fig_re.add_subplot(gs_re[2, 0])
        ax3_re = fig_re.add_subplot(gs_re[2, 1])
        ax4_re = fig_re.add_subplot(gs_re[2, 2])

        # plot everything
        ax1_re.plot(fit.w, fit.V, linewidth=3, color='black', label='Fit', zorder=0)
        ax1_re.scatter(data.w, data.V, linewidth=0, s=40, color='grey', label='Data', zorder=1)
        ax1_re.legend(loc='upper right')

        # plot left sats
        ax2_re.plot(fit.w, fit.V, linewidth=3, color='black', zorder=0)
        ax2_re.scatter(data.w, data.V, linewidth=0, s=40, color='grey', zorder=1)
        ax2_re.set_ylim((0, ht * 1.5))
        ax2_re.set_xlim(set1Range)

        # plot main peaks
        ax3_re.plot(fit.w, fit.V, linewidth=3, color='black', zorder=0)
        ax3_re.scatter(data.w, data.V, linewidth=0, s=40, color='grey', zorder=1)
        ax3_re.set_xlim(peakRange)

        # plot right satellites
        ax4_re.plot(fit.w, fit.V, linewidth=3, color='black', zorder=0)
        ax4_re.scatter(data.w, data.V, linewidth=0, s=40, color='grey', zorder=1)
        ax4_re.set_ylim((0, ht * 1.5))
        ax4_re.set_xlim(set2Range)

        # # imag
        # fig_im = plt.figure(2)
        # ax1_im = plt.subplot(211)
        # ax2_im = plt.subplot(234)
        # ax3_im = plt.subplot(235)
        # ax4_im = plt.subplot(236)

        # # plot everything
        # ax1_im.plot(data.w, data.I, linewidth=2, alpha=0.5, color='black', label='data')
        # ax1_im.plot(fit.w, fit.I, linewidth=4, color='black', label='Fit')

        # # plot left sats
        # ax2_im.plot(data.w, data.I, linewidth=2, alpha=0.5, color='black')
        # ax2_im.plot(fit.w, fit.I, linewidth=4, color='black')
        # ax2_im.set_xlim(set1Range)

        # # plot main peaks
        # ax3_im.plot(data.w, data.I, linewidth=2, alpha=0.5, color='black')
        # ax3_im.plot(fit.w, fit.I, linewidth=4, color='black')
        # ax3_im.set_xlim(peakRange)

        # # plot right satellites
        # ax4_im.plot(data.w, data.I, linewidth=2, alpha=0.5, color='black')
        # ax4_im.plot(fit.w, fit.I, linewidth=4, color='black')
        # ax4_im.set_xlim(set2Range)

        # display
        fig_re.tight_layout()
        plt.show()