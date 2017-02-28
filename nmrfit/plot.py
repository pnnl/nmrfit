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
    xs = fit.w
    if component.lower() == 'real':
        y_data = data.V
        ys = fit.real_contribs
    elif component.lower() == 'imag':
        y_data = data.I
        ys = fit.imag_contribs
    else:
        raise ValueError("Valid options for the component parameter are 'real' and 'imag'.")

    fig = plt.figure(1, figsize=(10, 8), dpi=150)
    ax = plt.subplot('111')
    plt.plot(x_data, y_data, linewidth=2, color='black', label='Data')
    for i, peak in enumerate(ys):
        if i == 0:
            plt.plot(xs, peak, linewidth=2, color='grey', alpha=0.7, label='Fit')
        else:
            plt.plot(xs, peak, linewidth=2, color='grey', alpha=0.7, label=None)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_yticklabels([])
    ax.tick_params(top='off', left='off', right='off')
    ax.set_xlabel('ppm', fontsize=16, fontweight='bold')
    ax.set_xlim((x_data.max(), x_data.min()))
    ax.legend(loc='upper right', fontsize=14)
    fig.tight_layout()
    plt.show()


def residual(data, fit, component='real'):
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

    """
    x_data = data.w
    xs = fit.w
    if component.lower() == 'real':
        y_data = data.V
        ys = fit.V
    elif component.lower() == 'imag':
        y_data = data.I
        ys = fit.I
    else:
        raise ValueError("Valid options for the component parameter are 'real' and 'imag'.")

    if len(x_data) != len(xs):
        raise IndexError("Dimension mismatch.  Regenerate result with scale=1.")

    resid = np.abs(ys - y_data)

    # set up figures
    fig = plt.figure(1, figsize=(10, 8), dpi=150)
    gs = gridspec.GridSpec(4, 1)

    axes = []
    axes.append(plt.subplot(gs[0:3, :]))
    axes.append(plt.subplot(gs[3, :]))

    for a, label in zip(axes, ['A', 'B']):
        a.spines['top'].set_color('none')
        a.spines['right'].set_color('none')
        if label == 'A':
            a.set_yticklabels([])
            a.spines['left'].set_color('none')
            a.tick_params(top='off', left='off', right='off')
            a.set_xticklabels([])
        else:
            a.tick_params(top='off', right='off')
            a.set_ylabel('Residual')

        a.text(0.02, 1.0, label, fontsize=16, fontweight='bold',
               transform=a.transAxes, va='top', ha='left')

    axes[1].plot(x_data, resid, color='grey', label='Residual')
    axes[1].set_xlim((x_data.max(), x_data.min()))

    axes[0].plot(x_data, y_data, color='black', label='Data', zorder=0)
    axes[0].set_xlim((x_data.max(), x_data.min()))

    axes[0].plot(xs, ys, color='grey', label='Fit', alpha=0.7, zorder=1)
    axes[0].legend(loc='upper right', fontsize=14)

    axes[1].set_xlabel('ppm', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()


def isotope_ratio(data, fit):
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
        fig = plt.figure(1, figsize=(10, 8), dpi=150)
        gs = gridspec.GridSpec(3, 3)

        axes = []
        axes.append(plt.subplot(gs[0:2, :]))
        axes.append(plt.subplot(gs[2, 0]))
        axes.append(plt.subplot(gs[2, 1]))
        axes.append(plt.subplot(gs[2, 2]))

        for a, label in zip(axes, ['A', 'B', 'C', 'D']):
            a.set_yticklabels([])
            a.spines['top'].set_color('none')
            a.spines['left'].set_color('none')
            a.spines['right'].set_color('none')
            a.tick_params(top='off', left='off', right='off')

            a.text(0.0, 1.0, label, fontsize=16, fontweight='bold',
                   transform=a.transAxes, va='top', ha='left', zorder=9)

        alpha = 0.7
        lw = 2
        # plot everything
        axes[0].plot(fit.w, fit.V, linewidth=lw, color='black', label='Fit', zorder=0)
        axes[0].plot(data.w, data.V, linewidth=lw, color='grey', alpha=alpha, label='Data', zorder=1)
        axes[0].set_xlim((data.w.max(), data.w.min()))
        axes[0].legend(loc='upper right', fontsize=14)

        # plot left sats
        axes[1].plot(fit.w, fit.V, linewidth=lw, color='black', zorder=0)
        axes[1].plot(data.w, data.V, linewidth=lw, color='grey', alpha=alpha, zorder=1)
        axes[1].set_ylim((0, ht * 1.5))
        axes[1].set_xlim(set2Range[::-1])

        # plot main peaks
        axes[2].plot(fit.w, fit.V, linewidth=lw, color='black', zorder=0)
        axes[2].plot(data.w, data.V, linewidth=lw, color='grey', alpha=alpha, zorder=1)
        axes[2].set_xlim(peakRange[::-1])
        axes[2].set_xlabel('ppm', fontsize=16, fontweight='bold')

        # plot right satellites
        axes[3].plot(fit.w, fit.V, linewidth=lw, color='black', zorder=0)
        axes[3].plot(data.w, data.V, linewidth=lw, color='grey', alpha=alpha, zorder=1)
        axes[3].set_ylim((0, ht * 1.5))
        axes[3].set_xlim(set1Range[::-1])

        # display
        fig.tight_layout()
        plt.show()
