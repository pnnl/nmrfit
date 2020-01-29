import matplotlib.pyplot as plt
from matplotlib import gridspec


pfit = {'color': 'black',
        'lw': 2,
        'alpha': 1}
pdata = {'color': 'silver',
         'lw': 2,
         'alpha': 1}


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
    plt.close()

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
    plt.plot(x_data, y_data, linewidth=pdata['lw'], alpha=pdata['alpha'], color='black', zorder=0)
    for i, peak in enumerate(ys):
        plt.plot(xs, peak, linewidth=pfit['lw'], alpha=0.5, zorder=1)

    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_yticklabels([])
    ax.tick_params(top='off', left='off', right='off')

    ax.set_xlabel('ppm', fontsize=16, fontweight='bold')
    ax.set_xlim((x_data.max(), x_data.min()))

    fig.tight_layout()


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
    plt.close()

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

    resid = ys - y_data

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

    axes[1].plot(x_data, resid, linewidth=1, color='black', zorder=1)
    axes[1].axhline(linewidth=1, linestyle='--', color='grey', zorder=0)
    axes[1].set_xlim((x_data.max(), x_data.min()))
    # axes[1].set_ylim((min(resid.min(), -1), max(resid.max(), 1)))

    axes[0].plot(x_data, y_data, linewidth=pdata['lw'], alpha=pdata['alpha'], color=pdata['color'], label='Data', zorder=0)
    axes[0].set_xlim((x_data.max(), x_data.min()))

    axes[0].plot(xs, ys, linewidth=pfit['lw'], alpha=pfit['alpha'], color=pfit['color'], label='Fit', zorder=1)
    axes[0].legend(loc='upper right', fontsize=14)

    axes[1].set_xlabel('ppm', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Residual', fontweight='bold')
    fig.tight_layout()


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
        plt.close()

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
        mn = min(min(data.V), min(fit.V))

        # set up figures
        fig = plt.figure(1, figsize=(10, 8), dpi=150)
        gs = gridspec.GridSpec(4, 3)

        axes = []
        axes.append(plt.subplot(gs[0:2, :]))
        axes.append(plt.subplot(gs[2, 0]))
        axes.append(plt.subplot(gs[2, 1]))
        axes.append(plt.subplot(gs[2, 2]))
        axes.append(plt.subplot(gs[3, :]))

        for a, label in zip(axes, ['A', 'B', 'C', 'D', 'E']):
            if label == 'E':
                a.tick_params(top='off', right='off')
                a.set_ylabel('Residual', fontweight='bold')
            else:
                a.spines['left'].set_color('none')
                a.tick_params(top='off', left='off', right='off')
                a.set_yticklabels([])

            a.spines['top'].set_color('none')
            a.spines['right'].set_color('none')
            a.text(0.02, 1.0, label, fontsize=16, fontweight='bold',
                   transform=a.transAxes, va='top', ha='left', zorder=9)

        # plot everything
        axes[0].plot(fit.w, fit.V, linewidth=pfit['lw'], alpha=pfit['alpha'], color=pfit['color'], label='Fit', zorder=1)
        axes[0].plot(data.w, data.V, linewidth=pdata['lw'], alpha=pdata['alpha'], color=pdata['color'], label='Data', zorder=0)
        axes[0].set_xlim((data.w.max(), data.w.min()))
        axes[0].legend(loc='upper right', fontsize=14)

        # plot left sats
        axes[1].plot(fit.w, fit.V, linewidth=pfit['lw'], alpha=pfit['alpha'], color=pfit['color'], zorder=1)
        axes[1].plot(data.w, data.V, linewidth=pdata['lw'], alpha=pdata['alpha'], color=pdata['color'], zorder=0)
        axes[1].set_ylim(mn * 0.95, ht * 1.5)
        axes[1].set_xlim(set2Range[::-1])

        # plot main peaks
        axes[2].plot(fit.w, fit.V, linewidth=pfit['lw'], alpha=pfit['alpha'], color=pfit['color'], zorder=1)
        axes[2].plot(data.w, data.V, linewidth=pdata['lw'], alpha=pdata['alpha'], color=pdata['color'], zorder=0)
        axes[2].set_xlim(peakRange[::-1])

        # plot right satellites
        axes[3].plot(fit.w, fit.V, linewidth=pfit['lw'], alpha=pfit['alpha'], color=pfit['color'], zorder=1)
        axes[3].plot(data.w, data.V, linewidth=pdata['lw'], alpha=pdata['alpha'], color=pdata['color'], zorder=0)
        axes[3].set_ylim(mn * 0.95, ht * 1.5)
        axes[3].set_xlim(set1Range[::-1])

        # residual
        fit.generate_result(scale=1)
        axes[4].plot(fit.w, fit.V - data.V, linewidth=1, color='black', zorder=1)
        axes[4].axhline(linewidth=1, linestyle='--', color='grey', zorder=0)
        axes[4].set_xlim((fit.w.max(), fit.w.min()))
        axes[4].set_xlabel('ppm', fontsize=16, fontweight='bold')

        # display
        fig.tight_layout()
