import numpy as np
import matplotlib.pyplot as plt


def exp_penalty(K, xl, xr, x):
    '''This function creates a penalty function that is zero inside [xl,xr] and
       then approaches a maximum value of K as we deviate from [xl,xr].  The
       function is C-infinity (infinitely differentiable at all points).'''
    if (x < xl):
        f = K * np.exp((xr - xl) / (x - xl))
    elif (x > xr):
        f = K * np.exp((xr - xl) / (xr - x))
    else:
        f = 0.
    return f


def cubic_penalty(K, xl, xr, x):
    '''This function creates a penalty function that is zero inside [xl,xr] and
       has constant value K for x<xl-h and x>xr+h, where h=xr-xl.
       For values in the transition zones [xl-h,xl] & [xr,xr+h],  a cubic interpolant is
       used so that the function is C1 (once continuously differentiable).'''
    h = xr - xl
    if (x < xl - h):
        f = K
    elif (x < xl):
        f = K * (3 * np.power((xl - x) / h, 2) - 2 * np.power((xl - x) / h, 3))
    elif (x < xr):
        f = 0.
    elif (x < xr + h):
        f = K * (3 * np.power((x - xr) / h, 2) - 2 * np.power((x - xr) / h, 3))
    else:
        f = K
    return f


if __name__ == "__main__":
    # Script showing effect of penalty functions on [3,4] confidence region.
    xl = 3.
    xr = 4.
    K = 10.
    dx = 0.001

    # Plot zoomed in
    xvals = np.arange(xl - 2, xr + 2, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(K, xl, xr, xvals[i])
        cubvals[i] = cubic_penalty(K, xl, xr, xvals[i])
    plot1 = plt.subplot(1, 3, 1)
    Kincr = K * 1.1
    plot1.set_ylim([0, Kincr])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r')
    plt.title('Penalty for [3,4] zone', fontsize=10)
    plt.ylabel('Penalty value')

    # Zoom out some
    xvals = np.arange(xl - 6, xr + 6, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(K, xl, xr, xvals[i])
        cubvals[i] = cubic_penalty(K, xl, xr, xvals[i])
    plot2 = plt.subplot(1, 3, 2)
    plot2.set_ylim([0, Kincr])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r')
    plt.title('Zoomed out from [3,4] zone', fontsize=10)

    # Really zoom out
    xvals = np.arange(xl - 50, xr + 50, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(K, xl, xr, xvals[i])
        cubvals[i] = cubic_penalty(K, xl, xr, xvals[i])
    plot3 = plt.subplot(1, 3, 3)
    plot3.set_ylim([0, Kincr])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r')
    plt.title('Even more zoomed out', fontsize=10)
    plt.show()
