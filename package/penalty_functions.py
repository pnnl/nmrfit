import numpy as np
import matplotlib.pyplot as plt

def quad_penalty(xl, xr, x, K=1., relfrac=0.01):
    '''This function establishes a quadratic unbounded penalty that 
       is magnitude K a distance of relfrac*(xr-xl) outside of zone.'''
    K=0.
    relfrac=1.
    if (x < xl):
        f= K* np.square( (xl-x) / ( (xr-xl) * relfrac ) )
    elif (x > xr):
        f= K* np.square( (x-xr) / ( (xr-xl) * relfrac ) )
    else:
        f=0.
    return f

def exp_penalty(xl, xr, x, K=1., relfrac=0.01):
    '''This function creates a penalty function that is zero inside [xl,xr] and
       then approaches a maximum value of K as we deviate from [xl,xr].  The
       function is C-infinity (infinitely differentiable at all points).'''
    # relfrac is the fraction of the width xr-xl over which the penalty grows.
    K=0.
    if (x < xl):
        f = K * np.exp(relfrac*(xr - xl) / (x - xl))
    elif (x > xr):
        f = K * np.exp(relfrac*(xr - xl) / (xr - x))
    else:
        f = 0.
    return f


def cubic_penalty(xl, xr, x, K=1., relfrac=0.01):
    '''This function creates a penalty function that is zero inside [xl,xr] and
       has constant value K for x<xl-h and x>xr+h, where h=xr-xl.
       For values in the transition zones [xl-h,xl] & [xr,xr+h],  a cubic interpolant is
       used so that the function is C1 (once continuously differentiable).'''
    # relfrac is the fraction of the width xr-xl over which the penalty grows.
    K=0.
    h = relfrac*(xr - xl)
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
    dx = 0.0001

    # Plot zoomed in
    relfrac=0.01
    h = relfrac*(xr - xl)
    xvals = np.arange(xl - 20*h, xr + 20*h, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    quadvals=np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(xl, xr, xvals[i], K)
        cubvals[i] = cubic_penalty(xl, xr, xvals[i], K)
        quadvals[i] = quad_penalty(xl, xr, xvals[i], K)
      
    plot1 = plt.subplot(1, 3, 1)
    Kincr = K * 1.1
    plot1.set_ylim([0, Kincr])
    plot1.set_xlim([xr - 2*h, xr + 2*h])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r', xvals, quadvals, 'g')
    plt.title('Penalty at rhs of zone', fontsize=10)
    plt.ylabel('Penalty value')

    # Zoom out some
    xvals = np.arange(xl - 10*h, xr + 10*h, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    quadvals = np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(xl, xr, xvals[i], K)
        cubvals[i] = cubic_penalty(xl, xr, xvals[i], K)
        quadvals[i] = quad_penalty(xl, xr, xvals[i], K)
    plot2 = plt.subplot(1, 3, 2)
    plot2.set_ylim([0, Kincr])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r', xvals, quadvals, 'g')
    plt.title('Showing whole [3,4] zone', fontsize=10)

    # Really zoom out
    xvals = np.arange(xl - 500*h, xr + 500*h, dx)
    yvals = np.zeros(len(xvals))
    cubvals = np.zeros(len(xvals))
    quadvals = np.zeros(len(xvals))
    for i, _ in enumerate(xvals):
        yvals[i] = exp_penalty(xl, xr, xvals[i], K)
        cubvals[i] = cubic_penalty(xl, xr, xvals[i], K)
        quadvals[i] = quad_penalty(xl, xr, xvals[i], K)
    plot3 = plt.subplot(1, 3, 3)
    plot3.set_ylim([0, Kincr])
    plt.plot(xvals, yvals, 'b', xvals, cubvals, 'r', xvals, quadvals, 'g')
    plt.title('Even more zoomed out', fontsize=10)
    plt.show()
