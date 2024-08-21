import numpy as np
import matplotlib.pyplot as plt


labelsz = 12.
ticksz = 11
dpi = 300
nlines = 2


def plot_2d(x, y, z, title='', xlabel='', ylabel='', zlabel='',
            levels=None, cmap='jet'):
    ''' Generate 2D plot of data '''

    # define levels
    if(levels is None):
        lmax = np.max(z)
        lmin = np.min(z)
        ldel = (lmax - lmin) / 10
        levels = np.arange(lmin, lmax + ldel, ldel)
    # plot data
    fig, ax = plt.subplots()
    CS = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    CL = ax.contour(x, y, z, levels=CS.levels[::nlines], colors='k')
    # add colorbar
    CB = fig.colorbar(CS)
    CB.add_lines(CL)
    # set labels
    ax.set_title(title, size=labelsz)
    ax.set_xlabel(xlabel, size=labelsz)
    ax.set_ylabel(ylabel, size=labelsz)
    CB.set_label(zlabel, size=labelsz, rotation=90)

    return fig


def plot_1d_cuts(x, y, data_label=None, title='', xlabel='', ylabel='',
                 nfreq=5):
    ''' Generate 1D plot cuts of 2D data '''

    # plot data
    fig, ax = plt.subplots()
    ndim1, ndim2 = y.shape
    for i in range(0, ndim1, nfreq):
        if(data_label is None):
            ax.plot(x, y[i, :])
        else:
            ax.plot(x, y[i, :], label=data_label[i])
    # set labels
    ax.set_title(title, size=labelsz)
    ax.set_xlabel(xlabel, size=labelsz)
    ax.set_ylabel(ylabel, size=labelsz)
    if(data_label is not None):
        ax.legend()

    return fig


def plot_1d(x, y, yerr=None, title='', xlabel='', ylabel='', plot_error=False):
    ''' Generate 1D plot '''

    if(plot_error and yerr is None):
        raise ValueError('[plot_1d] Error. "plot_error" requested but '
                         '"yerr" not provided.')

    # plot data
    fig, ax = plt.subplots()
    if(plot_error):
        # ax.errorbar(x,y,yerr)
        ax.fill_between(x, y - yerr, y + yerr, alpha=.6)
    ax.plot(x, y)
    # set labels
    ax.set_title(title, size=labelsz)
    ax.set_xlabel(xlabel, size=labelsz)
    ax.set_ylabel(ylabel, size=labelsz)

    return fig
