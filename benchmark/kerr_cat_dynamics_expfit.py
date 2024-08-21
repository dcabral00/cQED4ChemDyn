import argparse
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import inspect

from kc_plot import plot_2d, plot_1d_cuts, plot_1d


HBAR = 1
labelsz = 12.
ticksz = 11
dpi = 300
nlines = 2


def main(args):
    ''' Perform exponential fit to traces and plot results '''

    ########################################
    # simulation parameters
    nbasis = args.nbasis
    K = args.K
    delta = args.delta
    gamma = args.gamma
    nbar = args.nbar

    ########################################
    # Load data from files
    filename = (f'kerr_cat_dynamics_K_{K}_delta_{delta}_'
                f'gamma_{gamma}_nbar_{nbar}_levels_{nbasis}')
    times = np.load(filename + '_times.npy')
    epsilon_1_list = np.load(filename + '_eps1.npy')
    epsilon_2_list = np.load(filename + '_eps2.npy')
    obs1_array = np.load(filename + '_obs1.npy')
    obs2_array = np.load(filename + '_obs2.npy')

    print('epsilon_1_list = ', epsilon_1_list)
    print('epsilon_2_list = ', epsilon_2_list)

    ########################################
    # Check if data is defined

    try:
        obs1_array
    except NameError:
        obs1_array = None

    try:
        obs2_array
    except NameError:
        obs2_array = None

    # obs1
    if obs1_array is not None:
        x = times
        y = obs1_array
        f_model = f_model1
        fit_param1, fit_error1 = perform_fit(x, y, f_model)
        # fit_param1 = search_decay_time(x,y) #TODO
        print('fit1 done')

    # obs2
    if obs2_array is not None:
        x = times
        y = obs2_array
        f_model = f_model3
        fit_param2, fit_error2 = perform_fit(x, y, f_model)
        print('fit2 done')

    ########################################
    # plot results
    ########################################

    # --------------------------------------
    # plot countour T vs (eps1,eps2)
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'
    y = epsilon_2_list / K
    ylabel = r'$\varepsilon_2 / K$'

    # obs1
    if obs1_array is not None:
        title = r'$<\psi_L>$'
        z = fit_param1
        zlabel = r'$T*K$'
        z = np.log(fit_param1)
        zlabel = r'$ln(T*K)$'
        levels = np.linspace(0, np.max(z), 50)
        # levels=np.linspace(np.min(z),np.max(z),50)
        # levels=None
        print('min,max = ', np.min(z), np.max(z))
        if(z.shape[0] != 1 and z.shape[1] != 1):
            fig = plot_2d(x, y, z, title=title,
                          xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                          levels=levels)
            filename = (f'kerr_cat_dynamics_K_{K}_delta_{delta}_'
                        f'gamma_{gamma}_nbar_{nbar}_levels_{nbasis}_TL.pdf')
            plt.savefig(filename, format='pdf', dpi=dpi)

    # obs2
    if obs2_array is not None:
        title = r'$<\Theta_X>$'
        z = fit_param2
        zlabel = r'$T*K$'
        z = np.log(fit_param2)
        zlabel = r'$ln(T*K)$'
        # levels=np.linspace(0,np.max(fit_param2[np.where(epsilon_2_list==10.),:]),50)
        # levels=np.linspace(0,2.*np.max(fit_param1),50)
        levels = np.linspace(0, np.max(z), 50)
        # levels=np.linspace(np.min(z),np.max(z),50)
        # levels=None
        print('min,max = ', np.min(z), np.max(z))
        if(z.shape[0] != 1 and z.shape[1] != 1):
            fig = plot_2d(x, y, z, title=title,
                          xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                          levels=levels)
            filename = (f'kerr_cat_dynamics_K_{K}_delta_{delta}_'
                        f'gamma_{gamma}_nbar_{nbar}_levels_{nbasis}_TX.pdf')
            plt.savefig(filename, format='pdf', dpi=dpi)

    # exit()

    # --------------------------------------
    # plot fit parameter T vs eps1
    x = epsilon_1_list
    xlabel = r'$\varepsilon_1 / K$'

    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(round(epsilon_2_list[i], 2))

        # obs1
        if obs1_array is not None:
            y = fit_param1[i, :]
            yerr = fit_error1[i, :]
            ylabel = r'$T_{L}*K$'
            plot_error = False
            plot_error = True
            fig = plot_1d(x, y, yerr, title=title, xlabel=xlabel,
                          ylabel=ylabel, plot_error=plot_error)

        # obs2
        if obs2_array is not None:
            y = fit_param2[i, :]
            yerr = fit_error2[i, :]
            ylabel = r'$T_{\Theta}*K$'
            plot_error = False
            plot_error = True
            fig = plot_1d(x, y, yerr, title=title, xlabel=xlabel,
                          ylabel=ylabel, plot_error=plot_error)

    plt.show()

    # --------------------------------------
    # plot contour P(t) vs eps1
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'
    y = times * K
    ylabel = r'$t * K$'

    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(round(epsilon_2_list[i], 2))

        # obs1
        if obs1_array is not None:
            z = obs1_array[i, :, :].T
            zlabel = r'$<\psi_L(t)>$'
            levels = np.arange(0., 1.1, 0.1)
            if(z.shape[0] != 1 and z.shape[1] != 1):
                fig = plot_2d(x, y, z, title=title,
                              xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                              levels=levels)

        # obs2
        if obs2_array is not None:
            z = obs2_array[i, :, :].T
            zlabel = r'$<\Theta_X(t)>$'
            levels = np.arange(0., 1.1, 0.1)
            if(z.shape[0] != 1 and z.shape[1] != 1):
                fig = plot_2d(x, y, z, title=title,
                              xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                              levels=levels)

    # --------------------------------------
    # plot individual traces P(t) vs t
    x = times * K
    xlabel = r'$t * K$'
    nfreq = 10

    data_label = np.round(epsilon_1_list[:], 2)
    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(round(epsilon_2_list[i], 2))

        # obs1
        if obs1_array is not None:
            y = obs1_array[i, :, :]
            ylabel = r'$<\psi_L(t)>$'
            fig = plot_1d_cuts(x, y, data_label=data_label, title=title,
                               xlabel=xlabel, ylabel=ylabel, nfreq=nfreq)

        # obs2
        if obs2_array is not None:
            y = obs2_array[i, :, :]
            ylabel = r'$<\Theta_X(t)>$'
            fig = plot_1d_cuts(x, y, data_label=data_label, title=title,
                               xlabel=xlabel, ylabel=ylabel, nfreq=nfreq)

    # show plots
    plt.show()

    print('DONE!!')

    return


########################################
# perform exponential fit
def f_model1(x, T, C):
    ''' Decreasing exponential with y-shift '''
    return (1. - C) * np.exp(-x / T) + C


def f_model2(x, T, x0, C):
    ''' Sigmoidal with y-shift '''
    return C / (1. + np.exp(-(x - x0) / T))


def f_model3(x, T, C):
    ''' Increasing exponential with y-shift '''
    return C * (1. - np.exp(-x / T))


def perform_fit(x, y, f_model, popt=None):
    ndim1, ndim2, ndim3 = y.shape
    fit_param = np.zeros([ndim1, ndim2])
    fit_error = np.zeros([ndim1, ndim2])

    argspec = inspect.getfullargspec(f_model)
    if(len(argspec.args) == 3):
        popt0 = [x[-1] / 2, 0]
    elif(len(argspec.args) == 4):
        popt0 = [x[-1] / 2, 0, 1.]
    for i in range(ndim1):
        for j in range(ndim2):
            popt, pcov = curve_fit(
                f=f_model, xdata=x, ydata=y[i, j, :], p0=popt0)
            perr = np.sqrt(np.diag(pcov))
            fit_param[i, j] = popt[0]
            fit_error[i, j] = perr[0]
    return fit_param, fit_error


def search_decay_time(x, y, cutoff=0.63):
    ndim1, ndim2, ndim3 = y.shape
    fit_param = np.zeros([ndim1, ndim2])
    for i in range(ndim1):
        for j in range(ndim2):
            ydata = y[i, j, :]
            idx, = np.where(ydata < cutoff)
            fit_param[i, j] = x[idx[0]]
    return fit_param


def baseline_correction(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    ''' Correct data by sunstracting a baseline.
        Taken from the internet
    '''
    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        crit = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z


def test_fit(times, obs_arr, eps2_l, eps1_l, idx, f_model):
    x = times
    for i in range(0, len(eps2_l), 20):
        for j in range(0, len(eps1_l), 20):

            if(idx == 1):
                y = obs_arr[i, j, :]
                f_model = f_model1
            elif(idx == 2):
                y = obs_arr[i, j, :]
                f_model = f_model3
            elif(idx == 3):
                y = obs_arr[i, j, :]
                f_model = f_model2

            argspec = inspect.getfullargspec(f_model)
            if(len(argspec.args) == 3):
                popt0 = [times[-1] / 2, 0]
            elif(len(argspec.args) == 4):
                popt0 = [times[-1] / 2, 0, 1.]

            popt, pcov = curve_fit(f=f_model, xdata=x, ydata=y, p0=popt0)
            perr = np.sqrt(np.diag(pcov))

            print(popt, perr)

            fig, ax = plt.subplots()
            title = 'eps2={eps2_l[i]:.2f},eps1={eps1_l[j]:.2f}'
            ax.set_title(title)
            ax.set_xlabel('t')
            ax.set_ylabel('P(t)')
            ax.plot(x, y)
            ax.plot(x, f_model(x, *popt), '--')
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # number of oscillator modes
    parser.add_argument(
        '-nbasis', type=int, default=15,
        help='Number of Fock basis (default: 20)')
    # potential parameters
    parser.add_argument(
        '-K', type=float, default=1.0,
        help='Kerr non-linearity parameter (default: 1.)')
    parser.add_argument('-delta', type=float, default=0.0,
                        help='Delta Kerr parameter (default: 0.)')
    # dissipation parameters
    parser.add_argument(
        '-gamma', type=float, default=0.1,
        help='Dissipation gamma parameter (default: 0.1)')
    parser.add_argument(
        '-nbar', type=float, default=0.5,
        help='Dissipation nbar parameter (default: 0.5)')

    args = parser.parse_args()
    main(args)
