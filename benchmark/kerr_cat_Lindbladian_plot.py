import numpy as np
import matplotlib.pyplot as plt
import argparse

from kc_plot import plot_2d, plot_1d_cuts


HBAR = 1
labelsz = 12.
ticksz = 11
dpi = 300
nlines = 2


def main(args):
    ''' Plot results of Lindbladian analysis '''

    ########################################
    # simulation parameters
    nbasis = args.nbasis
    K = args.K
    delta = args.delta
    gamma = args.gamma
    nbar = args.nbar

    ########################################
    # Load data from files
    filename = (f'kerr_cat_Lindbladian_K_{K}_delta_{delta}_'
                f'gamma_{gamma}_nbar_{nbar}_levels_{nbasis}')
    epsilon_1_list = np.load(filename + '_eps1.npy')
    epsilon_2_list = np.load(filename + '_eps2.npy')
    Tx_array = np.load(filename + '_Tx.npy')

    print('Simulation parameters')
    print('nbasis = ', nbasis)
    print('Potential parameters - K = ', K)
    print('Potential parameters - delta = ', delta)
    print('Potential parameters - eps1 = ', epsilon_1_list)
    print('Potential parameters - eps2 = ', epsilon_2_list)
    print('Dissipator parameters - gamma = ', gamma)
    print('Dissipator parameters - nbar = ', nbar)

    ########################################
    # plot results
    ########################################

    # plot Tx vs (eps1, eps2)
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'
    y = epsilon_2_list / K
    ylabel = r'$\varepsilon_2 / K$'

    if args.plot_logTX:
        z = np.log(Tx_array)
        zlabel = r'$ln(T_{X}*K)$'
    else:
        z = Tx_array
        zlabel = r'$T_{X}*K$'
    # levels=np.linspace(0, np.max(z[np.where(epsilon_2_list==8.), :]), 50)
    # levels=np.linspace(np.min(z),
    #     np.max(z[np.where(epsilon_2_list==8.), :]), 50)
    # levels=np.linspace(0, np.max(z), 50)
    levels = np.linspace(np.min(z), np.max(z), 50)
    if(z.shape[0] != 1 and z.shape[1] != 1):
        fig = plot_2d(x, y, z, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      levels=levels)
        filename = (f'kerr_cat_Lindbladian_K_{K}_delta_{delta}_gamma_{gamma}_'
                    f'nbar_{nbar}_levels_{nbasis}_TX.pdf')
        plt.savefig(filename, format='pdf', dpi=dpi)
    # plt.show()
    # exit()

    # plot lambda vs eps1
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'

    save_plot = [6.0, 8.0]

    for i in range(0, len(epsilon_2_list), 10):
        eps2_label = round(epsilon_2_list[i] / K, 2)
        title = r'$\varepsilon_2 / K$ = {}'.format(eps2_label)

        y = Tx_array[i, :]
        ylabel = r'$T_{X}*K$'
        fig = plot_1d_cuts(x, y, title=title, xlabel=xlabel, ylabel=ylabel)
        filename = (f'kerr_cat_Lindbladian_K_{K}_delta_{delta}_gamma_{gamma}_'
                    f'nbar_{nbar}_levels_{nbasis}_TX_cut_eps2_{eps2_label}')
        if(eps2_label in save_plot):
            plt.savefig(filename + '.pdf', format='pdf', dpi=dpi)

    # plot lambda vs eps2
    x = epsilon_2_list / K
    xlabel = r'$\varepsilon_2 / K$'

    save_plot = [0.0]

    for i in range(0, len(epsilon_1_list), 10):
        eps1_label = round(epsilon_1_list[i] / K, 2)
        title = r'$\varepsilon_1 / K$ = {}'.format(eps1_label)

        y = Tx_array[:, i]
        ylabel = r'$T_{X}*K$'
        xlim = [0, 9]
        ylim = [0, y[np.where(epsilon_2_list == 9)]]
        fig = plot_1d_cuts(x, y, title=title, xlabel=xlabel, ylabel=ylabel,
                           xlim=xlim, ylim=ylim)
        filename = (f'kerr_cat_Lindbladian_K_{K}_delta_{delta}_gamma_{gamma}_'
                    f'nbar_{nbar}_levels_{nbasis}_TX_cut_eps1_{eps1_label}')
        if(eps1_label in save_plot):
            plt.savefig(filename + '.pdf', format='pdf', dpi=dpi)

    # show plots
    # plt.show()

    # =====================================================================
    # End program

    print('DONE!!')

    return


def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid value for boolean argument."
                                         "Use 'True' or 'False'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # number of oscillator modes
    parser.add_argument(
        '-nbasis',
        type=int,
        default=20,
        help='Number of basis (default: 20)')
    # potential parameters
    parser.add_argument(
        '-K',
        type=float,
        default=1.0,
        help='Kerr non-linearity parameter (default: 1.)')
    parser.add_argument('-delta', type=float, default=0.0,
                        help='Delta Kerr parameter (default: 0.)')
    # dissipation parameters
    parser.add_argument(
        '-gamma',
        type=float,
        default=0.1,
        help='Dissipation gamma parameter (default: 0.1)')
    parser.add_argument(
        '-nbar',
        type=float,
        default=0.5,
        help='Dissipation nbar parameter (default: 0.5)')
    # whether to use logscale for heatmap plot
    parser.add_argument(
        '-plot_logTX',
        type=str_to_bool,
        default=True,
        help='Whether to use logscale for heatmap timescale plot'
             '(default: 0.5)')

    args = parser.parse_args()
    main(args)
