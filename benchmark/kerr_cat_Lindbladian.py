import sys
sys.path.append('../')

import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.linalg import eig

from core_fxns import Liouvillian
from core_fxns import destroy_q, create_q
from core_fxns import make_Eigen_real
from core_fxns import Fock_to_Eigen_basis
from core_fxns import kerr_cat_hamiltonian

from kc_plot import plot_2d, plot_1d_cuts


HBAR = 1
labelsz = 12.
ticksz = 11
dpi = 300


def main(args):
    ''' Perform diagonalization and analysis of the Kerr-Cat Lindbladian '''

    # using Eigen ("E") or Fock ("F") basis?
    basis_type = args.basis_type
    # number of oscillator modes
    nbasis_fock = args.nbasis_fock
    nbasis_eigen = args.nbasis_eigen
    # potential parameters
    K = args.K
    delta = args.delta
    epsilon_1_list = np.arange(
        args.eps1_min,
        args.eps1_max + args.eps1_step,
        args.eps1_step) * K
    epsilon_2_list = np.arange(
        args.eps2_min,
        args.eps2_max + args.eps2_step,
        args.eps2_step) * K

    # dissipation parameters
    gamma = args.gamma * K
    nbar = args.nbar

    print('Simulation parameters')
    print('basis_type = ', basis_type)
    print('nbasis Fock = ', nbasis_fock)
    print('nbasis Eigen = ', nbasis_eigen)
    print('Potential parameters - K = ', K)
    print('Potential parameters - delta = ', delta)
    print('Potential parameters - eps1 = ', epsilon_1_list)
    print('Potential parameters - eps2 = ', epsilon_2_list)
    print('Dissipator parameters - gamma = ', gamma)
    print('Dissipator parameters - nbar = ', nbar)

    # start main loop of program
    Tx_list = []
    for epsilon_2 in epsilon_2_list:
        for epsilon_1 in epsilon_1_list:
            print('eps2,eps1 =', round(epsilon_2, 2), round(epsilon_1, 2))

            # Hamiltonian (in Fock basis)
            ham = kerr_cat_hamiltonian(
                nbasis_fock, K, epsilon_2, delta, epsilon_1, HBAR)
            vals, vecs = np.linalg.eigh(ham)
            vecs = make_Eigen_real(vecs)

            # Lindblad dissipator operators (in Fock basis)
            D1 = np.sqrt(gamma * (1 + nbar)) * destroy_q(nbasis_fock)
            D2 = np.sqrt(gamma * nbar) * create_q(nbasis_fock)

            # transform from Fock basis to Eigen basis (if optioned)
            if(basis_type == 'E'):
                d_matrix = vecs[:, :nbasis_eigen]
                ham = Fock_to_Eigen_basis(ham, d_matrix)
                D1 = Fock_to_Eigen_basis(D1, d_matrix)
                D2 = Fock_to_Eigen_basis(D2, d_matrix)

            # Liouvillian superoperator
            c_ops = [D1, D2]
            superop = Liouvillian(ham, c_ops, hbar=HBAR)

            # Finding matrices containing normalised right and left
            # eigenvectors
            eigenvals, eigenvecs_left, eigenvecs_right = eig(
                superop, left=True)

            # get Tx=[-Re(lambda)]^{-1}
            l_min = find_minima_lambda(eigenvals)
            Tx = 1. / l_min

            # save results
            Tx_list.append(Tx)

    ########################################
    # transform results to arrays
    ndim1 = len(epsilon_2_list)
    ndim2 = len(epsilon_1_list)
    Tx_array = np.reshape(Tx_list, [ndim1, ndim2])

    ########################################
    # save results to file
    if(basis_type == 'E'):
        levels = nbasis_eigen
    elif(basis_type == 'F'):
        levels = nbasis_fock
    filename = (f'kerr_cat_Lindbladian_K_{K}_delta_{delta}_'
                f'gamma_{gamma}_nbar_{nbar}_levels_{levels}')
    np.save(filename + '_eps1', epsilon_1_list)
    np.save(filename + '_eps2', epsilon_2_list)
    np.save(filename + '_Tx', Tx_array)

    ########################################
    # plot results

    # plot Tx vs (eps1,eps2)
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'
    y = epsilon_2_list / K
    ylabel = r'$\varepsilon_2 / K$'
    z = Tx_array
    zlabel = r'$T_{X}$'
    if(z.shape[0] != 1 and z.shape[1] != 1):
        fig = plot_2d(x, y, z, xlabel=xlabel, ylabel=ylabel,
                      zlabel=zlabel)  # ,levels=levels)

    # plot lambda vs eps1
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'

    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(
            round(epsilon_2_list[i] / K, 2))

        y = Tx_array[i, :]
        ylabel = r'$T_{X}$'
        fig = plot_1d_cuts(x, y, title=title, xlabel=xlabel, ylabel=ylabel)

    # show plots
    plt.show()

    # =====================================================================
    # End program
    print('DONE!!')

    return


def find_minima_lambda(eigenvals):
    '''
        Returns -Re(lambda), where -lambda is minima of eigenvals different
        from zero
    '''
    thr = 1.e-10
#    myeig = np.copy(np.abs(eigenvals.real))
    myeig = -np.copy(eigenvals.real)
    ndim = len(myeig)
    for i in range(ndim):
        temp = np.min(myeig)
        idx = np.where(myeig == temp)[0]
        if(abs(temp) > thr):
            return temp
        else:
            print('Excluding eigenvalue = {}'.format(temp))
            myeig = np.delete(myeig, idx)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # using Eigen ("E") or Fock ("F") basis?
    parser.add_argument(
        '-basis_type', type=str, choices=['E', 'F'], default='E',
        help='Type of basis (Fock or Eigen) to use (default: E)')
    # number of oscillator modes
    parser.add_argument(
        '-nbasis_fock', type=int, default=100,
        help='Number of Fock basis (default: 100)')
    parser.add_argument(
        '-nbasis_eigen', type=int, default=20,
        help='Number of Eigen basis (default: 20)')
    # potential parameters
    parser.add_argument(
        '-K', type=float, default=1.0,
        help='Kerr non-linearity parameter (default: 1.)')
    parser.add_argument('-delta', type=float, default=0.0,
                        help='Delta Kerr parameter (default: 0.)')
    parser.add_argument(
        '-eps1_min', type=float, default=0.0,
        help='Minimun Epsilon_1 Kerr parameter (default: 0.)')
    parser.add_argument(
        '-eps1_max', type=float, default=10.0,
        help='Maximun Epsilon_1 Kerr parameter (default: 10.)')
    parser.add_argument(
        '-eps1_step', type=float, default=0.1,
        help='Step Epsilon_1 Kerr parameter (default: 0.1)')
    parser.add_argument(
        '-eps2_min', type=float, default=0.0,
        help='Minimun Epsilon_2 Kerr parameter (default: 0.)')
    parser.add_argument(
        '-eps2_max', type=float, default=10.0,
        help='Maximun Epsilon_2 Kerr parameter (default: 10.)')
    parser.add_argument(
        '-eps2_step', type=float, default=0.1,
        help='Step Epsilon_2 Kerr parameter (default: 0.1)')
    # dissipation parameters
    parser.add_argument(
        '-gamma', type=float, default=0.1,
        help='Dissipation gamma parameter (default: 0.1)')
    parser.add_argument(
        '-nbar', type=float, default=0.5,
        help='Dissipation nbar parameter (default: 0.5)')

    args = parser.parse_args()
    main(args)
