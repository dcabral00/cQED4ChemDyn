import sys
sys.path.append('../')

import numpy as np
import argparse
import matplotlib.pyplot as plt

from core_fxns import filter_state_by_xvalues
from core_fxns import get_obs_traces
from core_fxns import propagation_by_semigroup
from core_fxns import Liouvillian
from core_fxns import destroy_q, create_q
from core_fxns import make_Eigen_real
from core_fxns import Fock_to_Eigen_basis
from core_fxns import get_heaviside_observable
from core_fxns import kerr_cat_hamiltonian

from kc_plot import plot_2d, plot_1d_cuts


HBAR = 1
labelsz = 12.
ticksz = 11
dpi = 300


def main(args):
    ''' Perform dynamics of Kerr-Cat Hamiltonian and compute some observables.
        Dynamics can be performed using  Eigen basis or Fock basis.
    '''

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

    # propagation timerange
    times = np.linspace(args.t_min, args.t_max, args.t_num) / K

    # initial state parameters
    grid_pts = args.grid_pts
    grid_lims = args.grid_lims
    x_cutoff = args.x_cutoff
    cutoff_value = args.cutoff_value
    sigmoidal_tail = args.sigmoidal_tail
    filter_type = args.filter_type

    # print to screen
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
    print('Timestep parameters - dt = ', times[1])
    print('Timestep parameters - total T = ', times[-1])
    print('Initial state parameters - filter_type = ', filter_type)

    # Heaviside observable (in Fock basis)
    # Note: Independent of the potential (need to be done only once)
    obs_heaviside = get_heaviside_observable(nbasis_fock, grid_pts, grid_lims)

    # start main loop of program
    obs1_list = []
    obs2_list = []
    obs3_list = []
    for epsilon_2 in epsilon_2_list:
        for epsilon_1 in epsilon_1_list:
            print('eps2,eps1 =', round(epsilon_2, 2), round(epsilon_1, 2))

            # Hamiltonian (in Fock basis)
            ham = kerr_cat_hamiltonian(
                nbasis_fock, K, epsilon_2, delta, epsilon_1, HBAR)
            vals, vecs = np.linalg.eigh(ham)
            vecs = make_Eigen_real(vecs)

            # initial state (in Fock basis)
            psi_l = filter_state_by_xvalues(
                vecs,
                grid_pts,
                grid_lims,
                x_cutoff,
                cutoff_value,
                left=True,
                filter_type=filter_type,
                sigmoidal_tail=sigmoidal_tail)
            psi_r = filter_state_by_xvalues(
                vecs,
                grid_pts,
                grid_lims,
                x_cutoff,
                cutoff_value,
                left=False,
                filter_type=filter_type,
                sigmoidal_tail=sigmoidal_tail)

            rho0 = np.outer(psi_l, psi_l.conjugate())
            rho0 /= np.trace(rho0)

            # observables (in Fock basis)
            obs1 = np.copy(rho0)
            obs2 = np.copy(obs_heaviside)
            obs3 = np.outer(psi_r, psi_r.conjugate())

            # Lindblad dissipator operators (in Fock basis)
            D1 = np.sqrt(gamma * (1 + nbar)) * destroy_q(nbasis_fock)
            D2 = np.sqrt(gamma * nbar) * create_q(nbasis_fock)

            # transform from Fock basis to Eigen basis (if optioned)
            if(basis_type == 'E'):
                d_matrix = vecs[:, :nbasis_eigen]
                ham = Fock_to_Eigen_basis(ham, d_matrix)
                rho0 = Fock_to_Eigen_basis(rho0, d_matrix)
                obs1 = Fock_to_Eigen_basis(obs1, d_matrix)
                obs2 = Fock_to_Eigen_basis(obs2, d_matrix)
                obs3 = Fock_to_Eigen_basis(obs3, d_matrix)
                D1 = Fock_to_Eigen_basis(D1, d_matrix)
                D2 = Fock_to_Eigen_basis(D2, d_matrix)

            # Liouvillian superoperator
            c_ops = [D1, D2]
            superop = Liouvillian(ham, c_ops, hbar=HBAR)

            # run dynamics
            temp, rhos = propagation_by_semigroup(
                rho0, superop, times[1], len(times))

            # compute observable traces
            obs1_traces = get_obs_traces(rhos, obs1)
            obs2_traces = get_obs_traces(rhos, obs2)
            obs3_traces = get_obs_traces(rhos, obs3)

            # save results
            obs1_list.append(obs1_traces)
            obs2_list.append(obs2_traces)
            obs3_list.append(obs3_traces)

    ########################################
    # transform results to arrays
    ndim1 = len(epsilon_2_list)
    ndim2 = len(epsilon_1_list)
    ndim3 = len(times)
    obs1_array = np.reshape(obs1_list, [ndim1, ndim2, ndim3])
    obs2_array = np.reshape(obs2_list, [ndim1, ndim2, ndim3])

    ########################################
    # save results to file
    if(basis_type == 'E'):
        levels = nbasis_eigen
    elif(basis_type == 'F'):
        levels = nbasis_fock
    filename = (f'kerr_cat_dynamics_K_{K}_delta_{delta}_'
                f'gamma_{gamma}_nbar_{nbar}_levels_{levels}')
    np.save(filename + '_times', times)
    np.save(filename + '_eps1', epsilon_1_list)
    np.save(filename + '_eps2', epsilon_2_list)
    np.save(filename + '_obs1', obs1_array)
    np.save(filename + '_obs2', obs2_array)

    ########################################
    # plot results

    # plot contour P(t) vs eps1
    x = epsilon_1_list / K
    xlabel = r'$\varepsilon_1 / K$'
    y = times * K
    ylabel = r'$t * K$'

    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(
            round(epsilon_2_list[i] / K, 2))

        # obs1
        z = obs1_array[i, :, :].T
        zlabel = r'$<\psi_L(t)>$'
        levels = np.arange(0., 1.1, 0.1)
        fig = plot_2d(x, y, z, title=title,
                      xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      levels=levels)

        # obs2
        z = obs2_array[i, :, :].T
        zlabel = r'$<\Theta_X(t)>$'
        levels = np.arange(0., 1.1, 0.1)
        fig = plot_2d(x, y, z, title=title,
                      xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      levels=levels)

    # plot individual traces P(t) vs t
    x = times * K
    xlabel = r'$t * K$'

    data_label = np.round(epsilon_1_list[:] / K, 2)
    for i in range(0, len(epsilon_2_list), 10):
        title = r'$\varepsilon_2 / K$ = {}'.format(
            round(epsilon_2_list[i] / K, 2))

        # obs1
        y = obs1_array[i, :, :]
        ylabel = r'$<\psi_L(t)>$'
        fig = plot_1d_cuts(x, y, data_label=data_label, title=title,
                           xlabel=xlabel, ylabel=ylabel)

        # obs2
        y = obs2_array[i, :, :]
        ylabel = r'$<\Theta_x(t)>$'
        fig = plot_1d_cuts(x, y, data_label=data_label, title=title,
                           xlabel=xlabel, ylabel=ylabel)

    # show plots
    plt.show()

    print('DONE!!')

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
    # propagation timerange
    parser.add_argument(
        '-t_min', type=float, default=0.0,
        help='Minimun propagation time (default: 0.)')
    parser.add_argument(
        '-t_max', type=float, default=100.0,
        help='Maximun propagation time (default: 100.)')
    parser.add_argument(
        '-t_num', type=int, default=1000,
        help='Number of propagation steps (default: 1000)')
    # initial state parameters
    parser.add_argument(
        '-grid_pts', type=int, default=5000,
        help='Number of points for grid representation (default: 5000)')
    parser.add_argument(
        '-grid_lims', type=float, default=10,
        help='Maximum grid limit (default: 10)')
    parser.add_argument(
        '-x_cutoff', type=float, default=0.,
        help='Cutoff x0 for LHS/RHS initial condition selection (default: 0.)')
    parser.add_argument(
        '-cutoff_value', type=float, default=0.5,
        help=('Cutoff probability for LHS/RHS initial condition selection '
              '(default: 0.5)'))
    parser.add_argument(
        '-sigmoidal_tail', type=float, default=0.5,
        help='Decay of sigmoidal filter (default: 0.5)')
    parser.add_argument(
        '-filter_type', type=str, default=None,
        help='Filter type for initial condition (default: None)')

    args = parser.parse_args()
    main(args)
