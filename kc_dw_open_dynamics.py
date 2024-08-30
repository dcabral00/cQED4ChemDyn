import argparse
import numpy as np
import matplotlib.pyplot as plt

from core_fxns import Fock_to_Eigen_basis
from core_fxns import make_Eigen_real
from core_fxns import propagation_by_semigroup
from core_fxns import get_obs_traces
from core_fxns import filter_state_by_xvalues, get_heaviside_observable
from core_fxns import get_H_kc, get_H_dw, gen_KC_params_from_DW
from core_fxns import get_lindbladian_xp, x_op, p_op


HBAR = 1
labelsz = 12.
ticksz = 11
dpi = 300
plt.style.use('plot_style.txt')
plt.rcParams['figure.constrained_layout.use'] = True
plt.figure(figsize=(14.4, 7.5))


def str_to_bool(value):
    '''
        Converts a string representation of truth to a boolean value.

        Arguments:
        value (str): The string representation of the boolean
                     ('true'/'false').

        Returns:
        bool: The corresponding boolean value.
    '''
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid value for boolean argument."
                                         "Use 'True' or 'False'.")


def main():
    parser = argparse.ArgumentParser()

    #  using Eigen ("E") or Fock ("F") basis?
    parser.add_argument('-basis_type', type=str, choices=['E', 'F'],
                        default='E',
                        help='Type of basis (Fock or Eigen) to use'
                             '(default: E)')
    #  number of oscillator modes
    parser.add_argument('-nbasis_fock', type=int, default=100,
                        help='Number of Fock basis (default: 100)')
    parser.add_argument('-nbasis_eigen', type=int, default=20,
                        help='Number of Eigen basis (default: 20)')
    #  potential parameters; supports negative numbers if using '=' sign
    parser.add_argument('-mass', type=float, default=1836.0,
                        help='Mass for kinetics problem'
                             '(default: 1836. amu for proton)')
    parser.add_argument('-k4', type=float, default=1.0,
                        help='Quartic position coefficient for chemical '
                             'double well; Controls potential walls '
                             '(default: 1.)')
    parser.add_argument('-k2', type=float, default=1.0,
                        help='Quadratic position coefficient for chemical '
                             'double well; Controls barrier walls '
                             '(default: 1.)')
    parser.add_argument('-k1', type=float, default=1.0,
                        help='Linear position coefficient for chemical '
                             'double well; Controls inter-well asymmetry '
                             '(default: 1.)')
    #  mapping parameters
    parser.add_argument('-c_min', type=float, default=0.4,
                        help='Minimun C mapping parameter (default: 0.4)')
    parser.add_argument('-c_max', type=float, default=0.41,
                        help='Maximun C mapping parameter (default: 0.31); '
                             'upperbound is included, '
                             '(ie c_max = c_max + c_step')
    parser.add_argument('-c_inc', type=float, default=0.1,
                        help='Step C mapping parameter (default: 0.1)')

    #  dissipation parameters
    parser.add_argument('-gamma', type=float, default=0.1,
                        help='Dissipation gamma parameter (default: 0.1)')
    parser.add_argument('-nbar', type=float, default=0.5,
                        help='Dissipation nbar parameter (default: 0.5)')
    #  propagation timerange
    parser.add_argument('-t_min', type=float, default=0.0,
                        help='Minimun propagation time (default: 0.)')
    parser.add_argument('-t_max', type=float, default=10000.0,
                        help='Maximun propagation time (default: 100.)')
    parser.add_argument('-t_num', type=int, default=1000,
                        help='Number of propagation steps (default: 1000)')
    #  initial state parameters
    parser.add_argument('-grid_pts', type=int, default=5000,
                        help='Number of points for grid representation '
                             '(default: 5000)')
    parser.add_argument('-grid_lims', type=float, default=10,
                        help='Maximum grid limit (default: 10)')
    parser.add_argument('-x_cutoff', type=float, default=0.,
                        help='Cutoff x0 for LHS/RHS initial condition '
                             'selection (default: 0.)')
    parser.add_argument('-cutoff_value', type=float, default=0.5,
                        help='Cutoff probability for LHS/RHS init condition'
                             'selection (default: 0.5)')
    parser.add_argument('-sigmoidal_tail', type=float, default=0.5,
                        help='Decay of sigmoidal filter (default: 0.5)')
    parser.add_argument('-filter_type', type=str, default=None,
                        help='Filter type for initial condition'
                             '(default: None)')
    #  hamiltonian model
    parser.add_argument('-hamiltonian', type=str, default='KC',
                        choices=['KC', 'DW'],
                        help='Hamiltonian type for dynamics'
                             '(default: KC)')
    parser.add_argument('-system_name', type=str, default='sysname',
                        help='Name of the system being studied')
    parser.add_argument('-LDWcvar', type=str_to_bool, default='False',
                        help='Whether to enable variable c for DW')
    parser.add_argument('-gammaK', type=str_to_bool, default='False',
                        help='Whether to enable gamma dependence on K')
    parser.add_argument('-time_ratio_gn', type=str_to_bool, default='True',
                        help='Whether to enable time array dependence on '
                        'gamma and nth dissipation parameters '
                        '(ie smaller dissipation params require longer time '
                        'for decay to be observed)')

    args = parser.parse_args()
    run_oqd_simul(args)


def print_oqd_params(system, basis_type, nbasis_fock, nbasis_eigen,
                     mass, k4, k2, k1, cmin, cmax, cinc,
                     gamma, nbar,
                     dt, T, filter_type, ham_type):
    #  print to screen
    print('Simulation parameters')
    print('System = ', system)
    print('basis_type = ', basis_type)
    print('nbasis Fock = ', nbasis_fock)
    print('nbasis Eigen = ', nbasis_eigen)
    print(f'Potential parameters - mass = {mass}')
    print(f'Potential parameters - k4 = {k4}')
    print(f'Potential parameters - k2 = {k2}')
    print(f'Potential parameters - k1 = {k1}')
    print(f'C-values range = ({cmin},  {cmax},  {cinc})')
    print(f'Dissipator parameters - gamma = {gamma:1.4e}')
    print(f'Dissipator parameters - nbar = {nbar:1.4e}')
    print(f'Timestep parameters - dt = {dt:1.4f}')
    print(f'Timestep parameters - total T = {T:.2f}')
    print('Initial state parameters - filter_type = ', filter_type)
    print('Hamiltonian - ham_type = ', ham_type)


def get_op_prep(x, p, mass, k4, k2, k1, cval,
                gamma, nbar, grid_pts, grid_lims,
                x_cutoff, cutoff_value, filter_type,
                sigmoidal_tail,
                nbasis_eigen=1, basis_type='E', H_type='KC',
                obs_heaviside=None, LDWcvar=False):
    """
    Prepares the operators for the quantum evolution and observable evaluation
    of double-well like Hamiltonians based upon physical parameters.

    Parameters:
    x (array-like): The position operator matrix for the system.
    p (array-like): The momentum operator matrix for the system.
    mass (float): The mass of the particle or system coordinate under 
                  consideration.
    k4 (float): The coefficient for the quartic term in the potential.
    k2 (float): The coefficient for the quadratic term in the potential.
    k1 (float): The coefficient for the linear term in the potential.
    cval (float): A constant value defining the lengthscale equivalence 
                  between the cQED Kerr-Cat Hamiltonian and chemical double
                  well potential
    gamma (float): Bath coupling damping coefficient.
    nbar (float): Bath thermal parameter.
    grid_pts (int): Number of grid points for discretization of initial state
                    guess.
    grid_lims (tuple): Limits for the grid (min, max).
    x_cutoff (float): Cutoff value for estimating the transition state 
                      position (x) to specify initial state localization. 
    cutoff_value (float): The density cutoff value used to select initial
                          state guess.
    filter_type (str): Type of filter to apply ('S'=Sigmoidal, 'H'=Heaviside).
    sigmoidal_tail (float): Determines decay rate of sigmoidal tail for filter
    nbasis_eigen (int, optional): Number of basis eigenfunctions to use when
                                  truncating the operators (default is 1).
    basis_type (str, optional): Type of basis to use for dynamics propagation
                                ('E'=eigenbasis, 'F'=Fock). Defaults to 'E'.
    H_type (str, optional): Type of Hamiltonian ('KC'=cQED Kerr-Cat, 
                            'DW'=chemical Double-Well). Defaults to 'KC'.
    obs_heaviside (array-like, optional): If provided, prepares the Heaviside
                                          observable for calculation of the
                                          product/reactant correlation 
                                          function. Defaults to None.
    LDWcvar (bool, optional): Whether to use the lengthscale matching 
                              parameter cval for the chemical Double-Well
                              Lindbladian. Defaults to False.

    Returns:
    tuple: A tuple containing:
        - rho0 (array-like): The initial density matrix for dynamics.
        - superop (array-like): The Lindbladian superoperator for the system.
        - obs1 (array-like): The first observable measurement operator.
        - obs2 (array-like): The second observable measurement operator.
        - obs3 (array-like): The third observable measurement operator.

    Notes:
    - The function can handle both quantum and classical systems depending on
      the parameters provided.
    - Ensure that the grid parameters (grid_pts, grid_lims) are chosen to
      accurately represent the system's physical characteristics.
    - The basis and Hamiltonian types should be selected based on the specific
      system and study goals.
    """
    assert H_type in ['KC', 'DW']

    # Hamiltonian selection and construction
    if H_type == 'KC':
        ham = get_H_kc(x, p, mass, k4, k2, k1, c=cval, hbar=HBAR)
    elif H_type == 'DW':
        # TODO CHECK THIS c=1!
        if not LDWcvar:
            cval = 1
        ham = get_H_dw(x, p, mass, k4, k2, k1)
    vals, vecs = np.linalg.eigh(ham)
    vecs = make_Eigen_real(vecs)

    # initial state construction in Fock basis
    psi_l = filter_state_by_xvalues(vecs, grid_pts, grid_lims,
                                    x_cutoff, cutoff_value,
                                    left=True,
                                    filter_type=filter_type,
                                    sigmoidal_tail=sigmoidal_tail)
    psi_r = filter_state_by_xvalues(vecs, grid_pts, grid_lims,
                                    x_cutoff, cutoff_value,
                                    left=False,
                                    filter_type=filter_type,
                                    sigmoidal_tail=sigmoidal_tail)
    # TODO: Check if need to perform double pass here? ^^

    rho0 = np.outer(psi_l, psi_l.conjugate())
    rho0 /= np.trace(rho0)
    #  ------------------------------------------------------------
    #  observables (in Fock basis)
    obs1 = np.copy(rho0)
    obs2 = np.copy(obs_heaviside)
    obs3 = np.outer(psi_r, psi_r.conjugate())
    #  ------------------------------------------------------------
    #  transform from Fock basis to Eigen basis (if optioned)
    if (basis_type == 'E'):
        d_matrix = vecs[:, :nbasis_eigen]
        rho0 = Fock_to_Eigen_basis(rho0, d_matrix)
        obs1 = Fock_to_Eigen_basis(obs1, d_matrix)
        obs2 = Fock_to_Eigen_basis(obs2, d_matrix)
        obs3 = Fock_to_Eigen_basis(obs3, d_matrix)
        superop = get_lindbladian_xp(ham, gamma, nbar, c=cval,
                                 basis_change_matrix=d_matrix)
    else:
        superop = get_lindbladian_xp(ham, gamma, nbar, c=cval)

    return (rho0, superop, obs1, obs2, obs3)


def run_oqd_simul(args):
    '''
        Perform dynamics of Kerr-Cat Hamiltonian and compute some observables.
        Dynamics can be performed using  Eigen basis or Fock basis.
    '''

    #  using Eigen ("E") or Fock ("F") basis?
    basis_type = args.basis_type
    #  number of oscillator modes
    nbasis_fock = args.nbasis_fock
    nbasis_eigen = args.nbasis_eigen
    #  mapping parameter
    cmin = args.c_min
    cmax = args.c_max
    cinc = args.c_inc
    cvalz = np.arange(cmin, cmax + cinc, cinc)

    #  potential parameters
    mass = args.mass
    k4 = args.k4
    k2 = args.k2
    k1 = args.k1
    K, delta, eps2, eps1 = gen_KC_params_from_DW(mass, k4, k2, k1,
                                                 cmin)

    #  dissipation parameters
    gamma = args.gamma
    nbar = args.nbar
    if args.gammaK:
        gamma *= K

    #  propagation timerange
    tmin = args.t_min
    tmax = args.t_max
    tnum = args.t_num
    # times = np.linspace(args.t_min, args.t_max, args.t_num) / K
    times = np.linspace(args.t_min, args.t_max, args.t_num)
    if args.time_ratio_gn:
        times = times / gamma / nbar

    #  initial state parameters
    grid_pts = args.grid_pts
    grid_lims = args.grid_lims
    x_cutoff = args.x_cutoff
    cutoff_value = args.cutoff_value
    sigmoidal_tail = args.sigmoidal_tail
    filter_type = args.filter_type
    ham_type = args.hamiltonian
    system = args.system_name
    LDWcvar = args.LDWcvar

    print_oqd_params(system, basis_type, nbasis_fock, nbasis_eigen,
                     mass, k4, k2, k1, cmin, cmax, cinc, gamma, nbar,
                     times[1], times[-1], filter_type, ham_type)

    #  Heaviside observable (in Fock basis)
    #  Note: Independent of the potential (need to be done only once)
    obs_heaviside = get_heaviside_observable(nbasis_fock, grid_pts,
                                             grid_lims)
    x = x_op(nbasis_fock)
    p = p_op(nbasis_fock)

    #  start main loop of program
    obs1_list = []
    obs2_list = []
    obs3_list = []
    # -----------------------------------------------------------
    for cval in cvalz:
        # -------------------------------------------------------
        op_tuple = get_op_prep(x, p, mass, k4, k2, k1, cval,
                               gamma, nbar, grid_pts, grid_lims,
                               x_cutoff, cutoff_value, filter_type,
                               sigmoidal_tail, nbasis_eigen,
                               basis_type, ham_type, obs_heaviside,
                               LDWcvar)
        rho0, superop, obs1, obs2, obs3 = op_tuple
        # -------------------------------------------------------

        #  run dynamics
        temp, rhos = propagation_by_semigroup(rho0, superop,
                                              times[1], len(times))

        #  compute observable traces
        obs1_traces = get_obs_traces(rhos, obs1)
        obs2_traces = get_obs_traces(rhos, obs2)
        obs3_traces = get_obs_traces(rhos, obs3)

        #  save results
        obs1_list.append(obs1_traces)
        obs2_list.append(obs2_traces)
        obs3_list.append(obs3_traces)

    # transform results to arrays
    # take into account loop over c?
    ndim1 = len(cvalz)
    ndim3 = len(times)
    obs1_array = np.reshape(obs1_list, [ndim1, ndim3])
    obs2_array = np.reshape(obs2_list, [ndim1, ndim3])
    obs3_array = np.reshape(obs3_list, [ndim1, ndim3])

    # # # # # # # # # # # # # #
    #  save results to file
    if(basis_type == 'E'):
        levels = nbasis_eigen
    elif(basis_type == 'F'):
        levels = nbasis_fock
    filename = (f'{system}_{ham_type}_LDWcvar{LDWcvar}_'
                f'dynamics_cmin{cmin:1.2f}'
                f'cmax{cmax:1.2f}_cinc{cinc:1.3f}_'
                f'tmax{tmax:.1f}_tnum{tnum:.1f}_'
                f'gamma_{gamma:1.4e}_nbar_{nbar:1.4e}_'
                f'NbE_{nbasis_eigen}_NbF_{nbasis_fock}')
    np.save(filename + '_times', times)
    np.save(filename + '_cvalz', cvalz)
    np.save(filename + '_obs1', obs1_array)
    np.save(filename + '_obs2', obs2_array)
    np.save(filename+'_obs3', obs3_array)

    print('DYNAMICS DONE!!')

    return


if __name__ == '__main__':
    main()
