import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special
from scipy.integrate import simpson

# from kc_dw_hams import kerr_cat_hamiltonian


HBAR = 1


def Fock_to_Eigen_basis(target, d_matrix):
    '''
        Given a 'target' matrix in the Fock basis and a
        transformation matrix to the Eigen basis, returns
        the similarity transformation of the target as follows:

        $D^{*}AD$

        Arguments:
            target (numpy array, N-by-N): target matrix in the Fock
                                          basis to be transformed
            d_matrix (numpy array, N-by-M): Fock-to-Eigen
                                            transformation matrix
        Returns:
            transformed_matrix (numpy array, M-by-M): matrix in
                                                      Eigen basis
    '''
    d_matrix_t = np.transpose(d_matrix)
    transformed_matrix = d_matrix_t @ target @ d_matrix
    return transformed_matrix


def make_Eigen_real(vecs):
    '''
        Transformed Eigenvectors from complex to real (BRUTE FORCE)
    '''
    ndim1, ndim2 = vecs.shape
    for i in range(ndim2):
        temp = vecs[:, i].imag
        if(temp.any() != 0):
            raise ValueError(f'[make_Eigen_real] Error. > {temp}'
                             'Imaginary part of Eigenvector not zero')
    return np.real(vecs)


def destroy_q(N, offset=0):
    '''
        Destruction (lowering) operator.

        Parameters:
            N (int): dimension of Hilbert space
            offset (float): Offsetting value entries

        Returns:
            off-diagonal matrix containing destruction operator
    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset + 1, N + offset, dtype=complex))
    return np.diag(data[:], 1)


def create_q(N, offset=0):
    '''
        Creation (raising) operator (adjoint of destruction operator)

        Parameters:
            N (int): dimension of Hilbert space
            offset (float): Offsetting value entries

        Returns:
            off-diagonal matrix containing destruction operator
    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    q0 = destroy_q(N, offset=offset)
    return np.conjugate(q0.T)


def x_op(N, c=1):
    return c * (destroy_q(N) + create_q(N)) / np.sqrt(2)


def p_op(N, c=1, hbar=HBAR):
    return 1j * hbar * (create_q(N) - destroy_q(N)) / np.sqrt(2) / c


def ladder_op(N, c=1, hbar=HBAR, dagger=False):
    x_comp = x_op(N, 1) / c
    p_comp = 1j * c * p_op(N, 1) / hbar
    if dagger:
        p_comp *= -1
    return (x_comp + p_comp) / np.sqrt(2)


def kerr_cat_hamiltonian(N, K, epsilon_2, delta, epsilon_1, hbar=HBAR):
    '''
        Implements the asymmetric Kerr-Cat Hamiltonian based on the following
        expression:

        H = delta * (a_dag * a)
            - K * (a_dag)^2a^2
            + epsilon_2 ((a_dag)^2 + a^2)
            + epsilon (a_dag + a)

        Parameters:
            N (int): dimension of the Hilbert space
            K (float): Kerr non-linearity parameter
            epsilon_2 (float): drive frequency
            delta (float):
            epsilon_1 (float): asymmetry parameter
    '''
    # Hamiltonian basis operators
    ac = destroy_q(N)
    ac_dag = create_q(N)
    # Term-by-term construction
    ac_sq = ac @ ac  # a^2
    ac_dag_sq = ac_dag @ ac_dag  # (a^+) ^2
    nc = ac_dag @ ac  # a a^+
    kc_sq = ac_dag_sq @ ac_sq  # (a^+)^2 a^2
    xc = ac_dag + ac  # (a^+ + a)
    xc_sq = ac_dag_sq + ac_sq  # (a^+)^2 + a^2

    # hamiltonian
    H = (delta * nc - K * kc_sq + epsilon_2 * xc_sq + epsilon_1 * xc)
    return -1 * H * hbar


def get_V_dw(x, k4, k2, k1, grid=False):
    '''
    Generates the chemical potential V(x) defined with prefactors k4, k2, k1:

    $$ V(x) = k_{4} x^{4} - k_{2} x^{2} + k_{1} x $$
    '''
    if not grid:
        x_sq = np.matmul(x, x)
        x_qt = np.matmul(x_sq, x_sq)
    else:
        x_sq = x**2
        x_qt = x_sq**2
    return k4*x_qt - k2*x_sq + k1*x


def get_H_dw(x, p, m, k4, k2, k1):
    '''
    Generates the Hamiltonian for the chemical potential:

    $$ H = \frac{p^{2}}{2m} + V(x) $$

    Inputs:
        - `x`: np.array of position values
        - `p`: np.array of momentum values
        - `m`: mass of the system of interest
        - `k4`: prefactor for chemical potential (see get_V_dw function)
        - `k2`: prefactor for chemical potential (see get_V_dw function)
        - `k1`: prefactor for chemical potential (see get_V_dw function)
    '''
    p_sq = np.matmul(p, p)
    return p_sq/(2*m) + get_V_dw(x, k4, k2, k1)


def get_H_dw_fock(create_q, destroy_q, m, k4, k2, k1, c=1.0, hbar=HBAR):
    '''
    Generates the Hamiltonian for the chemical potential:

    $$ H = \frac{p^{2}}{2m} + V(x) $$

    Inputs:
        - `x`: np.array of position values
        - `p`: np.array of momentum values
        - `m`: mass of the system of interest
        - `k4`: prefactor for chemical potential (see get_V_dw function)
        - `k2`: prefactor for chemical potential (see get_V_dw function)
        - `k1`: prefactor for chemical potential (see get_V_dw function)
    '''
    p = (destroy_q - create_q) / c / np.sqrt(2)
    x = (destroy_q + create_q) * c / np.sqrt(2)
    p_sq = np.matmul(p, p)
    return - hbar**2 * p_sq/(2*m) + get_V_dw(x, k4, k2, k1)


def get_H_kc(x, p, m, k4, k2, k1, c=1.0, hbar=HBAR):
    '''
        Generates the Kerr-Cat Hamiltonian in phase space representation, by
        adding the 4th order momentum and mixed momentum/position operators.

        Inputs:
            - `x`: np.array of position values
            - `p`: np.array of momentum values
            - `m`: mass of the system of interest
            - `k4`: prefactor for chemical potential (see get_V_dw function)
            - `k2`: prefactor for chemical potential (see get_V_dw function)
            - `k1`: prefactor for chemical potential (see get_V_dw function)
            - `c`: scaling factor for equivalence of the chemical
                   and Kerr-Cat Hamiltonian
        Output:
            - : Kerr-Cat Hamiltonian expressed in phase space representation
    '''
    dw_segment = get_H_dw(x, p, m, k4, k2, k1)
    K = get_K(k4, c)
    x_sq = np.matmul(x, x)
    p_sq = np.matmul(p, p)
    p_qt = np.matmul(p_sq, p_sq)
    kc_segment = 0.25/hbar**4 * K * (c**4*p_qt
                                     + np.matmul(x_sq, p_sq)
                                     + np.matmul(p_sq, x_sq))
    return dw_segment + kc_segment


def get_K(k4, c):
    '''
        Expression for K in terms of chemical variables and correspondence
        'c' constant
    '''
    return 4 * c**4 * k4


def get_Delta(k4, k2, m, c, hbar=HBAR):
    '''
        Expression for Delta in terms of chemical variables and correspondence
        'c' constant
    '''
    return c**2 * k2 - hbar**2 / (2 * m * c**2) - 2 * get_K(k4, c)


def get_epsilon2(k2, m, c, hbar=HBAR):
    '''
        Expression for epsilon2 in terms of chemical variables and
        correspondence 'c' constant
    '''
    return c**2 * k2 / 2 + hbar**2 / (4 * m * c**2)


def get_epsilon1(k1, c):
    '''
        Expression for epsilon1 in terms of chemical variables and
        correspondence 'c' constant
    '''
    return - k1 * c / np.sqrt(2)


def gen_KC_params_from_DW(m, k4, k2, k1, c=1.0):
    K = get_K(k4, c)
    Delta = get_Delta(k4, k2, m, c)
    e2 = get_epsilon2(k2, m, c)
    e1 = get_epsilon1(k1, c)
    return (K, Delta, e2, e1)


def len_white_space(lwspace, lstr):
    lws = (lwspace - len(lstr)) / 2
    return (int(np.ceil(lws)), int(np.floor(lws)))


def pretty_print(m, k4, k2, k1, c,
                 print_header=False, K_units=True):
    K, Delta, e2, e1 = gen_KC_params_from_DW(m, k4, k2, k1, c)

    str_Delta = 'D'
    str_e2 = 'e2'
    str_e1 = 'e1'

    if K_units:
        Delta /= K
        e2 /= K
        e1 /= K
        str_Delta = str_Delta + '/K'
        str_e2 = str_e2 + '/K'
        str_e1 = str_e1 + '/K'

    if print_header:
        plwD, slwD = len_white_space(13, str_Delta)
        plwe2, slwe2 = len_white_space(13, str_e2)
        plwe1, slwe1 = len_white_space(13, str_e1)

        print('   c   |',
              ' '*6, 'K', ' '*6, '|',
              ' '*plwD, str_Delta, ' '*slwD, '|',
              ' '*plwe2, str_e2, ' '*slwe2, '|',
              ' '*plwe1, str_e1, ' '*slwe1, '|', sep='')
        print('-'*65)

    if K_units:
        print(f' {c:1.3f} | {K:+1.8f} | {Delta:+1.4E} |'
              f' {e2:+1.4E} | {e1:+1.4E} |', sep='')
    else:
        print(f' {c:1.3f} | {K:+1.8f} | {Delta:+1.8f} |'
              f' {e2:+1.8f} | {e1:+1.8f} |', sep='')




def get_dissipator_xp(N, kappa, nth, c=1, basis_change_matrix=None,
                      hbar=HBAR):
    # operator definitions
    ac = ladder_op(N, c, hbar)
    ac_dag = ladder_op(N, c, hbar, dagger=True)
    if basis_change_matrix is not None:
        assert N == basis_change_matrix.shape[0]
        ac = Fock_to_Eigen_basis(ac, basis_change_matrix)
        ac_dag = Fock_to_Eigen_basis(ac_dag, basis_change_matrix)
    d_op = len(ac)

    aa_dag = ac @ ac_dag
    a_daga = ac_dag @ ac
    eye = np.eye(d_op)

    # dissipator definitions
    Da = (np.kron(ac_dag, ac_dag)
          - 0.5 * (np.kron(eye, aa_dag) + np.kron(aa_dag, eye)))
    Dadag = (np.kron(ac, ac)
             - 0.5 * (np.kron(eye, a_daga) + np.kron(a_daga, eye)))

    return kappa * ((1 + nth) * Dadag + nth * Da)


def get_lindbladian_xp(H, kappa, nth, c=1.0, basis_change_matrix=None,
                       hbar=HBAR):
    d = len(H)  # dimension of the system

    if basis_change_matrix is not None:
        assert d == basis_change_matrix.shape[0]
        H = Fock_to_Eigen_basis(H, basis_change_matrix)

    dH = len(H)

    # Hamiltonian part
    superH = -1j / hbar * (np.kron(np.eye(dH), H)
                           - np.kron(H.T, np.eye(dH)))
    superL = get_dissipator_xp(d, kappa, nth, c, basis_change_matrix,
                               hbar)
    return superH + superL


def Liouvillian(H, Ls, hbar=1):
    d = len(H)  # dimension of the system
    # Hamiltonian part
    superH = -1j / hbar * (np.kron(np.eye(d), H) - np.kron(H.T, np.eye(d)))
    # L* \otimes L - 1/2 (1 \otimes L^{dag} @ L + L^T @ L^* \otimes 1)
    superL = sum([np.kron(L.conjugate(), L)
                  - 1 / 2 * (np.kron(np.eye(d), L.conjugate().T.dot(L)) +
                             np.kron(L.T.dot(L.conjugate()), np.eye(d))
                             ) for L in Ls])
    return superH + superL


def propagation_by_semigroup(rho0, superop, dt=0.5, m=20):
    '''
        Propagates for a density matrix rho0 with a superoperator for
        m timesteps in increments of dt (superoperator propagates for only
        a single step)
    '''
    d = len(rho0)  # dimension or the state
    P = scipy.linalg.expm(superop * dt)  # propagator
    times, rhos = [], []  # allocate lists
    # times, pops = [], []  # allocate lists
    t, rho = 0.0, rho0  # initialise
    # propagate
    # for k in range(m):
    k = 0
    while k < m:
        rhos.append(rho)
        times.append(t)  # append current time
        # propagate time and state
        t, rho = t + dt, np.reshape(P.dot(np.reshape(rho, (d**2, 1))), (d, d))
        k += 1
    return (np.array(times), np.array(rhos))


def get_obs_traces(rhots, obs):
    '''
        Computes the partial trace between the density operators (rhots) and
        the observable operator (obs)

        $\text{Tr}\\{ \rho _t \rho _{\text{obs}}\\}$
    '''
    obs_t = []
    for rhot in rhots:
        obs_t.append(np.real(np.trace(rhot @ obs)))
    return np.array(obs_t)


def get_QHO_basis(q, level, max_level=250):
    # harmonic oscillator eigenfunctions in position representation
    if level > max_level:
        print(f"ERROR WARNING: REPRESENTATION FAILS BECAUSE {level}! is large"
              f"Setting level={level} to level=MAX_LEVEL={max_level}")
        level = max_level
    norm = (1. / np.pi)**.25 / np.sqrt(2.**level * special.factorial(level))
    hermite_polynomial = special.hermite(level)(q)
    gaussian_factor = np.exp(-q**2 / 2.)
    return norm * hermite_polynomial * gaussian_factor


def get_grid_init_state_in_fock(levels, grid_state,
                                n_grid_pts=None, grid_bound=None):
    '''
        Given a number of QHO levels and a grid-based initial state,
        generates the initial state representation in terms of fock
        states.


        Adapted from code sample by Pouya Khazaei
    '''
    if n_grid_pts is None or grid_bound is None:
        n_grid_pts = grid_state.size
        grid_bound = (max(grid_state) - min(grid_state)) / 2
    # grid
    x = np.linspace(-grid_bound, grid_bound, n_grid_pts)
    # array for fock state coefficient
    wav = np.zeros(levels)
    # make initial gaussian wavepacket
    for i in range(levels):
        qho_basis = get_QHO_basis(x, i)
        # get expansion coeffs by projecting eigenfxns on gaussian init state
        wav[i] = simpson(np.conj(qho_basis) * grid_state, x)
    return wav / np.linalg.norm(wav)


def get_fock_init_state_in_gaussian_grid(fock_state, n_grid_pts,
                                         grid_bound, levels=None):
    '''
        Given a particular fock state returns its grid based representation
        for a given number of grid points and grid boundary.

        Test:
        ng = 5000 # size of grid
        l = 10 # grid goes between -l and l

        fock_state = np.array([0, 1, 0], dtype=float)
        fock_state /= np.linalg.norm(fock_state)
        x_fock, gaussianfock = get_fock_init_state_in_gaussian_grid(
                                                        fock_state, ng, l)
        plt.plot(x_fock, gaussianfock)
        plt.show()

        Adapted from code sample by Pouya Khazaei
    '''
    fock_state = np.array(fock_state)
    fock_state = fock_state / np.linalg.norm(fock_state)
    if levels is None:
        levels = fock_state.size
    # grid
    x = np.linspace(-grid_bound, grid_bound, n_grid_pts)
    # use coeffs to get position wavefxn with QHO wavefunctions
    wav_pos = np.zeros(n_grid_pts)
    for i in range(levels):
        qho_basis = get_QHO_basis(x, i)
        # use coeffs to make the initial state with the HO wavefunctions
        wav_pos += fock_state[i] * qho_basis
    return x, wav_pos


def get_sigmoidal(x, x0=0.0, tail=1.0):
    '''
        Note: sigmoidal function should have negative exponent;
        ie 1/ (1+e^(-(x-x0)/tail)) according to the definition
    '''
    sigmoidal = []
    for y in x:
        sigmoidal.append(1 / (1 + np.exp((y - x0) / tail)))
    return np.array(sigmoidal)


def get_heaviside(x, x0=0.0):
    heaviside = np.array([1 if y > x0 else 0 for y in x])
    return heaviside


def filter_state_by_xvalues(all_states, n_grid_pts, grid_bound, x_split=0.,
                            cutoff=0.5, left=True, filter_type=None,
                            sigmoidal_tail=1.0, return_state=False):
    '''
        Find the first state in the Fock basis representation satisfying 
        a LHS/RHS criteria, convert the selected state to the grid
        representation and apply a filtering function (if optioned), and
        return the filtered state in the Fock basis representation.

        The LHS criteria is defined as:
                |psi(x<x_split)|^2 > cutoff
        whereas the RHS criteria is defined as:
                |psi(x>x_split)|^2 > cutoff

        Inputs:
            - all_states: (list or np.array)
                    List or array of eigenstates defined in Fock Basis
            - n_grid_pts: (int)
                    Specifies number of points to use in grid construction
            - grid_bound: (float)
                    Specifies the outer limits of the position grid.
            - x_split: (float)
                    The central value used to divide xgrid into LHS/RHS.
            - cutoff: (float)
                    The probability cutoff to use.
            - left: (bool)
                    Specifies whether to filter based on the LHS or RHS.
            - filter_type: ('H' or 'S')
                    String specifying how to filter things; currently not used
            - sigmoidal_tail (float)
                    Determines the decay rate of the sigmoidal tail
                    (only used if filter_type='S')
                    (default=1.0)
            - return_state (bool)
                    Determines whether to return the final state and index
                    (default=False)

        Returns:
            First state matching filter criteria in the Fock basis
            representation.

    '''
    # Define position grid:
    xgrid = np.linspace(-grid_bound, grid_bound, n_grid_pts)
    # Grab indices of the x-grid for the left/right
    if left:
        x_indices = np.argwhere(xgrid < x_split)
    else:
        x_indices = np.argwhere(xgrid > x_split)

    # Loop over eigenstates, checking each for criteria
    n_eigenstates = len(all_states)

    for ii in range(n_eigenstates):
        tmp_state = all_states[:, ii]

        # Convert from Fock to Coordinate Space
        xs, input_state = get_fock_init_state_in_gaussian_grid(tmp_state,
                                                               n_grid_pts,
                                                               grid_bound)

        # Compute the probability density
        prob_density = input_state**2

        # Get probability in region of interest
        prob_reduced = prob_density[x_indices].sum()
        prob_reduced = prob_reduced / prob_density.sum()

        # Check if probability in region of interest is above cutoff and apply
        # filter.
        if prob_reduced >= cutoff:

            if filter_type is None:
                filtered_state = input_state

            elif filter_type == 'H':
                # TODO make product of two vectors to support other filters``
                # Eliminate other values based on Heaviside
                filtered_state = np.zeros_like(input_state)
                filtered_state[x_indices] = input_state[x_indices]
                # filter_vector = get_heaviside(xgrid, x_split)
                # filtered_state = np.multiply(filter_vector, input_state)

            elif filter_type == 'S':
                filter_vector = get_sigmoidal(xgrid, x_split, sigmoidal_tail)
                if left:
                    filtered_state = np.multiply(filter_vector, input_state)
                else:
                    filtered_state = np.multiply(
                        np.flip(filter_vector), input_state)

            else:
                raise ValueError('[filter_state_by_xvalues]. Error. '
                                 f'filter_type={filter_type} not implemented.'
                                 'Use one of [None, "H", "S"]')

            # Convert to Fock space
            output_fock = get_grid_init_state_in_fock(n_eigenstates,
                                                      filtered_state,
                                                      n_grid_pts, grid_bound)
            output_fock = output_fock / np.linalg.norm(output_fock)
            side = ''
            if left:
                side = 'left'
            else:
                side = 'right'
            print(f'found state {ii} matching {cutoff*100}% '
                  f'density on the {side}')
            if return_state:
                return (output_fock, ii)
            else:
                return output_fock
        else:
            pass
    raise ValueError('[filter_state_by_xvalues]. Error. '
                     'Could not find state satisfying criteria')


def get_heaviside_observable(levels, n_grid_pts, grid_bound, x0=0.0):
    ''' Generates a Heaviside observable on a Fock basis representation '''

    # define grid
    x = np.linspace(-grid_bound, grid_bound, n_grid_pts)

    # define grid
    heav = get_heaviside(x, x0)

    # construct Heaviside observable
    obs = np.zeros([levels, levels])
    for i in range(levels):
        psi_i = get_QHO_basis(x, i).conj()
        for j in range(levels):
            psi_j = get_QHO_basis(x, j)
            integrand = psi_i * psi_j * heav
            obs[i, j] = simpson(integrand, x)

    return obs


if __name__ == '__main__':
    # number of oscillator modes
    N = 30
    # Kerr-Cat Hamiltonian parameters
    K = 1.
    epsilon_2 = 2 / K
    delta = 0. / K
    epsilon_1 = 1. / K
    epsilon_1_list = np.arange(1, 1.1, 1) / K

    print('Simulation parameters')
    print('nbasis = ', N)
    print('Potential parameters - K = ', K)
    print('Potential parameters - delta = ', delta)
    print('Potential parameters - eps1 = ', epsilon_1)
    print('Potential parameters - eps2 = ', epsilon_2)

    ham = kerr_cat_hamiltonian(N, K, epsilon_2, delta, epsilon_1, HBAR)
    vals, vecs = np.linalg.eig(ham)
    vals = np.round(np.real(vals), decimals=2)
    vecs = np.real(vecs)

    grid_pts = 5000
    grid_lims = 10
    x_cutoff = 0
    cutoff_value = 0.9
    state_list = []
    left = True
    xgrid = np.linspace(-grid_lims, grid_lims, grid_pts)
    dx = xgrid[1] - xgrid[0]
    for idx, wvfxn in enumerate(vecs[:].T):
        print(vals[idx])
        x, level = get_fock_init_state_in_gaussian_grid(
            wvfxn, grid_pts, grid_lims)
        prob = level**2
        x_l = np.argwhere(x < x_cutoff)
        x_r = np.argwhere(x > x_cutoff)
        prob_l = sum(prob[x_l]) / sum(prob)
        prob_r = sum(prob[x_r]) / sum(prob)

        if(left):
            prob = prob_l
        else:
            prob = prob_r

        if prob > cutoff_value:
            state_list.append(level)
            break

    if len(state_list) > 0:
        # obtain filter
        heaviside_filter = np.ones(x.size)
        heaviside_filter[x_r] = 0
        filter_vector = heaviside_filter
        filter_vector = get_sigmoidal(xgrid, x_cutoff, .5)
        # apply filter
        filtered_state = np.multiply(state_list[0], filter_vector)
        # filtered_state /= np.linalg.norm(filtered_state)
        # print(np.linalg.norm(filtered_state))

    filtered_fock = get_grid_init_state_in_fock(
        N, filtered_state, grid_pts, grid_lims)
    rec_x, rec_level = get_fock_init_state_in_gaussian_grid(
        filtered_fock, grid_pts, grid_lims)

    # apply H filter
    h_fock = filter_state_by_xvalues(vecs, grid_pts, grid_lims, x_cutoff,
                                     cutoff_value, left=left,
                                     filter_type='H')
    rec_h, rec_h_level = get_fock_init_state_in_gaussian_grid(h_fock,
                                                              grid_pts,
                                                              grid_lims)

    # apply S filter
    s1_fock = filter_state_by_xvalues(vecs, grid_pts, grid_lims, x_cutoff,
                                      cutoff_value, left=left,
                                      filter_type='S', sigmoidal_tail=1.0)
    rec_s1, rec_s1_level = get_fock_init_state_in_gaussian_grid(s1_fock,
                                                                grid_pts,
                                                                grid_lims)

    s2_fock = filter_state_by_xvalues(vecs, grid_pts, grid_lims, x_cutoff,
                                      cutoff_value, left=left,
                                      filter_type='S', sigmoidal_tail=0.5)
    rec_s2, rec_s2_level = get_fock_init_state_in_gaussian_grid(s2_fock,
                                                                grid_pts,
                                                                grid_lims)

    s3_fock = filter_state_by_xvalues(vecs, grid_pts, grid_lims, x_cutoff,
                                      cutoff_value, left=left,
                                      filter_type='S', sigmoidal_tail=0.1)
    rec_s3, rec_s3_level = get_fock_init_state_in_gaussian_grid(s3_fock,
                                                                grid_pts,
                                                                grid_lims)

    s4_fock = filter_state_by_xvalues(vecs, grid_pts, grid_lims, x_cutoff,
                                      cutoff_value, left=left,
                                      filter_type='S', sigmoidal_tail=0.01)
    rec_s4, rec_s4_level = get_fock_init_state_in_gaussian_grid(s4_fock,
                                                                grid_pts,
                                                                grid_lims)

    # plot
    plt.plot(x, state_list[0], label='unfiltered state')
    plt.plot(x, filtered_state, label='filtered state')
    # plt.plot(rec_x, rec_level, label='rec_level')
    plt.plot(rec_h, rec_h_level, label='Heavide')
    plt.plot(rec_s1, rec_s1_level, label='sigmoidal, width=1')
    plt.plot(rec_s2, rec_s2_level, label='sigmoidal, width=0.5')
    plt.plot(rec_s3, rec_s3_level, label='sigmoidal, width=0.1')
    plt.plot(rec_s4, rec_s4_level, label='sigmoidal, width=0.01')
    plt.legend()
    plt.show()
