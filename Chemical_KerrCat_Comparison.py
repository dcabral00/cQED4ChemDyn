import qutip as qt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib import ticker as mticker
from tqdm.auto import trange
from scipy.special import eval_hermite, factorial
from numpy.polynomial.hermite import hermval


class Chemical_KerrCat_Analysis:
    def __init__(self, N_basis=None, system_name=None, mass=None, k1=None,
                 k2=None, k4=None, savedir=None, hbar=1.):

        '''
        potential_keywords:
        '''
        self.hbar             = hbar
        self.system_name      = system_name
        self.N_basis          = N_basis
        self.mass             = mass
        self.k1               = k1
        self.k2               = k2
        self.k4               = k4
        self.savedir          = savedir

        # Explicitly Set Parameters
        self.X_op             = self._init_X()
        self.P_op             = self._init_P()
        self.H_chem           = self._get_H_chem()
        # Calculate Chemical Info
        self.chemical_data    = self._generate_chemical_data()
        # KC Device Parameters
        self.K_valz           = None
        self.Delta_valz       = None
        self.epsilon2_valz    = None
        self.epsilon1_valz    = None

        # Attributes to be filled by the calculation
        self.KC_data            = None
        self.energy_differences = None
        self.overlaps           = None
        self.corrected_indices  = None
        self.corrected_overlaps = None


    def _init_X(self):
        '''
            Create the X operator defined in terms of a/a^{\dagger}
        '''
        _a_op     = qt.destroy(self.N_basis)
        X_op = (_a_op.dag() + _a_op)/np.sqrt(2.0)
        return(X_op)


    def _init_P(self):
        '''
            Create the P operator defined in terms of a/a^{\dagger}
        '''
        _a_op     = qt.destroy(self.N_basis)
        P_op = 1j*(_a_op.dag() - _a_op)/(np.sqrt(2.0))
        return(P_op)


    def _generate_chemical_data(self):
        chemical_keys         = ['Energies', 'Wavefunctions']
        _chem_ens, _chem_wfns = self.H_chem.eigenstates()
        chemical_data    = dict(zip(chemical_keys, [_chem_ens, _chem_wfns]))
        return(chemical_data)


    def _get_V_chem(self, xv):
        '''
        Generates the chemical potential V(x) defined with prefactors k4, k2, k1:

        $$ V(x) = k_{4} x^{4} - k_{2} x^{2} + k_{1} x $$

        Arguments:
            - `xv` (qutip.Qobj or np.array): position values

        Returns:
            - (qutip.Qobj or np.array): potential energy
        '''
        return(self.k4*xv**4 - self.k2*xv**2 + self.k1*xv)


    def _get_H_chem(self):
        '''
        Generates the Hamiltonian for the chemical potential:

        $$ H = \frac{p^{2}}{2m} + V(x) $$

        Arguments:
            - `x` (qt.Qobj or np.array): position operator/list of position values at which to evaluate the potential energy
            - `p`(qt.Qobj or np.array): momentum operator/list of momentum values at which to evaluate the potential energy
            - `mass` (float): mass of the system of interest
            - `k4` (float): prefactor for chemical potential (see get_V_dw function)
            - `k2` (float): prefactor for chemical potential (see get_V_dw function)
            - `k1` (float): prefactor for chemical potential (see get_V_dw function)

        Returns:
            - `H_op` (qutip.Qobj or np.array): Hamiltonian as defined above.
        '''
        T_op        = self.P_op**2 / (2. * self.mass)
        V_op        = self._get_V_chem(self.X_op)
        H_op        = T_op + V_op
        return(H_op)


    def _get_K(self, tmp_c=None):
        '''
            Calculate value of K in terms of chemical variable ('k4') and correspondence
            'c' constant.
        '''
        if tmp_c:
            return(4 * self.k4 * tmp_c**4)
        else:
            return(4 * self.k4 * self.cvalz**4)


    def _get_Delta(self, tmp_c=None):
        '''
            Calculate value of $\Delta$ in terms of chemical variables 'k4', 'k2', 'm' and correspondence
            'c' constant
        '''
        if tmp_c:
            return(self.k2 * tmp_c**2 - self.hbar**2 / (2 * self.mass * tmp_c**2) - 2 * self._get_K(tmp_c))
        else:
            return(self.k2 * self.cvalz**2 - self.hbar**2 / (2 * self.mass * self.cvalz**2) - 2 * self._get_K())


    def _get_epsilon2(self, tmp_c=None):
        '''
            Calculate value of $\epsilon_2$ in terms of chemical variables 'k2', 'm' and
            correspondence 'c' constant.
        '''
        if tmp_c:
            return((0.5 * self.k2 * tmp_c**2) + (self.hbar**2 / (4 * self.mass * tmp_c**2)))
        else:
            return((0.5 * self.k2 * self.cvalz**2) + (self.hbar**2 / (4 * self.mass * self.cvalz**2)))


    def _get_epsilon1(self, tmp_c=None):
        '''
            Calculate value of $\epsilon_1$ in terms of chemical variable 'k1' and
            correspondence 'c' constant
        '''
        if tmp_c:
            return(- self.k1 * tmp_c / np.sqrt(2))
        else:
            return(- self.k1 * self.cvalz / np.sqrt(2))


    def _calculate_device_params(self):
        '''
        Calculate device parameters in terms of chemical parameters for all values of c.
        '''
        self.K_valz        = self._get_K()
        self.Delta_valz    = self._get_Delta()
        self.epsilon2_valz = self._get_epsilon2()
        self.epsilon1_valz = self._get_epsilon1()
        return


    def _calculate_scaled_params(self):
        scaled_Delta = self.Delta_valz / self.K_valz
        scaled_eps1  = self.epsilon1_valz / self.K_valz
        scaled_eps2  = self.epsilon2_valz / self.K_valz
        device_keys  = ['Delta/K', 'eps1/K', 'eps2/K']
        device_vals  = [scaled_Delta, scaled_eps1, scaled_eps2]
        self.scaled_device_params = dict(zip(device_keys, device_vals))
        return


    def get_H_kc(self, tmp_c, tmp_K, tmp_Delta, include_constants=False):
        '''
            Generates the Kerr-Cat Hamiltonian in phase space representation, by
            adding the 4th order momentum and mixed momentum/position operators.

            Arguments:
                - `x` (qt.Qobj or np.array): position operator/list of position values at which to evaluate the potential energy
                - `p`(qt.Qobj or np.array): momentum operator/list of momentum values at which to evaluate the potential energy
                - `mass` (float): mass of the system of interest
                - `k4` (float): prefactor for chemical potential (see get_V_dw function)
                - `k2` (float): prefactor for chemical potential (see get_V_dw function)
                - `k1` (float): prefactor for chemical potential (see get_V_dw function)
                - `c` (float): scaling factor for equivalence of the chemical
                       and Kerr-Cat Hamiltonian
            Returns:
                - (qt.Qobj or np.array): Kerr-Cat Hamiltonian expressed in phase space representation
        '''
        dw_segment = self.H_chem
        kc_segment = 0.25 * tmp_K / self.hbar**4 * ((tmp_c**4 * self.P_op**4) + self.hbar**2 * (self.X_op**2 * self.P_op**2 + self.P_op**2 * self.X_op**2))
        if include_constants:
            kc_segment += 0.75 * tmp_K - tmp_Delta / 2

        return(dw_segment + kc_segment)


    def _unscramble_indices(self, kc_ens, kc_wfns):
        '''
        Function to unscramble the wavefunction indices.
        Arguments:
            - `kc_ens` (np.array or list): kerr-cat energies
            - `kc_wfns` (np.ndarray or list): chemical hamiltonian eigenstates

        Returns:
            - `index_mapping` (np.ndarray): mapping of scrambled to properly ordered indices for the KC wavefunctions
            - `out_overlaps` (list): calculated overlaps
        '''
        chem_ens  = self.chemical_data['Energies']
        chem_wfns = self.chemical_data['Wavefunctions']
        index_mapping = np.arange(0, len(chem_ens), step=1)
        pairwise_energy_difference = abs(np.subtract.outer(chem_ens, kc_ens))
        out_overlaps = []
        for ii in range(self.N_check):
            current_best_0 = 0.
            for jj in range(self.N_check):
                current_best = max(current_best_0, np.abs(kc_wfns[ii].overlap(chem_wfns[jj])))
                if current_best > current_best_0:
                    index_mapping[ii] = jj
                    current_best_0 = current_best
            out_overlaps.append(current_best_0)
        return(index_mapping, out_overlaps)


    def loop_over_c(self, N_check=40, include_constants=False):

        # Initialize Empty Arrays
        self.N_check = N_check
        all_overlaps = np.zeros(shape=(len(self.cvalz), self.N_check))
        all_energy_diffs = np.zeros(shape=(len(self.cvalz), self.N_basis))
        # Create Empty dict to store data
        ens_chem  = self.chemical_data['Energies']
        wfns_chem = self.chemical_data['Wavefunctions']
        data_dict = {}
        for ii in trange(len(self.cvalz)):
            cval = self.cvalz[ii]
            cval = np.round(cval, 4) # TODO: This rounding will cause problems with ultra fine c-scan
            # Calculate the Kerr-Cat Hamiltonian at this value of c
            # TODO: include_constants flag exists but isn't allowed to be set as of right now
            H_kc = self.get_H_kc(cval, self.K_valz[ii], self.Delta_valz[ii], include_constants=include_constants)
            # Calculate Energies and Wavefunctions
            ens_kc, wfns_kc = H_kc.eigenstates()
            # Unscramble
            index_mapping, tmp_overlaps = self._unscramble_indices(ens_kc, wfns_kc)
            data_dict[cval] = {'Energies': ens_kc,
                               'Wavefunctions': wfns_kc,
                               'index_mapping': index_mapping,
                               'tmp_overlaps': tmp_overlaps}
            all_overlaps[ii] = tmp_overlaps

            # Calculate energy differences
            all_energy_diffs[ii] = np.abs(ens_kc - ens_chem)

        self.KC_data = data_dict
        self.energy_differences = all_energy_diffs
        self.overlaps = all_overlaps
        return


    def unscramble_over_c(self, n_states, state_span):
        '''
        Function to unscramble states over the c index.

        Arguments:
                - `n_states` (int): determines the number of states to do the unscrambling for
                - `state_span` (int): determines the span of neighboring c-values to check
        '''
        corrected_overlaps = []
        corrected_idx = []
        wfns_dw = self.chemical_data['Wavefunctions']
        for dw_idx in trange(n_states):
            tmp_overlap_list = []
            c_idx = []
            for jj in range(len(self.cvalz)):
                cval_key = np.round(self.cvalz[jj], 4)
                if dw_idx < state_span:
                    _overlap = wfns_dw[dw_idx].overlap(self.KC_data[cval_key]['Wavefunctions'][dw_idx])
                    tmp_overlap_list.append(abs(_overlap))
                    c_idx.append(dw_idx)
                else:
                    # Calculate overlaps with all wfns in a given range
                    left_idx = dw_idx-state_span
                    right_idx = dw_idx+state_span
                    _dw_overlaps = [abs(wfns_dw[dw_idx].overlap(ii))
                                        for ii in self.KC_data[cval_key]['Wavefunctions'][left_idx:right_idx]]
                    # Determine the index associated with the maximal overlap
                    subset_idx = np.argmax(_dw_overlaps)
                    kerr_idx = dw_idx + subset_idx - state_span
                    tmp_overlap_list.append(_dw_overlaps[subset_idx])
                    c_idx.append(kerr_idx)
            corrected_idx.append(c_idx)
            corrected_overlaps.append(tmp_overlap_list)
            self.corrected_indices  = corrected_idx
            self.corrected_overlaps = corrected_overlaps
        return


    def initialize_plotting_data(self, xplot_min=-4, xplot_max=4, N_xplot=None):
        '''
        Function to initialize np.array of x-values used to calculate
        the potential V(x) for plotting purposes.

        Arguments:
            - `xplot_min` (int/float): minimum value of array (Default: -4)
            - `xplot_max` (int/float): maximum value of array (Default: 4)
            - `N_xplot` (int): Number of points in array spanning from xplot_min to xplot_max
        '''
        self.N_xplot          = self.N_basis if N_xplot is None else N_xplot
        self.xplot            = np.linspace(xplot_min, xplot_max, self.N_xplot)
        self.V_chem           = self._get_V_chem(self.xplot)
        return


    def initialize_cvalues(self, c_min=0.005, c_max=0.6, c_step=0.0125, do_phi_ZPS=False):
        '''
        Function to create an array of c-values to calculate Kerr-cat parameters with.

        Arguments:
            - `c_min` (float): minimum of array (Default: 0.0)
            - `c_max` (float): maximum of array (Default: 0.6)
            - `c_step` (float): step_size used in the array creation
            - `do_phi_ZPS` (Boolean): flag to determine whether we do the calculation in terms of c of $\phi_{ZPS}$
        '''
        _cvalz    = np.arange(c_min, c_max, step=c_step)
        if do_phi_ZPS:
            self.do_phi_ZPS = True
            self.cvalz = _cvalz / np.sqrt(2)
        else:
            self.do_phi_ZPS = False
            self.cvalz = _cvalz

        # Calculate KC Parameters
        self._calculate_device_params()
        self._calculate_scaled_params()
        return



    def plot_energies_and_parameters(self, n_plotted=12, plot_interval=3, save_figs=False, plot_title=None, do_eps1=True):


        if plot_title:
            title_text = str(plot_title)
        else:
            title_text = ''
        ckeys = list(self.KC_data.keys())

        xmin = min(self.cvalz)
        xmax = max(self.cvalz)
        # Begin Plotting Routine
        fig, ax = plt.subplots(2, 1,
                               figsize=(7.2, 7.5),
                               sharex=True, dpi=100)

        for ii in range(0, n_plotted):
            # Energy Difference Plot
            ax[0].plot(self.cvalz, self.energy_differences.T[ii], markevery=plot_interval)
            ax[0].hlines(1.5e-3, 0, 0.6, color='k', linestyle='dashed')
            ax[0].set_yticks([0, 1e-3, 2e-3, 3e-3, 4e-3], labels=[0, 1, 2, 3, 4])
            ax[0].set_ylabel(r'$\left| E_{\mathrm{KC}} - E_{\mathrm{DW}} \right|$ (m$E_{h}$)')
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(0, 0.003)


        ax[1].plot(self.cvalz, np.abs(self.scaled_device_params['Delta/K']), label=r'$\left|\Delta / K \right|$', color='orange')
        ax[1].plot(self.cvalz, np.abs(self.scaled_device_params['eps2/K']), label=r'$\left|\epsilon_{2} / K \right|$', color='k')
        if do_eps1:
            ax[1].plot(self.cvalz, np.abs(self.scaled_device_params['eps1/K']), label=r'$\left|\epsilon_{1} / K \right|$', color='#56B4E9')


        l1 = ax[1].hlines(20,
                            0., max(self.cvalz),
                            linestyle='--',
                            color='orange')
        l2 = ax[1].hlines(40,
                            0., max(self.cvalz),
                            linestyle='--',
                            color='k')
        if do_eps1:
            l3 = ax[1].hlines(100,
                               0., max(self.cvalz),
                               linestyle='--',
                               color='#56B4E9')
        if self.do_phi_ZPS:
            xlabel = r'$\phi_{\mathrm{zps}}$'
        else:
            xlabel = r'$c\ (a_{0})$'
        ax[1].set_xlabel(xlabel)
        ax[1].set_yscale('log')
        ax[1].set_yticks(ticks=[0.1, 1, 10, 100, 1000, 10000][::2])
        ax[1].set_ylim(0.01, 10001)
        ax[1].yaxis.set_minor_locator(mticker.LogLocator(numticks=3, subs=[2]))
        ax[1].set_xticks(ticks=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        plt.legend(fontsize=18, loc='lower left')
        plt.suptitle(title_text, y=0.95, fontsize=20)
        plt.tight_layout()
        if save_figs:
            os.makedirs(self.savedir, exist_ok=True)
            plt.savefig('{}/{}-Overlaps-Params_Eoffset.pdf'.format(self.savedir, self.system_name), dpi=600)
            plt.savefig('{}/{}-Overlaps-Params_Eoffset.png'.format(self.savedir, self.system_name), dpi=600)
        plt.show()
        return

    def _get_hermite_polynomial(self, k, x, x0, p0, alpha):
        '''
        Obtain the analytical solutions of the harmonic oscillator by expansion in the basis of Hermite polynomials.
        This function takes the following parameters:
        - `k` (int): order of the Hermite Polynomial
        - x (np.array): position values over which the Hermite Polynomial is calculated
        - x0 (float): initial displacement of the oscillator
        - p0 (float): initial momentum of the oscillator
        - alpha (float): width of the Gaussian
        '''
        herman = eval_hermite(k, x)*np.exp(-alpha/2*((x - x0)**2)+1j*p0*(x - x0))*((alpha/np.pi)**(0.25))/(2**(k/2))/np.sqrt(factorial(k))
        return herman


    def _expand_in_hermite_basis(self, input_arr, xvalz):
        N = len(input_arr)
        # Expand in basis of Hermite polynomials, using the expansion coefficients in input_arr
        psi_xx = np.zeros((N), dtype=complex)
        for i in range(N):
            psi_xx += input_arr[i]*self._get_hermite_polynomial(i, xvalz, 0.0, 0, 1.0)
        return psi_xx


    def plot_wavefunctions_on_potential(self, cval2plot=None, wfn_scale=0.005, n_plotted=12,
                                        plot_title=None, save_figs=False, max_ytick=0.03,
                                        min_ytick=0.000, n_ytick=5):
        '''
            This is a function to plot the wavefunctions of the chemical system Hamiltonian alongside
            the wavefunctions of the Kerr-Cat Hamiltonian at a given value of c.

            Arguments:
                - `cval2plot` (float): The value of c at which the Kerr-Cat Hamiltonian wavefunctions were computed.
                - `wfn_scale` (float): An arbitrary scaling factor to ensure that the wavefunctions look nice when plotted on the P.E. surface.
                - `n_plotted` (int): The number of wavefunctions to plot.
                - `plot_title` (str): The title for the plot.
        '''
        if type(cval2plot) == type(None):
            print('You must provide a value of c to plot the wavefunctions at!')
            return
        else:
            pass

        ckeys = np.asarray(list(self.KC_data.keys()))

        if cval2plot not in ckeys:
            print('You must provide a value of c within the existing cval array!')
            return

        if type(self.xplot) == type(None):
            # If no plotting arrays, we will generate them using default params
            self.initialize_plotting_data()
        else:
            pass

        if plot_title:
            title_text = str(plot_title)
        else:
            title_text = ''

        # Use a y-offset such that the minimum of the P.E. surface is at 0.
        y_offset = abs(min(self.V_chem))
        scaled_V = self.V_chem + y_offset
        scaled_ens_dw = self.chemical_data['Energies'] + y_offset
        c_idx = np.argwhere(np.isclose(ckeys, 0.1)==True)
        index_mapping = self.KC_data[cval2plot]['index_mapping']
        # Get the KerrCat Wavefunctions
        wfns_kc = self.KC_data[cval2plot]['Wavefunctions']
        wfns_dw = self.chemical_data['Wavefunctions']

        # Determine some useful quantities for plot limiting
        x_min  = min(self.xplot)
        x_max  = max(self.xplot)

        y_max  = np.ceil(max(1000*scaled_ens_dw[:n_plotted]))/1000
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.plot(self.xplot, scaled_V, color='k')
        for ii in range(n_plotted):
            fixed_index = index_mapping[ii]
            # Expand in Hermite basis, then normalize
            gs_dw = self._expand_in_hermite_basis(wfns_dw[ii].full().flatten(), self.xplot)
            gs_dw /= np.linalg.norm(gs_dw)
            gs_kc = self._expand_in_hermite_basis(wfns_kc[ii].full().flatten(), self.xplot)
            gs_kc /= np.linalg.norm(gs_kc)

            overlap = gs_dw.conj().dot(gs_kc).real
            if overlap < 0:
                print(overlap)
                gs_kc = -1.*gs_kc
                overlap = gs_dw.conj().dot(gs_kc).real
            print('Overlap for State {}:\t{}'.format(ii, np.round(overlap, 6)))
            ax.plot(self.xplot, wfn_scale*gs_dw.real+scaled_ens_dw[fixed_index], color='crimson')
            ax.plot(self.xplot, wfn_scale*gs_kc.real+scaled_ens_dw[fixed_index], color='dodgerblue', linestyle='dashed')

        ax.set_xlabel(r'$x$ ($a_0$)')
        ax.set_ylabel(r'Energy (m$E_{\mathrm{h}}$)')
        # Set ytickz dynamically, as well as x-limits and y-limits (check these!!!)
        ytickz = np.linspace(min_ytick, max_ytick, n_ytick)
        scaled_ytickz = (1000*ytickz).astype(int)
        y_maxrange = max(max_ytick, y_max)
        ax.set_yticks(ytickz, scaled_ytickz)
        ax.set_ylim(-0.0001, y_maxrange) # Adenine-Thymine
        ax.set_xlim(x_min, x_max)
        ax.set_title(plot_title)
        plt.tight_layout()

        if save_figs:
            os.makedirs(self.savedir, exist_ok=True)
            plt.savefig('{}/{}-Wavefunctions_on_Potential.pdf'.format(self.savedir, self.system_name), dpi=600)
            plt.savefig('{}/{}-Wavefunctions_on_Potential.png'.format(self.savedir, self.system_name), dpi=600)
        else:
            pass
        plt.show()
        return

    # TODO: Save/reload the data
