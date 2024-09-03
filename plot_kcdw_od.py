'''
    This module can post-process the arrays generated by
    kc_dw_open_dynamics.py and generate analysis plots.
'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use('plot_style.txt')
plt.rcParams['figure.constrained_layout.use'] = True
plt.figure(figsize=(14.4, 7.5))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'


def perform_fit(x, y, f_model, upper_bound=None):
    '''
        Perform a curve fitting using a specified model function.

        Arguments:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data to fit.
        f_model (callable): Model function to fit to the data. Must accept x
                            as the first argument and parameters to fit as
                            subsequent arguments.
        upper_bound (array-like, optional): Upper bounds for the parameters.
                                            If None, no upper bounds are
                                            applied.

        Returns:
        tuple: A tuple containing:
            fit_param (array): Optimized parameters.
            fit_error (array): Error matrix for optimized parameters.
    '''
    ndim1, ndim2 = y.shape
    fit_param = np.zeros(ndim1)
    fit_error = np.zeros(ndim1)
    if upper_bound is not None:
        idx = np.argwhere(x < upper_bound)
        x = x[:np.max(idx)]
        y = np.array(y)[:, :np.max(idx)]
    for i in range(ndim1):
        popt, pcov = curve_fit(f=f_model, xdata=x, ydata=y[i, :])
        perr = np.sqrt(np.diag(pcov))
        fit_param[i] = popt[0]
        fit_error[i] = perr[0]
    return fit_param, fit_error


def f_model1(x, T, C):
    ''' Decreasing exponential with y-shift '''
    return (1.-C)*np.exp(-x/T)+C


def f_model2(x, T, C):
    ''' Increasing exponential with y-shift '''
    return C*(1.-np.exp(-x/T))


def get_data_plot_traj(system, ham_type='KC', NbF=100, NbE=50,
                       gamma=0.1, nbar=0.1,
                       cmin=0.1, cmax=1.0, cinc=0.1,
                       tmax=1000.0, tnum=1000, folder='./',
                       LDWcvar='False', obs=2):
    '''
        Generate data to plot trajectory and rates for a specified physical
        system.

        Arguments:
        system (str): Identifier the physical system to simulate.
        ham_type (str, optional): Type of Hamiltonian used in the system
                                  simulation ('KC' by default).
        NbF (int, optional): Number of Fock basis used (default is 100).
        NbE (int, optional): Number of Eigenstate basis used (default is 50).
        gamma (float, optional): Bath interaction parameter for the simulation
                                 (default is 0.1).
        nbar (float, optional): Bath thermal coefficient (default is 0.1).
        cmin (float, optional): Minimum value for the c parameter
                                (default is 0.1).
        cmax (float, optional): Maximum value for the c parameter
                                (default is 1.0).
        cinc (float, optional): Increment step for the c parameter
                                (default is 0.1).
        tmax (float, optional): Maximum time for the simulation
                                (default is 1000.0).
        tnum (int, optional): Number of time points for the simulation 
                              (default is 1000).
        folder (str, optional): Directory to save the generated plots
                                (default is './').
        LDWcvar (str, optional): Indicator of whether to LDW variable was used
                                 in data generation as a filename tag.
                                 ('False' by default).
        obs (int, optional): Observable index to use for plotting
                             (default is 2).

        Returns:
        data_dict (dict): Dictionary contained keyed data entries
            time (array-like): Simulation time array
            cvalz (array-like): Lengthscale equivalence variable c
            obs_data (array-like): Observable data (trace of correlation)
            traj_labels (array-like): Trajectory labels
            relax_timescale (array-like): Relaxation timescales
            timescale_error (array-like): Relaxation timescales errors
    '''

    filename = (f'{system}_{ham_type}_LDWcvar{LDWcvar}_dynamics_'
                f'cmin{cmin:1.2f}cmax{cmax:1.2f}_cinc{cinc:1.3f}_'
                f'tmax{tmax:.1f}_tnum{tnum:.1f}_'
                f'gamma_{gamma:1.4e}_nbar_{nbar:1.4e}_'
                f'NbE_{NbE}_NbF_{NbF}_')

    times = np.load(folder+filename+'times.npy')
    cvalz = np.load(folder+filename+'cvalz.npy')
    traj_labels = np.round(cvalz, decimals=3)

    if obs == 1:
        fit_model = f_model1
    elif obs == 2:
        fit_model = f_model2
    else:
        raise Exception('Pick either 1 or 2 for rho0rhot or heaviside obs')

    obs_array = np.load(folder+filename+f'obs{obs}.npy')
    fit_param, fit_error = perform_fit(times, obs_array, fit_model)

    data_dict = {
            'time': times,
            'cvalz': cvalz,
            'obs_data': obs_array,
            'traj_labels': traj_labels,
            'relax_timescale': fit_param.reshape(-1, 1),
            'timescale_error': fit_error.reshape(-1, 1),
            }
    return data_dict


def plot_traj(times, obs, ax=None, data_label=None,
              xlabel='Time',
              ylabel='Population',
              t_xlims=(0., 1000.0, 1.0),
              t_ylims=(0.0, 1.0),
              phi_zps=False, style='-'):
    '''
        Plot trajectory data against time.

        Arguments:
        times (array-like): Array of time points.
        obs (array-like): Observable data corresponding to each time point.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on.
                                             If None, a new figure and axes are
                                             created.
        data_label (str, optional): Label for the data in the plot legend.
        xlabel (str, optional): Label for the x-axis (default is 'Time').
        ylabel (str, optional): Label for the y-axis (default is 'Population').
        t_xlims (tuple, optional): Limits for the x-axis in the form (min, max, 
                                    step) (default is (0., 1000.0, 1.0)).
        t_ylims (tuple, optional): Limits for the y-axis in the form (min, max) 
                                    (default is (0.0, 1.0)).
        phi_zps (bool, optional): Whether to use \phi_{ZPS} as x-variable
                                  (default is False).
        style (str, optional): Line style for the plot (default is '-').

        Returns:
        ax (matplotlib.axes.Axes): Matplotlib axes with figure data
    '''

    if ax is None:
        plt.close()
        ax = plt.gca()
    if data_label is None:
        data_label = [None] * len(obs)
    data_label_prefix = 'c = '
    if phi_zps:
        data_label /= np.sqrt(2)
        data_label_prefix = r'\varphi _{\mathrm{ZPS}} = '
    data_label = np.round(data_label, decimals=2)
    data_label = [data_label_prefix+str(i) for i in data_label]
    xticks_pos = np.round(np.arange(t_xlims[0],
                                    t_xlims[1]+t_xlims[2],
                                    t_xlims[1]/10),
                          3)

    for ii in range(obs.shape[0]):
        ax.plot(times, obs[ii],
                label=data_label[ii],
                linestyle=style[0],
                marker=style[1], markersize=5, markevery=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if t_xlims is not None:
        ax.set_xlim(*t_xlims[:2])
    if xticks_pos is not None:
        ax.set_xticks(xticks_pos)
    if t_ylims is not None:
        ax.set_ylim(*t_ylims[:2])
        y_ticks = np.round(np.linspace(t_ylims[0], t_ylims[1], 5),
                           3)
        ax.set_yticks(y_ticks)
    ax.legend(loc='center left',
              bbox_to_anchor=(1, 0.5))
    return ax


def plot_rates(cvalz, timescales, ax=None,
               data_label=None,
               xlabel=r'$\varphi _{\mathrm{ZPS}}$',
               ylabel=r'$T_X$',
               t_xlims=(0.1, 1.0, 0.1),
               t_ylims=(0.0, 1.05),
               phi_zps=False, style='-', yscale='linear',
               plt_Txerror=False, Tx_error=None):
    '''
        Plot rates as a function of control parameter.

        Arguments:
        cvalz (array-like): Array of c lenghtscale parameter values.
        timescales (array-like): Corresponding timescales for each c
                                  parameter value.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on.
                                             If None, a new figure and axes are
                                             created.
        data_label (str, optional): Label for the data in the plot legend.
        xlabel (str, optional): Label for the x-axis (default is r'$\varphi 
                                 _{\mathrm{ZPS}}$').
        ylabel (str, optional): Label for the y-axis (default is r'$T_X$').
        t_xlims (tuple, optional): Limits for the x-axis in the form (min, max, 
                                    step) (default is (0.1, 1.0, 0.1)).
        t_ylims (tuple, optional): Limits for the y-axis in the form (min, max) 
                                    (default is (0.0, 1.05)).
        phi_zps (bool, optional): Whether to use \phi_{ZPS} as x-variable
                                  (default is False).
        style (str, optional): Line style for the plot (default is '-').

        Returns:
        ax (matplotlib.axes.Axes): Matplotlib axes with figure data
    '''

    if ax is None:
        plt.close()
        ax = plt.gca()
    if phi_zps:
        t_xlims = tuple(np.round(np.ceil(tx*10)/10, decimals=1)
                        for tx in t_xlims)
        xticks_pos = np.arange(t_xlims[0], t_xlims[1]+t_xlims[2],
                               t_xlims[2])
        xlabel = r'$\varphi _{\mathrm{ZPS}}$'
    else:
        xticks_pos = np.arange(np.ceil(t_xlims[0]*10)/10,
                               np.ceil((t_xlims[1]+t_xlims[2])*10)/10,
                               np.round((t_xlims[1]-t_xlims[0])/(10), 1))
        xlabel = '$c$'

    ax.plot(cvalz, timescales.reshape(-1, 1), linestyle=style[0],
            label=data_label, marker=style[1], markersize=5, markevery=2)
    if plt_Txerror:
        lower_bound = np.ndarray.flatten(timescales - Tx_error)
        upper_bound = np.ndarray.flatten(timescales + Tx_error)
        ax.fill_between(cvalz, lower_bound, upper_bound, alpha=.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*t_xlims[:2])
    if xticks_pos is not None:
        ax.set_xticks(xticks_pos)

    if yscale == 'log':
        ax.set_yscale('log')
        timescales = np.round(timescales, decimals=5)
        t_val = max(t_ylims[1], np.max(timescales))*1.11
        b_val = max(t_ylims[0], np.min(timescales))*0.5
        ax.set_ylim(b_val, t_val)
        # setting yticks
        min_y_order = np.floor(np.log10(b_val))
        max_y_order = np.ceil(np.log10(t_val))
        n_major_ticks = max_y_order - min_y_order+1
        y_major = mpl.ticker.LogLocator(base=10.0,
                                        numticks=n_major_ticks)
        ax.yaxis.set_major_locator(y_major)
        y_minor = mpl.ticker.LogLocator(base=10.0,
                                        subs=np.arange(1.0,
                                                       10.0) * 0.1,
                                        numticks=10)
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        t_val = min(t_ylims[1], np.max(timescales))*1.1
        b_val = max(t_ylims[0], np.min(timescales))*0.95

        ax.set_ylim(b_val, t_val)
        t_val = np.round(t_val, decimals=-1)
        b_val = np.round(b_val, decimals=-1)
        y_ticks = np.round(np.linspace(b_val, t_val, 5),
                           0)
        ax.set_yticks(y_ticks)
    ax.legend(loc='center left',
              bbox_to_anchor=(1, 0.5))
    return ax


def get_traj_plot(xs, ys, obs=2,
                  t_clims=(0.1, 1.0, 0.1),
                  tx_lims=(0.0, 10e2, 1.0),
                  t_lims=(0.0, 1000.0, 1.0),
                  pop_lims=(0.0, 1.0, 0.1),
                  data_label=(None,),
                  plt_traj=False, plt_Tx=False, phi_zps=False,
                  plt_KC=False, plt_DW=False, LDWcvar=True,
                  tx_yscale='linear', y_units=r'$\hbar / E_h$',
                  outdir='./', savename='test', system='',
                  plt_Txerror=False, Tx_error=(None,)):
    '''
        Generate and save plots of trajectories and rates given set of data.

        Arguments:
        xs (array-like): Array of x values for plotting.
        ys (array-like): Array of y values for plotting.
        obs (int, optional): Observable index to use for plotting (default=2).
        t_clims (tuple, optional): Limits for the c parameter in the form
                                   (min, max, step) (default=(0.1, 1.0, 0.1)).
        tx_lims (tuple, optional): Limits for the timescale plot in the form 
                                   (min, max, step) (default=(0.0, 10e2, 1.0)).
        t_lims (tuple, optional): Limits for the time plot in the form (min,
                                  max, step) (default=(0.0, 1000.0, 1.0)).
        pop_lims (tuple, optional): Limits for the population plot in the form 
                                    (min, max, step) (default=(0.0, 1.0, 0.1)).
        data_label (tuple, optional): Labels for the data in the plot legend 
                                       (default is (None,)).
        plt_traj (bool, optional): Whether to plot the trajectory (default is
                                   False).
        plt_Tx (bool, optional): Whether to plot timescales (default is False).
        phi_zps (bool, optional): Whether to use \phi_{ZPS} as x-variable
                                  (default is False).
        plt_KC (bool, optional): Whether to plot KC data (default is False).
        plt_DW (bool, optional): Whether to plot DW data (default is False).
        LDWcvar (bool, optional): Indicator of whether to LDW variable was used
                                 in data generation as a filename tag.
                                 ('False' by default).
        tx_yscale (str, optional): Y-axis scale for timescales ('linear' or
                                   'log', default is 'linear').
        y_units (str, optional): Units for the y-axis (default is
                                 r'$\hbar/E_h$').
        outdir (str, optional): Directory to save the plots
                                (default is './').
        savename (str, optional): Base name for saved files
                                  (default is 'test').
        system (str, optional): Identifier or description of the physical
                                system (default is '').
        plt_Txerror (bool, optional): Whether to plot error bars for timescales 
                                       (default is False).
        Tx_error (tuple, optional): Error values for timescales. Required if 
                                     plt_Txerror is True (default is (None,)).

        Returns:
        None
    '''

    nrows = plt_traj + plt_Tx
    ncolumns = 1
    plt_row_idx = 0
    plt_col_idx = 0

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = fig.add_gridspec(nrows, ncolumns, hspace=0.0001)
    axs = gs.subplots()
    axs = np.array(axs)
    axs = axs.reshape(nrows, ncolumns)
    fig.suptitle(system, fontsize=16)

    if obs == 1:
        ylabel = r'Correlation $(\rho_0, \rho_t)$'
    elif obs == 2:
        ylabel = r'Product Population, $P_P(t)$'
    else:
        raise Exception('Pick either 1 or 2 for rho0rhot or heaviside obs')

    # Plotting the trajectories as function of time
    if plt_traj:
        style = [('-', ''), ('', 'd')]
        ax_traj = axs[plt_row_idx, plt_col_idx]
        for ii in range(plt_KC+plt_DW):
            ax_traj.set_prop_cycle(None)
            ax_traj = plot_traj(xs[ii], ys[ii],
                                ax_traj,
                                data_label[ii],
                                xlabel='Time',
                                ylabel=ylabel,
                                t_xlims=t_lims,
                                t_ylims=pop_lims,
                                phi_zps=phi_zps, style=style[ii])
        plt_row_idx += 1
    if plt_Tx:
        style = [('-', ''), ('', 'd')]
        ax_tx = axs[plt_row_idx, plt_col_idx]
        if plt_DW and plt_KC:
            data_label = ['DW', 'KC']
        elif plt_DW:
            data_label = ['DW']
        elif plt_KC:
            data_label = ['KC']
        else:
            data_label = ['']
        for ii in range((plt_KC+plt_DW)*plt_traj, (plt_KC+plt_DW)*nrows):
            ax_tx = plot_rates(xs[ii], ys[ii],
                               ax_tx,
                               data_label=data_label[ii % 2],
                               xlabel='c',
                               ylabel=(r'$T_X$ ('+y_units + ')'),
                               t_xlims=t_clims,
                               t_ylims=tx_lims,
                               phi_zps=phi_zps, style=style[ii % 2],
                               yscale=tx_yscale, plt_Txerror=plt_Txerror,
                               Tx_error=Tx_error[ii % 2])
        plt_row_idx += 1
    if outdir and savename:
        fname = (f'{outdir}/{system}_{savename}_pltTraj_{plt_traj}_'
                 f'pltTx_{plt_Tx}_LDWcvar{LDWcvar}_obs{obs}')
        plt.savefig(fname + '.png', dpi=600)
        plt.savefig(fname + '.pdf', dpi=600)
        return None
    else:
        plt.show()
        return None


if __name__ == '__main__':
    nbasis_E = 50
    nbasis_F = 100
    observable = 2
    system_list = ['ciscis_malonaldehyde',
                   'cistrans_malonaldehyde',
                   'adenine_thymine',
                   'guanine_cytosine']

    folder_list = [f'gamma0.1_nbar0.1_NbF_{nbasis_F}']
    diss_list = [(0.1, 0.1)]
    t_lims_list = [(0.0, 200.0, 1.0)]
    plot_params_dict = {
            folder_list[0]: [diss_list[0], t_lims_list[0]],
            }

    for idx, folder in enumerate(folder_list):
        gamma, nbar = plot_params_dict[folder][0]
        t_lims_spec = plot_params_dict[folder][1]
        print(f'gamma={gamma}, nbar={nbar}')
        for entry in system_list:
            data_KC = get_data_plot_traj(entry, ham_type='KC',
                                         NbF=nbasis_F, NbE=nbasis_E,
                                         gamma=gamma, nbar=nbar,
                                         cmin=0.01, cmax=1.0, cinc=0.01,
                                         tmax=10, tnum=1000,
                                         folder=folder+'/',
                                         LDWcvar='True', obs=observable)
            data_DW = get_data_plot_traj(entry, ham_type='DW',
                                         NbF=nbasis_F, NbE=nbasis_E,
                                         gamma=gamma, nbar=nbar,
                                         cmin=0.01, cmax=1.0, cinc=0.01,
                                         tmax=10, tnum=1000,
                                         folder=folder+'/',
                                         LDWcvar='True', obs=observable)
            print(f'{entry} rates (KC) & '
                  f'${float(data_KC["relax_timescale"][9]):1.2f}\\pm'
                  f'{float(data_KC["timescale_error"][9]):1.2f}$ &')
            print(f'{entry} rates (DW) & '
                  f'${float(data_DW["relax_timescale"][9]):1.2f}\\pm'
                  f'{float(data_DW["timescale_error"][9]):1.2f}$ &')

            # choosing which segments of the trajectory data to plot
            npts_kc_traj = int((plot_params_dict[folder][1][1]
                                - plot_params_dict[folder][1][0])
                               / (20*plot_params_dict[folder][1][2]))
            dw_traj_data = np.vstack((data_DW['obs_data'][4],
                                      data_DW['obs_data'][19::20]))
            kc_traj_data = np.vstack((data_KC['obs_data'][4,
                                                          ::npts_kc_traj],
                                      data_KC['obs_data'][19::20,
                                                          ::npts_kc_traj]))
            # choosing the corresponding labels
            kc_traj_labels = np.hstack((data_KC['cvalz'][4],
                                        data_KC['cvalz'][19::20]))
            dw_traj_labels = np.hstack((data_DW['cvalz'][4],
                                        data_DW['cvalz'][19::20]))

            get_traj_plot([data_DW['time'], data_KC['time'][::npts_kc_traj],
                           data_DW['cvalz'][4:], data_KC['cvalz'][4:]],
                          [dw_traj_data, kc_traj_data,
                           data_DW['relax_timescale'][4:].T,
                           data_KC['relax_timescale'][4:].T],
                          obs=observable,
                          t_clims=(0.05, 1.00, 0.1),
                          tx_lims=(0.0, 10e2, 1.0),
                          t_lims=t_lims_spec,
                          pop_lims=(0.0, 1.0, 0.1),
                          data_label=[dw_traj_labels, kc_traj_labels],
                          plt_traj=True,
                          plt_Tx=True,
                          phi_zps=False,
                          plt_KC=True, plt_DW=True, LDWcvar=True,
                          tx_yscale='log',
                          outdir='./', savename=folder, system=entry,
                          plt_Txerror=True,
                          Tx_error=[data_DW['timescale_error'][4:].T,
                                    data_KC['timescale_error'][4:].T])
