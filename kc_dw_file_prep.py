def generate_script_string(sysname, ham_type,
                           nBF,
                           nBE_tuple=(5, 10, 20, 30),
                           t_tuple=(0.0, 100.0, 1000),
                           script_prefix='kc_dw_open_dynamics.py',
                           t_total_ng_bool=False):
    # -----------------------------------------------------------------------
    #              Reading File Data Parameters
    # -----------------------------------------------------------------------

    script_params = {
        'basis_type': 'E',
        'mass': 1836.0,
        'c_min': 0.01,
        'c_max': 1.0,
        'c_inc': 0.01,
        'gamma': 0.1,
        'nbar': 0.1,
        't_min': 0.,
        't_max': 1000.,
        't_num': 1000,
        'grid_pts': 5000,
        'grid_lims': 10,
        'x_cutoff': 0.,
        'cutoff_value': 0.5,
        'sigmoidal_tail': 0.5,
        'filter_type': 'S',
        'LDWcvar': 'True',
        'gammaK': 'False',
    }

    command_list = []

    script_params['system_name'] = sysname
    script_params['k4'] = chem_systems[sysname][0]
    script_params['k2'] = chem_systems[sysname][1]
    script_params['k1'] = chem_systems[sysname][2]

    script_params['hamiltonian'] = ham_type
    script_params['nbasis_fock'] = nBF

    script_params['t_min'] = t_tuple[0]
    script_params['t_max'] = t_tuple[1]
    script_params['t_num'] = t_tuple[2]
    if t_total_ng_bool:
        script_params['t_max'] = (10 / script_params['nbar']
                                  / script_params['gamma'])

    for nBE in nBE_tuple:
        script_params['nbasis_eigen'] = nBE
        script_params_str = f'python {script_prefix}'
        # Append the current set of parameters to the script string
        script_params_str += f" {' '.join([f'-{key}={val}' for key, val in script_params.items()])}"
        command_list.append(script_params_str)
    return command_list


def write_to_file_sh(fname, header, content):
    file = open(fname + '.sh', 'w')
    file.close()
    with open(fname + '.sh', 'a') as file:
        for line in header.split('\n'):
            file.write(line)
            file.write('\n')
        file.write(content)
        file.write('\n')
    return None


if __name__ == '__main__':
    # Define arrays
    chem_systems = {
        'ciscis_malonaldehyde': (0.000714286, 0.004, -0.),
        'cistrans_malonaldehyde': (9.374e-5, 2.99e-3, -3.59373e-3),
        'adenine_thymine': (0.00140175, 0.01076113, -0.00515828),
        'guanine_cytosine': (0.00076778, 0.00688838, -0.00452306),
    }

    ham_type = ("KC", "DW")
    n_fock_basis = (50, 100, 150, 200)
    n_eigen_basis = (5, 10, 20, 30, 50,)
    t_tuple = (0.0, 1000.0, 1000)

    file_path_list = []
    for chem_sys in list(chem_systems.keys()):
        print(f'start system: {chem_sys}')
        for ham in ham_type:
            for nBF in n_fock_basis:
                print('module load miniconda; conda activate qiskit_env; ',
                      sep='', end='')
                command_list = generate_script_string(chem_sys, ham, nBF,
                                                      n_eigen_basis,
                                                      t_tuple)
                for j in command_list:
                    print(f'{j}; ', sep='', end='')
                print()
        print('end system')
        print('############################################################')
