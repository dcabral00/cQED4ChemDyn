# start system: ciscis_malonaldehyde  &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=0.1 -c_max=1 -c_inc=0.1 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=ciscis_malonaldehyde -k4=0.000714286 -k2=0.004 -k1=-0.0 -hamiltonian=KC -nbasis_fock=100 -nbasis_eigen=50 &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=1 -c_max=1 -c_inc=0.01 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=ciscis_malonaldehyde -k4=0.000714286 -k2=0.004 -k1=-0.0 -hamiltonian=DW -nbasis_fock=100 -nbasis_eigen=50 &
# end system &
############################################################ &
# start system: cistrans_malonaldehyde &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=0.1 -c_max=1 -c_inc=0.1 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=cistrans_malonaldehyde -k4=9.374e-05 -k2=0.00299 -k1=-0.00359373 -hamiltonian=KC -nbasis_fock=100 -nbasis_eigen=50 &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=1 -c_max=1 -c_inc=0.01 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=cistrans_malonaldehyde -k4=9.374e-05 -k2=0.00299 -k1=-0.00359373 -hamiltonian=DW -nbasis_fock=100 -nbasis_eigen=50 &
# end system &
############################################################ &
# start system: adenine_thymine &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=0.1 -c_max=1 -c_inc=0.1 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=adenine_thymine -k4=6.435325865645717e-05 -k2=0.0032961632991796736 -k1=-0.005834209039548023 -hamiltonian=KC -nbasis_fock=100 -nbasis_eigen=50 &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=1 -c_max=1 -c_inc=0.01 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=adenine_thymine -k4=6.435325865645717e-05 -k2=0.0032961632991796736 -k1=-0.005834209039548023 -hamiltonian=DW -nbasis_fock=100 -nbasis_eigen=50 &
# end system &
############################################################ &
# start system: guanine_cytosine &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=0.1 -c_max=1 -c_inc=0.1 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=guanine_cytosine -k4=0.00076778 -k2=0.00688838 -k1=-0.00452306 -hamiltonian=KC -nbasis_fock=100 -nbasis_eigen=50  &
python kc_dw_open_dynamics.py -basis_type=E -mass=1836.0 -c_min=1 -c_max=1 -c_inc=0.01 -gamma=0.1 -nbar=0.1 -t_min=0.0 -t_max=1000.0 -t_num=1000 -grid_pts=5000 -grid_lims=10 -x_cutoff=0.0 -cutoff_value=0.5 -sigmoidal_tail=0.5 -filter_type=S -LDWcvar=False -gammaK=False -system_name=guanine_cytosine -k4=0.00076778 -k2=0.00688838 -k1=-0.00452306 -hamiltonian=DW -nbasis_fock=100 -nbasis_eigen=50 &
# end system &
############################################################ &
wait
