# cQED4ChemDyn

This repository provides an implementation that connects chemical kinetics of elementary reactivity models with the framework of the Kerr-Cat circuit quantum electrodynamics (cQED), using the Hamiltonian describing the physics of the hardware and a Lindbladian open quantum dynamics formalism for the time-evolution of the system. For more information, check the existing citation for the publication at the bottom of this Readme; any usage of this code should cite the preprint (and publications) once available.

## Getting Started

### Installation

To set up the environment for executing the code in this repository, you have two options: using `conda` or installing packages manually.

#### Using Conda

1. **Clone the Repository**:
   Choose a target folder location and clone the repository:
   ```bash
   git clone git@github.com:dcabral00/cQED4ChemDyn.git
   cd cQED4ChemDyn
   ```

2. **Create and Activate the Conda Environment**:
   Set up the environment using the provided `.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate cQED4ChemDyn_env
   ```

   This will install all necessary dependencies for running the scripts in this repository.

#### Manual Installation

Alternatively, you can manually install the required packages listed in the `requirements.txt` file:

1. **Clone the Repository**:
   As above, clone the repository and navigate to the project directory:
   ```bash
   git clone git@github.com:dcabral00/cQED4ChemDyn.git
   cd cQED4ChemDyn
   ```

2. **Install Dependencies**:
   Use `pip` to install the packages:
   ```bash
   pip install -r requirements.txt
   ```

   This command will install all the dependencies specified in the `requirements.txt` file.

### Running cQED4ChemDyn

Once the environment is set up, the `kc_dw_open_dynamics.py` can simulate the open quantum dynamics of chemical double-well systems and output data suitable for calculation of kinetic observables. It contains a command line interface supporting the necessary arguments,
   ```bash
   usage: kc_dw_open_dynamics.py [-h] [-basis_type {E,F}] [-nbasis_fock NBASIS_FOCK]
                                 [-nbasis_eigen NBASIS_EIGEN] [-mass MASS] [-k4 K4] [-k2 K2] [-k1 K1]
                                 [-c_min C_MIN] [-c_max C_MAX] [-c_inc C_INC] [-gamma GAMMA] [-nbar NBAR]
                                 [-t_min T_MIN] [-t_max T_MAX] [-t_num T_NUM] [-grid_pts GRID_PTS]
                                 [-grid_lims GRID_LIMS] [-x_cutoff X_CUTOFF] [-cutoff_value CUTOFF_VALUE]
                                 [-sigmoidal_tail SIGMOIDAL_TAIL] [-filter_type FILTER_TYPE]
                                 [-hamiltonian {KC,DW}] [-system_name SYSTEM_NAME] [-LDWcvar LDWCVAR]
                                 [-gammaK GAMMAK] [-time_ratio_gn TIME_RATIO_GN]
   
   optional arguments:
     -h, --help            show this help message and exit
     -basis_type {E,F}     Type of basis (Fock or Eigen) to use(default: E)
     -nbasis_fock NBASIS_FOCK
                           Number of Fock basis (default: 100)
     -nbasis_eigen NBASIS_EIGEN
                           Number of Eigen basis (default: 20)
     -mass MASS            Mass for kinetics problem(default: 1836. amu for proton)
     -k4 K4                Quartic position coefficient for chemical double well; Controls potential walls
                           (default: 1.)
     -k2 K2                Quadratic position coefficient for chemical double well; Controls barrier walls
                           (default: 1.)
     -k1 K1                Linear position coefficient for chemical double well; Controls inter-well
                           asymmetry (default: 1.)
     -c_min C_MIN          Minimun C mapping parameter (default: 0.4)
     -c_max C_MAX          Maximun C mapping parameter (default: 0.31); upperbound is included, (ie c_max
                           = c_max + c_step
     -c_inc C_INC          Step C mapping parameter (default: 0.1)
     -gamma GAMMA          Dissipation gamma parameter (default: 0.1)
     -nbar NBAR            Dissipation nbar parameter (default: 0.5)
     -t_min T_MIN          Minimun propagation time (default: 0.)
     -t_max T_MAX          Maximun propagation time (default: 100.)
     -t_num T_NUM          Number of propagation steps (default: 1000)
     -grid_pts GRID_PTS    Number of points for grid representation (default: 5000)
     -grid_lims GRID_LIMS  Maximum grid limit (default: 10)
     -x_cutoff X_CUTOFF    Cutoff x0 for LHS/RHS initial condition selection (default: 0.)
     -cutoff_value CUTOFF_VALUE
                           Cutoff probability for LHS/RHS init conditionselection (default: 0.5)
     -sigmoidal_tail SIGMOIDAL_TAIL
                           Decay of sigmoidal filter (default: 0.5)
     -filter_type FILTER_TYPE
                           Filter type for initial condition(default: None)
     -hamiltonian {KC,DW}  Hamiltonian type for dynamics(default: KC)
     -system_name SYSTEM_NAME
                           Name of the system being studied
     -LDWcvar LDWCVAR      Whether to enable variable c for DW
     -gammaK GAMMAK        Whether to enable gamma dependence on K
     -time_ratio_gn TIME_RATIO_GN
                           Whether to enable time array dependence on gamma and nth dissipation parameters
                           (ie smaller dissipation params require longer time for decay to be observed)

   ```

It can also be run with the default options for demonstration purposes or with real parameters as listed
in the associated publication.

Benchmark to reproduce the Kerr-Cat cQED device physics is also provided within the folder. These modules reproduce the dynamical data used in the supporting information (Lindblad diagonalization and propagation) for timescale fit with a scan over $\epsilon _1$ and $\epsilon _2$ for $\Delta = 0$.

### Jupyter Notebook Demo

For a quick and interactive way to explore the parameter equivalence between chemical systems and the Kerr-Cat
cQED platform, use the Jupyter Notebook through Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcabral00/cQED4ChemDyn/blob/main/KerrCat-Parameter-Plots-Github.ipynb)  or
   ```bash
   jupyter notebook KerrCat-Parameter-Plots-Github.ipynb
   ```

For a demonstration of the open quantum dynamics functionality, use [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcabral00/cQED4ChemDyn/blob/main/kcdw_od_demo.ipynb) or
   ```bash
   jupyter notebook kcdw_od_demo.ipynb
   ```

### Citation

Please cite the preprint of our work when using this code until the journal version becomes available.

### Disclaimer

While the code can be executed within a Jupyter notebook, we recommend using the provided Python modules for research applications and production-level development. The Jupyter notebook version, while convenient for initial exploration and experimentation, may lack the robustness and scalability of the provided Python modules. This repository is a snapshot of a broader collection of tools developed for continued research advancement, which forms the foundation of ongoing studies aimed at advancing the research presented. The complete library will be released at a later time.

### Contact

For questions, comments, or support, please contact:

Delmar G. A. Cabral (delmar.azevedocabral@yale.edu)
Brandon C. Allen (brandon.allen@yale.edu)
