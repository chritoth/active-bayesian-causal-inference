# Active Bayesian Causal Inference

This repository contains the implementation of the Active Bayesian Causal Inference framework for non-linear additive Gaussian noise models as described in our  [NeurIPS'22 ABCI paper](https://arxiv.org/abs/2206.02063). In summary, it provides functionality for generating groundtruth environments, running ABCI of course, and generating plots as in the paper. We also provide example notebooks to illustrate the basic usage of the code base and get you started quickly.  Feel free to reach out if you have questions about the paper or code!


## Getting Started

##### Python Environment Setup

These instructions should help you set up a suitable Python environment. We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for easily recreating the Python environment. You can install the latest version like so:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Once you have Miniconda installed,  create a virtual environment from the included environment description in `environment.yaml` like so:

```
conda env create -f environment.yaml
```
Finally, activate the conda environment via 
```
conda activate abci
```
and set your python path to the project root
```
export  PYTHONPATH="${PYTHONPATH}:/path/to/abci"
```

#### Running the Code

You can get started and play around with the example notebooks  `example_abci_categorical_gp.ipynb` or `example_abci_dibs_gp.ipynb` to get the gist of how to use the code base. For running larger examples you first need to generate benchmark environments (e.g. with `generate_benchmark_envs.ipynb`), run ABCI by starting either one of the scripts in `./src/scripts/`, and then plotting the results in `plot_benchmark_results.ipynb`. If you prefer to run ABCI from the command line, you can use the script `./src/scripts/run_single_env.py` (see e.g. `python run_single_env.py -h ` for usage instructions).

You can implement your own ground truth models by building upon the `Environment` base class in `./src/environments/environments.py`.  In principle it is also possible to run the Bayesian causal inference part of this implementation on a static dataset without the active learning part.

### Project Layout

The following gives you a brief overview on the organization and contents of this project. Note: in general it should be clear where to change the default paths in the scripts and notebooks, but if you don't want to waste any time just use the default project structure.

```
    │
    ├── README.md           <- This readme file.
    │
    ├── environment.yml     <- The Python environment spec for running the code in this project.
    │
    ├── data				<- Directory for generated ground truth models.
    │
    ├── figures             <- Output directory for generated figures.
    │
    ├── notebooks           <- Jupyter notebooks for running interactive experiments and analysis.
    |
    ├── results             <- Simulation results.
    |
    ├── src                 <- Contains the Python source code of this project.
    │   ├── __init__.py     <- Makes src a Python module
    │   ├── abci_base.py    <- ABCI base class.
    │   ├── abci_categorical_gp.py <- ABCI with categorical distribution over graphs & GP models.
    │   ├── abci_dibs_gp.py <- ABCI with DiBS approximate graph inference & GP models.
    │   ├── environments    <- Everything pertaining to ground truth environments.
    │   ├── experimental_design    <- Everything pertaining to experimental design (utility functions, optimization,...).
    │   ├── models          <- Everything pertaining to models (DiBS, GPs, ...)
    │   ├── scripts         <- Scripts for running experiments.
    │   ├── utils           <- Utils for plotting, metrics,...
    │
```
