## Install Conda
Assume python 3.13 is already installed.

The Conda installation follows the following https://www.anaconda.com/docs/getting-started/miniconda/install#to-download-an-older-version

Latest supported version od MacOS Intel is 25.7.0-2 (Aug 25, 2025).
NOTE: THIS IS THE LAST VERSION FOR MacOS Intel.
% mkdir -p ~/miniconda3
% curl https://repo.anaconda.com/miniconda/Miniconda3-py313_25.7.0-2-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh
% bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

Activate the conda environment:
% source ~/miniconda3/bin/activate

Initialize conda on all environments:
% conda init --all

Close and reopen the terminal shell. Check the conda version is 25.7:
% conda info

## Create Environment
It is recommended to create a conda environment for deep learning projects.
Edit a file named "environment.yml" in the project folder as shown below.
Note: ipykernel is required for Jupyter notbook to function inside the Visual Code IDE.

name: DeepLearn
channels:
  - defaults
dependencies:
  - python=3.13
  - numpy
  - pandas
  - matplotlib
  - tiktoken
  - tqdm
  - pytorch
  - psutil
  - ipykernel

% cd ~/Projects/LLM/xlang
% conda env create --name DeepLearn --file environment.yml
% conda activate DeepLearn

If the conda environment is already created, then update the environment:
% conda env update --name DataScience --file environment.yml

Edit the ~/.zshrc or ~/.bashrc files to set default conda environment shell.

Install PyTorch from an Apache distro
% conda install conda-forge::apache-beam-with-torch
