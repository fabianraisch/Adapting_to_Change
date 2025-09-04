## A Comparison of Continuous and Transfer Learning for a Changing Built Environment

The paper can be found here: [ArXiv](https://arxiv.org/abs/2508.21615).

This repository provides code and data for comparing Transfer Learning (TL) and Continual Learning (CL) methods for modeling building thermal dynamics. We evaluate different strategies on long-term simulated data of single-family houses in Central Europe, including scenarios with retrofits and occupancy changes. Our proposed Seasonal Memory Learning (SML) method achieves the best accuracy, outperforming standard fine-tuning while keeping computation low.
In the following we will present the code for the experiments.

### Getting Started
To get started, clone this repository. 
```bash
git clone https://github.com/fabianraisch/Adapting_to_Change
```
Then, change the directory.
```bash
cd Adapting_to_Change
```

For the dependencies, you need to set up the conda environment. First, install [Miniconda](https://www.anaconda.com/download). Then run:
```bash
conda env create -f requirements.yml
```
to activate the environment
```bash
conda activate adaptive_learning
```

Alternatively, you can also use the requirements.txt as follows: 
```
pip install -r requirements.txt
```

Note: We use [CUDA](https://developer.nvidia.com/cuda-12-6-0-download-archive) version 12.6 here. 

After the setup, you can run the notebook adaptive_learning.ipynb
It may be the case, that ipykernel package must be installed. This can be done using
```bash
conda install -n cl ipykernel --update-deps --force-reinstall
```