# PAIRING
Perturbation identifier to induce desired cell states using generative deep learning
 
# Requirements
Please refer "requirements.txt".

For training from scratch, PAIRING requires at least 9GB of system memory and 5GB of GPU memory.

# Installation
```
git clone https://github.com/hanyh0807/PAIRING.git
cd PAIRING
conda create -n pairing python=3.9.16
conda activate pairing
pip install -r requirements.txt
```

PAIRING has been tested in Ubuntu18.04 and Python 3.9.16.

# Instructions

**Required data**

Please download data from the link in the file 'l1000_data/google-drive'

`example_datasets/` directory has example files.

**Usage**

`PAIRING_example_git.ipynb` has example codes to run PAIRING.
