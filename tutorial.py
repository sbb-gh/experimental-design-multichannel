"""
Copyright 2024 Stefano B. Blumberg and Paddy J. Slator

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


"""
Python-based tutorial for our upcoming conference and journal paper.

See README.md on replicating results in the paper.

`We encourage users to explore different options for data generation, preprocessing
and TADRED hyperparameters - check base_cfg.yaml.`


Overview for cells:
    - Choose data size splits 2
    - Generate data examples 3-A/B/C
    - Data format for TADRED 4
    - Option to pass data directly, or save to disk and load 5-A/B
    - Option to save output 6
    - TADRED hyperparameters 7,8,9 in order of importance.
"""


########## (1)
# Import modules, see requirements.txt for tadred requirements, set global seed

import numpy as np
from tadred import tadred_main, utils

np.random.seed(0)  # Random seed for entire script


########## (2)
# Data split sizes

n_train = 10**3  # No. training voxels, reduce for faster training speed
n_val = n_train // 10  # No. validations set voxels
n_test = n_train // 10  # No. test set voxels


########## (3-A)
# Create dummy, randomly generated (positive) data

Cbar = 220  # Num features of densely-sampled data
M = 12  # Number of target regressors
rand = np.random.lognormal  # Random genenerates positive
train_inp, train_tar = rand(size=(n_train, Cbar)), rand(size=(n_train, M))
val_inp, val_tar = rand(size=(n_val, Cbar)), rand(size=(n_val, M))
test_inp, test_tar = rand(size=(n_test, Cbar)), rand(size=(n_test, M))

"""
########## (3-B)
# Generate data using VERDICT model and scheme [link](https://pubmed.ncbi.nlm.nih.gov/25426656/)
# Requires python package dmipy [link](https://github.com/AthenaEPI/dmipy) tested on v1.0.5

import simulations

# Create train, val, test sets for our example from a scheme
train_inp, train_tar = simulations.create_verdict_data(n_train)
val_inp, val_tar = simulations.create_verdict_data(n_val)
test_inp, test_tar = simulations.create_verdict_data(n_test)
Cbar = train_inp.shape[1]
M = train_tar.shape[1]  # Cbar,M same for val, test data
"""

"""
########## (3-C)
# Generate data using NODDI model [link](https://pubmed.ncbi.nlm.nih.gov/22484410/)
# Uses acquisition scheme [link](https://pubmed.ncbi.nlm.nih.gov/28643354/)
# Requires python package dmipy [link](https://github.com/AthenaEPI/dmipy) tested on v1.0.5

import simulations

# Create train, val, test sets for our example from a scheme
train_inp, train_tar = simulations.create_noddi_data(n_train)
val_inp, val_tar = simulations.create_noddi_data(n_val)
test_inp, test_tar = simulations.create_noddi_data(n_test)
Cbar = train_inp.shape[1]
M = train_tar.shape[1]  # Cbar,M same for val, test data
"""
########## (4)
# Load data into TADRED format

# Data in TADRED format, \bar{C} measurements, M target regresors
data = dict(
    train=train_inp,  # Shape n_train x \bar{C}
    train_tar=train_tar,  # Shape n_train x M
    val=val_inp,  # Shape n_val x \bar{C}
    val_tar=val_tar,  # Shape n_val x M
    test=test_inp,  # Shape n_test x \bar{C}
    test_tar=test_tar,  # Shape n_test x M
)

args = utils.load_base_args()


"""
########## (5-A)
# Option to save data to disk, and TADRED load

data_fil: str = ""  # Add path to saved file
np.save(data_fil, data)
print("Saving data as", data_fil)
pass_data = None
args.data_norm.data_fil = data_fil
"""


########## (5-B)
# Option to pass data to TADRED directly

pass_data = data


########## (6)
# Option to save the output
"""
# Output saved as dict in save_fil=<out_base>/<proj_name>/results/<run_name>_all.pkl
# Load with pickle
args.output.out_base = <ADD>
args.output.proj_name = <ADD>
args.output.run_name = <ADD>
"""

########## (7)
# Simplest version of TADRED, modifying the most important hyperparameters


# Decreasing feature subsets sizes for TADRED to consider
args.tadred_train_eval.feature_set_sizes_Ci = [Cbar, Cbar // 2, Cbar // 4, Cbar // 8, Cbar // 16]

# Feature subset sizess for TADRED evaluated on test data
args.tadred_train_eval.feature_set_sizes_evaluated = [Cbar // 2, Cbar // 4, Cbar // 8, Cbar // 16]

# Scoring net Cbar -> num_units_score[0] -> num_units_score[1] ... -> Cbar units
args.network.num_units_score = [1000, 1000]

# Task net Cbar -> num_units_task[0] -> num_units_task[1] ... -> M units
args.network.num_units_task = [1000, 1000]

tadred_main.run(args, pass_data)


########## (8)
# Modify more TADRED hyperparameters, less impASDASortant, may change results

# Fix score after epoch, E_1 in paper
args.tadred_train_eval.epochs_decay = 25

# Progressively set score to be sample independent across no. epochs, E_2 - E_1 in paper
args.tadred_train_eval.epochs_decay_sigma = 10

# Progressively modify mask across number epochs, E_3 - E_2 in paper
args.tadred_train_eval.epochs_decay = 10

tadred_main.run(args, pass_data)


########## (9)
# Deep learning training hyperparameters for inner loop

# Training epochs per step, set large to use early stopping
args.tadred_train_eval.epochs = 10000

# Training learning rate
args.train_pytorch.optimizer_params.lr = 0.0001

# Training batch size
args.train_pytorch.dataloader_params.batch_size = 1500

tadred_main.run(args, pass_data)

# TODO think of logging on main script
print("EOF", __file__)
