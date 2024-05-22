# Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection

This is the official code for the paper available on [OpenReview](https://openreview.net/forum?id=MloaGA6WwX&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)) and on [arxiv](https://arxiv.org/abs/2210.06891v3).

The neural-network based algorithm TADRED is available [here](https://github.com/sbb-gh/tadred).

If you find this repo useful, please consider citing the paper:

@article{<br>
&nbsp; &nbsp; title={Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection},<br>
&nbsp; &nbsp; author={Stefano B. Blumberg and Paddy J. Slator and Daniel C. Alexander},<br>
&nbsp; &nbsp; journal={In: International Conference on Learning Representations (ICLR)},<br>
&nbsp; &nbsp; year={2024}<br>
}

## Contact

stefano.blumberg.17@ucl.ac.uk

## Installation Part 1: Environment

First create an environment and enter it, we use Python v3.10.4.  We provide two examples either using Pyenv or Conda:

## Pyenv

```bash
# Pyenv documentation is [link](https://github.com/pyenv), where <INSTALL_DIR> is the directory the virtual environment is installed in.
python3.10 -m venv <INSTALL_DIR>/experimental-design-multichannel-env # Use compatible Python version e.g. 3.10.4
. <INSTALL_DIR>/experimental-design-multichannel-env/bin/activate
```

## Conda

```bash
# Conda documentation is [link](https://docs.conda.io/en/latest/), where <INSTALL_DIR> is the directory the virtual environment is installed in.
conda create -n experimental-design-multichannel-env python=3.10.4
conda activate experimental-design-multichannel-env
```

## Installation Part 2: TADRED and other Packages

Code requires:<br>
[tadred](https://github.com/sbb-gh/tadred/tree/main): the novel method presented in the paper with dependencies pytorch, numpy, pyyaml, hydra,<br>
optional modules to generate the data: dipy, dmipy, nibabel.<br>
<br>
Code is tested using PyTorch v2.0.0, cuda 11.7 on the GPU, dipy v1.5.0, nibabel v5.1.0, dmipy v1.0.5.

We provide two options for installing the code:

### Python Package from Source


```bash
pip install git+https://github.com/sbb-gh/experimental-design-multichannel.git@main
```

### Using pip

```bash
pip install numpy==1.23.4 git+https://github.com/AthenaEPI/dmipy.git@1.0.1 # use compatible numpy
pip install dipy==1.5.0
pip install nibabel==5.1.0
pip install git+https://github.com/sbb-gh/tadred.git@main # can also install tadred from source: www.github.com/sbb-gh/tadred
```

## Tutorial

We provide a tutorial [in tutorial.py](./tutorial.py) that provides examples on generating data, options to load the data into TADRED, various hyperparameter choices for TADRED, and options to save the results.

## Results

We provide Python code to generate data, train TADRED and perform evaulation.  Note, to replicate exact results, we perform a hyperparameter search on the two networks - described in paper appendix A.

### VERDICT Results

Duplicating the results for VERDICT in table 1.

```python
from dmipy.data import saved_acquisition_schemes
from tadred import data_processing, tadred_main, utils

import simulations

scheme = saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**6, 10**5, 10**5

utils.set_numpy_seed(0)
train_sims = simulations.create_verdict_data(nsamples_train, scheme)
val_sims = simulations.create_verdict_data(nsamples_val, scheme)
test_sims = simulations.create_verdict_data(nsamples_test, scheme)
data = data_processing.tadred_data_format(train_sims,val_sims,test_sims)

args = utils.load_base_args_combine_with_yaml("./cfg_files/table1_cfg.yaml")
# Set the below to network sizes, see paper-section-B
args.network.num_units_score = [] # CHANGE e.g. [1000, 1000]
args.network.num_units_task = [] # CHANGE e.g. [1000, 1000]

tadred_main.run(args, data)
```

### NODDI Results

Duplicating the results for NODDI in appendix B table 7.

```python
from dmipy.data import saved_acquisition_schemes
from tadred import data_processing, tadred_main, utils

import simulations

scheme = saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**5, 10**4, 10**4

utils.set_numpy_seed(0)
train_sims = simulations.create_noddi_data(nsamples_train, scheme)
val_sims = simulations.create_noddi_data(nsamples_val, scheme)
test_sims = simulations.create_noddi_data(nsamples_test, scheme)
data = data_processing.tadred_data_format(train_sims,val_sims,test_sims)

args = utils.load_base_args_combine_with_yaml("./cfg_files/table7_cfg.yaml")

# Set the below to network sizes, see paper-section-B
args.network.num_units_score = [] # CHANGE e.g. [1000, 1000]
args.network.num_units_task = [] # CHANGE e.g. [1000, 1000]

tadred_main.run(args, data)
```

### HCP Test Retest Results

To reproduce results in figure 2.

```python
from pathlib import Path

import nibabel as nib
import numpy as np
from tadred_code import data_processing, tadred_main, utils

from parameter_fit import compute_mse_downstream_metrics

# Register at https://db.humanconnectome.org/data/projects/HCP_Retest
# TODO define load_base  where downloaded subjects below
subj_splits = dict(
    subj_train=("103818_1", "105923_1", "111312_1"),
    subj_val=("114823_1",),
    subj_test=("115320_1",),
)

subj_processed: dict[str, np.ndarray] = dict()

for subj_name in subj_splits["subj_train"] + subj_splits["subj_val"] + subj_splits["subj_test"]:
    # Load data, choose <load_base>
    subj_dir = Path(load_base, "data1", subj_name, "T1w", "Diffusion")
    subj_data = nib.load(Path(subj_dir, "data.nii.gz")).get_fdata().astype(np.float32)
    subj_mask = nib.load(Path(subj_dir, "nodif_brain_mask.nii.gz")).get_fdata().astype(np.float32)

    idxs_mask = np.where(subj_mask == 1)
    subj_brain = subj_data[idxs_mask]

    # Same for all subjects
    bvals = np.loadtxt(Path(subj_dir, "bvals"), dtype=np.float32)
    bvecs = np.loadtxt(Path(subj_dir, "bvecs"), dtype=np.float32)

    # Normalizing b0values
    bvals0_idx = np.where(bvals <= 5)
    bvals0 = subj_brain[:, bvals0_idx]
    bvals0 = np.mean(bvals0, axis=2)
    subj_brain /= bvals0

    # Remove nans
    not_nan = ~np.isnan(np.mean(subj_brain, axis=1))
    subj_brain = subj_brain[not_nan, :]
    idxs_mask_mod = tuple(idxs_mask_i[np.where(not_nan)] for idxs_mask_i in idxs_mask)
    subj_brain_3D = np.zeros(subj_data.shape, dtype=np.float32)
    subj_brain_3D[idxs_mask_mod] = subj_brain

    subj_processed[subj_name] = subj_brain

data: dict[str, np.ndarray | dict[str, np.ndarray]] = dict()
data["other"] = dict(bvals=bvals, bvecs=bvecs)

for split in ("train", "val", "test"):
    data[split] = []
    for subj_name in subj_splits["subj_" + split]:
        data[split].append(subj_processed[subj_name])
    data[split] = np.concatenate(data[split])

args = utils.load_base_args_combine_with_yaml("./cfg_files/table7_cfg.yaml")

# Set the below to network sizes, see paper-section-B
args.network.num_units_score = [] # CHANGE e.g. [1000, 1000]
args.network.num_units_task = [] # CHANGE e.g. [1000, 1000]

results = tadred_main.run(args, data)

# Compute downstream metrics
for Ci in results["args"]["tadred_train_eval"]["feature_set_sizes_evaluated"]:
    data_pred = results[Ci]["test_output"]
    data_tar = data["test"]
    downstream_metrics = compute_mse_downstream_metrics(data_pred, data_tar, bvals, bvecs)
    results[Ci]["downstream_metrics"] = downstream_metrics
```


### Remote Sensing Hyperspectral Imaging (AVIRIS) Results

Duplicating the results for AVIRIS hyperspectral imaging in table 3.

```python
import numpy as np
import skimage.io
from tadred import tadred_main, utils  # data_processing,

np.random.seed(0)

# Download data from https://purr.purdue.edu/publications/1947/serve/1?render=archive
# Unzip into chosen <data_dir>

trainval_load = <data_dir>'/10_4231_R7RX991C/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif'
test_load = <data_dir>'/10_4231_R7RX991C/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif'

# Prepare training and validation data removing outliers
trainval_input = skimage.io.imread(trainval_load).astype(np.float32)
trainval_input[0,...] = np.clip(trainval_input[0,...], None, np.percentile(trainval_input[0,...], q=99.999))
trainval_input[2,...] = np.clip(trainval_input[2,...], None, np.percentile(trainval_input[2,...], q=99.999))
trainval_input = np.reshape(trainval_input, (trainval_input.shape[0],-1))
trainval_input = np.transpose(trainval_input)
max_val = np.float32(np.percentile(np.abs(trainval_input),q=99,axis=0))
trainval_input = 255*trainval_input / max_val

# Prepare test data removing outliers
test_input = skimage.io.imread(test_load).astype(np.float32)
test_input[0,-1,...] = np.percentile(test_input[0,...], q=50)
test_input[2,...] = np.clip(test_input[2,...], None, np.percentile(test_input[2,...], q=99.999) )
test_input[3,...] = np.clip(test_input[3,...], None, np.percentile(test_input[3,...], q=99.999) )
test_input = np.reshape(test_input, (test_input.shape[0],-1))
test_input = np.transpose(test_input)
test_input = 255*test_input / max_val

# Separate training and validation data
n_train = int(trainval_input.shape[0]*0.9)
n_val = n_train_val - n_train
samples_idx = np.array(range(trainval_input.shape[0]))
np.random.shuffle(samples_idx)
data = dict(
    train=trainval_input[samples_idx[0:n_train],:],
    val=trainval_input[samples_idx[n_train:],:],
    test=test_input,
)

args = utils.load_base_args_combine_with_yaml("./cfg_files/table3_cfg.yaml")

# Set the below to network sizes, see paper-section-B
args.network.num_units_score = [] # CHANGE e.g. [1000, 1000]
args.network.num_units_task = [] # CHANGE e.g. [1000, 1000]

tadred_main.run(args, data)

```


# Acknowledgments

Many thanks to David Perez-Suarez, Stefan Piatek, Tom Young, who provided valuable feedback on the code.



