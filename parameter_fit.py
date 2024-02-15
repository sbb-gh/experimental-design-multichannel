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


import logging
import os

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst import dki, dti,

log = logging.getLogger(__name__)

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "1000"


def dti_dki_msdki_downstream_metrics(
    data: np.ndarray,  # (no_voxels, measurements)
    bvals: np.ndarray,  # 1D
    bvecs: np.ndarray,  # (3,measurements)
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fits dti,dki,msdki models and computes downstream metrics for data, used in paper-section 4.3"""

    out = dict()
    gtab = gradient_table(bvals, bvecs)

    # Fit DTI model and compute downstream metrics (`parameter maps')
    dtimodel = dti.TensorModel(gtab)
    dtifit = dtimodel.fit(data, mask=mask)  # (imgnii, mask=masknii)
    dti_params = dtifit.model_params
    dti_FA = np.clip(dtifit.fa, 0, 1)
    dti_MD = np.clip(dtifit.md, 0, 0.003)
    dti_AD = np.clip(dtifit.ad, 0, 0.003)
    dti_RD = np.clip(dtifit.rd, 0, 0.003)
    dtifit_dict = dict(
        dti_params=dti_params,
        dti_FA=dti_FA,
        dti_MD=dti_MD,
        dti_AD=dti_AD,
        dti_RD=dti_RD,
    )
    log.info("dtifit finished")
    out.update(dtifit_dict)

    # Fit DKI model and compute downstream metrics
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)
    dki_params = np.clip(dkifit.model_params, 0, 3)
    dki_MK = np.clip(dkifit.mk(), 0, 3)
    dki_AK = np.clip(dkifit.ak(), 0, 3)
    dki_RK = np.clip(dkifit.rk(), 0, 3)
    dki_FA = np.clip(dkifit.fa, 0, 1)
    dki_MD = np.clip(dkifit.md, 0, 0.03)
    dki_AD = np.clip(dkifit.ad, 0, 0.03)
    dki_RD = np.clip(dkifit.rd, 0, 0.03)

    dkifit_dict = dict(
        dki_params=dki_params,
        dki_MK=dki_MK,
        dki_AK=dki_AK,
        dki_RK=dki_RK,
        dki_FA=dki_FA,
        dki_MD=dki_MD,
        dki_AD=dki_AD,
        dki_RD=dki_RD,
    )
    log.info("dkifit finished")
    out.update(dkifit_dict)

    # Fit MSDKI model and compute downstream metrics
    # https://dipy.org/documentation/1.4.1./examples_built/reconst_msdki/
    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
    msdki_fit = msdki_model.fit(data, mask=mask)
    msdki_msd = np.clip(msdki_fit.msd, 0, 0.003)
    msdki_msk = np.clip(msdki_fit.msk, 0, 3)
    msdki_dict = dict(
        msdki_msd=msdki_msd,
        msdki_msk=msdki_msk,
    )
    log.info("msdki finished")
    out.update(msdki_dict)

    out = {key: val.astype(np.float32) for key, val in out.items()}
    return out


def compute_mse_downstream_metrics(
    data_pred: np.ndarray,  # (no_voxels,measurements)
    data_tar: np.ndarray,  # (no_voxels,measurements)
    bvals: np.ndarray,  # (measurements,)
    bvecs: np.ndarray,  # (3, measurements)
) -> dict[str, np.ndarray]:
    """Compute mean-squared-error of dti,dki,msdki downstream metrics (parameter maps).

    Returns:
    results: MSE difference between data_pred, data_tar on downstream metrics
    """

    data_tar_out = dti_dki_msdki_downstream_metrics(data_tar, bvals, bvecs)
    data_pred_out = dti_dki_msdki_downstream_metrics(data_pred, bvals, bvecs)

    results = dict()
    metrics = data_pred_out.keys()
    for metric in metrics:
        log.info(metric)
        diff = data_pred_out[metric] - data_tar_out[metric]
        diff_mse = diff**2
        diff_metric = np.mean(diff_mse)
        results[metric] = diff_metric

    return results
