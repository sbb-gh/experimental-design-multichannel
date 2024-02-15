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

import numpy as np
from dmipy.core import modeling_framework  # type: ignore
from dmipy.data import saved_acquisition_schemes  # type: ignore
from dmipy.distributions import distribute_models  # type: ignore
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models  # type: ignore
from dmipy.utils import utils  # type: ignore

log = logging.getLogger(__name__)


def create_noddi_data(
    num_samples: int,
    acq_scheme: saved_acquisition_schemes = saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme(),
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate NODDI data

    Returns:
        Signals_NODDI: Simulated MRI signal
        Parameters_NODDI: Parameters used to simulate signal
    """

    # Create NODDI model from https://pubmed.ncbi.nlm.nih.gov/22484410
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    watson_dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])

    # Fix parameters, tortuosity constraints, isotropic diffusivity
    # TODO magic numbers like 1.7e-9 or 3e-9 should be variables instead with a meaningful name that you could use and change when needed.
    watson_dispersed_bundle.set_tortuous_parameter(
        "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
    )
    watson_dispersed_bundle.set_equal_parameter("G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par")
    watson_dispersed_bundle.set_fixed_parameter("G2Zeppelin_1_lambda_par", 1.7e-9)
    NODDI_mod = modeling_framework.MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    NODDI_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 3e-9)

    # TODO add reference in table, paste directly into function or create dict?
    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2))
    odi = np.random.uniform(low=0.01, high=0.99, size=num_samples)
    f_bundle = np.random.uniform(low=0.01, high=0.99, size=num_samples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=num_samples)
    f_1 = 1 - f_0

    # Big parameter vector for simulate_signal
    Parameters_NODDI_dmipy = NODDI_mod.parameters_to_parameter_vector(
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
        SD1WatsonDistributed_1_partial_volume_0=f_bundle,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
    )

    log.info("Simulating NODDI data")
    Signals_NODDI = NODDI_mod.simulate_signal(acq_scheme, Parameters_NODDI_dmipy)
    # NODDI_mod.parameter_names
    Signals_NODDI = add_noise(Signals_NODDI)

    # add cartesian parameters parameters_NODDI are the parameters to learn
    Parameters_NODDI = add_cartesian(NODDI_mod, Parameters_NODDI_dmipy)
    Signals_NODDI = Signals_NODDI.astype(np.float32)
    Parameters_NODDI = Parameters_NODDI.astype(np.float32)

    return Signals_NODDI, Parameters_NODDI


def verdict_params_norm(Parameters_VERDICT: np.ndarray) -> np.ndarray:
    """Normalize parameters to be approximately equal so evaluation will penalize incorrect prediction roughly same"""

    log.info("Normalizing means of Parameters_VERDICT")
    Parameters_VERDICT[:, 0] = Parameters_VERDICT[:, 0] * (10**5) * 0.5
    Parameters_VERDICT[:, 1] = Parameters_VERDICT[:, 1] * (10**8)

    return Parameters_VERDICT


def create_verdict_data(
    num_samples: int,
    acq_scheme: saved_acquisition_schemes = saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme(),
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate VERDICT data"""

    """Create VERDICT Model https://pubmed.ncbi.nlm.nih.gov/25426656"""
    # Fix parameters and set ranges
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    VERDICT_mod = modeling_framework.MultiCompartmentModel(models=[sphere, ball, stick])
    VERDICT_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 0.9e-9)
    VERDICT_mod.set_parameter_optimization_bounds("C1Stick_1_lambda_par", [3.05e-9, 10e-9])

    # Random parameters wisth sensible upper and lower bounds
    # TODO add reference to table
    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2))
    lambda_par = np.random.uniform(low=1e-9, high=10e-9, size=num_samples)  # in m^2/s
    diameter = np.random.uniform(low=0.01e-06, high=20e-06, size=num_samples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=num_samples)
    f_1 = np.random.uniform(low=0.01, high=0.99 - f_0, size=num_samples)
    f_2 = 1 - f_0 - f_1

    # Big parameter vector to simulate_signal
    Parameters_VERDICT_dmipy = VERDICT_mod.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=lambda_par,
        S4SphereGaussianPhaseApproximation_1_diameter=diameter,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
        partial_volume_2=f_2,
    )

    log.info("Simulating VERDICT data")
    Signals_VERDICT = VERDICT_mod.simulate_signal(acq_scheme, Parameters_VERDICT_dmipy)
    Signals_VERDICT = add_noise(Signals_VERDICT)

    # Add Cartesian parameters
    Parameters_VERDICT = add_cartesian(VERDICT_mod, Parameters_VERDICT_dmipy)

    Signals_VERDICT = Signals_VERDICT.astype(np.float32)
    Parameters_VERDICT = Parameters_VERDICT.astype(np.float32)
    Parameters_VERDICT = verdict_params_norm(Parameters_VERDICT)

    return Signals_VERDICT, Parameters_VERDICT


def add_noise(data: np.ndarray, scale: float = 0.02) -> np.ndarray:
    """Add Rician noise to data"""

    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def add_cartesian(model_dmipy, parameters_dmipy: np.ndarray) -> np.ndarray:
    """Create parameters including spherical and Cartesian, parameters replacing spherical with Cartesian"""

    # mu_{name} is the name of the spherical coordinates to convert
    # Find mu parameter and therefore the index of theta and phi
    # TODO refactor below w: mu_index = next(i for i, s in enumerate(model.parameter_names) if "_mu" in s)
    mu_index = [i for i, s in enumerate(model_dmipy.parameter_names) if "_mu" in s][0]
    theta_phi_index = mu_index, mu_index + 1

    # Convert to cartesian coordinates
    mu_cartesian = utils.unitsphere2cart_Nd(parameters_dmipy[:, theta_phi_index])

    # Flip the direction of any cartesian points in the lower half of the sphere
    lower_index = mu_cartesian[:, 2] < 0
    mu_cartesian[lower_index, :] = -mu_cartesian[
        lower_index, :
    ]  # TODO check if can replace w mu_cartesian[lower_index, :] *= -1

    # Add cartesian coordinates to the parameter array
    parameters_spherical_and_cartesian = np.append(parameters_dmipy, mu_cartesian, axis=1)

    # Remove spherical coordinates ("mu") from the parameter
    parameters_cartesian_only = np.delete(
        parameters_spherical_and_cartesian, theta_phi_index, axis=1
    )

    return parameters_cartesian_only
