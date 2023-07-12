import pytest

import numpy as np

from tqdm import tqdm
from patient import Patient
from src.utils import resample_instances, perform_windowing
from src.constants import PATIENT_PATHS, BAD_IDS


def test_hu():
    """If an instance contains RescaleSlope and RescaleIntercept
    the pixel arrays are already Hounsfield Units."""
    for patient_path in tqdm(PATIENT_PATHS, desc="Reading patients..", leave=True):
        patinet_id = patient_path.name
        if patinet_id.startswith(".") or patinet_id in BAD_IDS:
            continue
        patient = Patient(patinet_id)
        instances = patient.load_scan()
        for instance in tqdm(instances, desc="Reading instances..", leave=False):
            assert "RescaleSlope" in instance
            assert "RescaleIntercept" in instance


def test_spacing():
    """Test if the spacing is correct after resampling. We need the spacing to be isotropic,
    because we will be using 3D convolutions.

    isotropic - the spacing between each slice is the same in all directions.
    """
    for patient_path in tqdm(PATIENT_PATHS, desc="Reading patients..", leave=True):
        patinet_id = patient_path.name
        if patinet_id.startswith(".") or patinet_id in BAD_IDS:
            continue
        patient = Patient(patinet_id)
        instances = patient.load_scan()
        new_spacing = [1, 1, 1]
        _, spacing = resample_instances(instances, new_spacing)
        assert np.allclose(spacing, new_spacing, atol=0.05)


def test_windowing():
    """Test if the windowing is correct. ("Not needed")"""
    for patient_path in tqdm(PATIENT_PATHS, desc="Reading patients..", leave=True):
        patinet_id = patient_path.name
        if patinet_id.startswith(".") or patinet_id in BAD_IDS:
            continue
        patient = Patient(patinet_id)
        instances = patient.load_scan()
        for instance in tqdm(instances, desc="Reading instances..", leave=False):
            windowed = perform_windowing(instance=instance, bounds=(-1000, 400))
            assert windowed.min() >= -1000
            assert windowed.max() <= 400


if __name__ == "__main__":
    pytest.main([__file__])
