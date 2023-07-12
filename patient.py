from typing import List

import matplotlib.pyplot as plt

from pydicom.filereader import dcmread
from pydicom import FileDataset

from src.utils import best_rect
from src.constants import TRAIN_DIR


class Patient:
    """A class to represent a patient in the dataset"""

    def __init__(self, patient_id: str):
        self.path = TRAIN_DIR / patient_id
        self.patient_id = patient_id
        self.dcm_files = list(self.path.rglob("*.dcm"))
        self.scan_details = self.load_scan()

    def load_scan(self, display: bool = False) -> List[FileDataset]:
        """Loads the scan details of a patient. A scan in this dataset
        in axial view is a collection of DICOM files.

        https://en.wikipedia.org/wiki/Anatomical_plane

        There are 3 orientations of a scan:
        1. Axial
        2. Coronal
        3. Sagittal

        1. Axial view is parallel to the ground; it separates the superior
        from the inferior, or the head from the feet.
        2. Coronal view is perpendicular to the ground; it separates the
        anterior from the posterior, or the front from the back.
        3. Sagittal view is perpendicular to the ground; it separates the
        left from the right."""

        slices: List[FileDataset] = [dcmread(dcm_file) for dcm_file in self.dcm_files]
        try:
            slices.sort(key=lambda x: x.ImagePositionPatient[2])
        except Exception:
            raise AttributeError(
                f"Patient {self.patient_id} does not have ImagePositionPatient"
            )

        if display:
            x, y = best_rect(len(slices))
            fig, axes = plt.subplots(nrows=x, ncols=y, figsize=(20, 10))
            axes = axes.flatten()
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.pixel_array, cmap=plt.cm.gray)

            plt.show()

        return slices

    def __str__(self) -> str:
        info_string = """
        Patient Info:
        -------------
        Patient ID                  : {0}
        Patient Name                : {1}
        Patient Sex                 : {2}
        Patient Modality            : {3}
        Patient Body Part Examined  : {4}
        Study Instance UID          : {5}
        Data exists                 : {6}
        """

        patient_id = self.patient_id
        patient_name = self.scan_details[0].PatientName
        patient_sex = self.scan_details[0].PatientSex
        patient_modality = self.scan_details[0].Modality
        body_part_examined = self.scan_details[0].BodyPartExamined
        study_instance_uid = self.scan_details[0].StudyInstanceUID
        data_exists = True if self.scan_details[0].PixelData else False

        return info_string.format(
            patient_id,
            patient_name,
            patient_sex,
            patient_modality,
            body_part_examined,
            study_instance_uid,
            data_exists,
        )
