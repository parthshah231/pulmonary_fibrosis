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
        slices.sort(key=lambda x: x.ImagePositionPatient[2])

        if display:
            x, y = best_rect(len(slices))
            fig, axes = plt.subplots(nrows=x, ncols=y, figsize=(20, 10))
            axes = axes.flatten()
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.pixel_array, cmap=plt.cm.gray)

            plt.show()

        return slices

    @property
    def patient_info(self):
        print("Patient Info:")
        print("-------------")
        print(f"Patient ID\t\t\t: {self.patient_id}")
        print(f"Patient Name\t\t\t: {self.scan_details[0].PatientName}")
        # print(f"Patient Age: {self.scan_details[0].PatientAge}")
        print(f"Patient Sex\t\t\t: {self.scan_details[0].PatientSex}")
        print(f"Patient Modality\t\t: {self.scan_details[0].Modality}")
        print(f"Patient Body Part Examined\t: {self.scan_details[0].BodyPartExamined}")
        print(f"Study Instance UID\t\t: {self.scan_details[0].StudyInstanceUID}")
        print(f"Data exists\t\t\t: {True if self.scan_details[0].PixelData else False}")
