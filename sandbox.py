from typing import List, Tuple, Dict

import cv2
import pydicom
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


from tqdm import tqdm
from pydicom.filereader import dcmread
from pathlib import Path
from pydicom import FileDataset

from patient import Patient
from src.utils import best_rect
from src.constants import DATA_DIR, TRAIN_DIR, TEST_DIR


def view_sample_data():
    patients = sorted(TRAIN_DIR.iterdir())

    for patient in patients[:5]:
        patient_id = patient.name
        if patient_id.startswith("."):
            continue
        patient = Patient(patient_id)
        patient.patient_info
        patient.load_scan(display=True)


def get_features(df):
    # Normalize with range of 30
    # So if the age is 30, then it will be 0
    # If the age is 60, then it will be 1 and so on..
    features = [(df.Age.values[0] - 30) / 30]

    if df.Sex.values[0].lower() == "male":
        features.append(0)
    else:
        features.append(1)

    if df.SmokingStatus.values[0] == "Never smoked":
        features.extend([0, 0])
    elif df.SmokingStatus.values[0] == "Ex-smoker":
        features.extend([1, 1])
    elif df.SmokingStatus.values[0] == "Currently smokes":
        features.extend([0, 1])
    else:
        features.extend([1, 0])
    return np.array(features)


if __name__ == "__main__":
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # # print(train_df.shape)  # (1549, 7)
    # # print(test_df.shape)  # (5, 7)
    # # print(train_df.head())

    # # print(train_df.isna().sum())
    # # print(test_df.isna().sum())
    # # No missing values

    # # print(train_df.columns)
    # # Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus
    # # print(test_df.columns)
    # # Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

    # duplicateEntries = train_df[
    #     train_df.duplicated(subset=["Patient", "Weeks"], keep=False)
    # ]
    # # keep = False means drop all duplicates (including the first & last occurrence)
    # # print(len(duplicateEntries))
    # # print(len(duplicateEntries) / len(train_df))
    # # 0.009% of the data is duplicated we can drop these

    # train_df = train_df.drop_duplicates(subset=["Patient", "Weeks"], keep=False)

    # # Group by patient and get the first entry
    # data = train_df.groupby("Patient").first().reset_index()
    # # print(data.head())

    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))
    # sns.histplot(
    #     data["Age"], ax=ax0, bins=data["Age"].max() - data["Age"].min() + 1, kde=True
    # )
    # ax0.annotate(
    #     "Min: {:,}".format(data["Age"].min()),
    #     xy=(data["Age"].min(), 0.005),
    #     xytext=(data["Age"].min(), 5),
    #     bbox=dict(boxstyle="round", fc="none", ec="gray"),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.2"),
    # )
    # ax0.annotate(
    #     "Max: {:,}".format(data["Age"].max()),
    #     xy=(data["Age"].max(), 0.005),
    #     xytext=(data["Age"].max() - 7, 5),
    #     bbox=dict(boxstyle="round", fc="none", ec="gray"),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"),
    # )
    # ax0.axvline(x=data["Age"].median(), color="k", linestyle="--", linewidth=1)
    # ax0.annotate(
    #     "Med: {:,}".format(data["Age"].median()),
    #     xy=(data["Age"].median(), 9.5),
    #     xytext=(data["Age"].max() - 15, 12),
    #     bbox=dict(boxstyle="round", fc="none", ec="gray"),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"),
    # )
    # sns.kdeplot(
    #     data[data["SmokingStatus"] == "Ex-smoker"].Age,
    #     ax=ax1,
    #     label="Ex-Smoker",
    # )
    # sns.kdeplot(
    #     data[data["SmokingStatus"] == "Never smoked"].Age,
    #     ax=ax1,
    #     label="Never Smoked",
    # )
    # sns.kdeplot(
    #     data[data["SmokingStatus"] == "Currently smokes"].Age,
    #     ax=ax1,
    #     label="Currently Smokes",
    # )
    # ax1.legend(loc="upper right", fontsize="small")
    # sns.countplot(x="Sex", data=data, ax=ax2)
    # sns.countplot(
    #     x="SmokingStatus",
    #     data=data,
    #     hue="Sex",
    #     ax=ax3,
    #     order=["Never smoked", "Ex-smoker", "Currently smokes"],
    # )
    # fig.suptitle("Distribution of data", fontsize=14)
    # plt.show()

    # patients = sorted(TRAIN_DIR.iterdir())

    # for patient in patients[:5]:
    #     patient_id = patient.name
    #     if patient_id.startswith("."):
    #         continue
    #     patient = Patient(patient_id)
    #     patient.patient_info
    #     slices = patient.load_scan()
    #     s = get_img(slices[5])
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    #     ax[0].imshow(s, cmap="gray")
    #     ax[1].imshow(slices[5].pixel_array, cmap="gray")
    #     plt.show()
    #     break
