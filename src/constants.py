from pathlib import Path


def ensure_dir(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ensure_dir(ROOT / "data")
TRAIN_DIR = ensure_dir(DATA_DIR / "train")
TEST_DIR = ensure_dir(DATA_DIR / "test")
BASE_AGE = 30

PATIENT_PATHS = sorted(TRAIN_DIR.iterdir())
# Bad ids here mean that patient or an instance in that patient does not
# have ImagePositionPatient attribute that helps in arranging the CT scans.
# So we avoid them.
BAD_IDS = [
    "ID00026637202179561894768",
    "ID00128637202219474716089",
    "ID00132637202222178761324",
]
