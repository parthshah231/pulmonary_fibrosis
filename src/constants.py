from pathlib import Path


def ensure_dir(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ensure_dir(ROOT / "data")
TRAIN_DIR = ensure_dir(DATA_DIR / "train")
TEST_DIR = ensure_dir(DATA_DIR / "test")
