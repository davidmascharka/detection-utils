from typing import Tuple, Union
from pathlib import Path
import numpy as np


def load_data(root_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_dir = Path(root_dir)
    images = np.load(root_dir / "images.npy")
    # we're loading object arrays here so we need to allow_pickle
    boxes = np.load(root_dir / "boxes.npy", allow_pickle=True)
    labels = np.load(root_dir / "labels.npy", allow_pickle=True)
    return images, boxes, labels
