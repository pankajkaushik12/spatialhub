from pathlib import Path
import numpy as np
import cv2


def load_image(img_input: str | Path | np.ndarray) -> np.ndarray:
    """
    Load image from path or use provided numpy array
    """

    if isinstance(img_input, (str, Path)):
        if not Path(img_input).exists():
            raise FileNotFoundError(f"Image not found at path: {img_input}")
        img = cv2.imread(str(img_input), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load or decode image from {img_input}")
        return img
    
    if isinstance(img_input, np.ndarray):
        return img_input.copy()
    
    raise TypeError(f"Expected file path or numpy array, got {type(img_input)}")

