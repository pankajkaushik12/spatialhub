from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .utils import visualize_matches

@dataclass
class MatchResult:
    """
    A dataclass to hold the results of a feature matching operation.
    """
    image_a: str | Path | np.ndarray
    image_b: str | Path | np.ndarray
    keypoints_a: np.ndarray
    keypoints_b: np.ndarray
    confidence: np.ndarray

    def visualize(self, conf_thresh: float = 0.5, max_side: int = 800, top_k: int | None = None, save_path: str | Path | None = None):
        """
        Visualizes the top-k matches between two images using the provided keypoints and confidence scores.
        Only matches with confidence above the specified threshold will be displayed.
        """
        visualize_matches(
            self.image_a, 
            self.image_b, 
            self.keypoints_a, 
            self.keypoints_b, 
            self.confidence, 
            conf_thresh, 
            max_side,
            top_k,
            save_path
        )

