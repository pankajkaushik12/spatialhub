from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Literal
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

@dataclass
class DepthPrediction:
    """
    Output of a depth estimation model.

    Attributes:
        depth: Depth maps of shape (N, H, W).
        conf: Optional confidence maps of shape (N, H, W).
        intrinsics: Optional camera intrinsics of shape (N, 3, 3).
        extrinsics: Optional camera extrinsics of shape (N, 4, 4).
        depth_type: Representation of the predicted depth.
    """
    
    depth: np.ndarray                  
    conf: np.ndarray | None = None     
    intrinsics: np.ndarray | None = None   
    extrinsics: np.ndarray | None = None   
    depth_type: Literal["metric", "relative", "inverse", "disparity"] = "metric"

