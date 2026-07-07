import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from huggingface_hub import hf_hub_download

from ...structures import MatchResult
from ..._imageio import load_image

import logging

logger = logging.getLogger(__name__)

class EfficientLoFTRAdapter():

    # Initialize the ONNX model
    def __init__(self, model_path: str | Path | None = None, provider: str = "CPUExecutionProvider", model_type: str = "full"):
        """
        Adapter for EfficientLoFTR. Handles ONNX execution, input normalization, and coordinate projection.
        """
        if model_type not in ['full', 'opt']:
            raise ValueError("model_type must be either 'full' or 'opt'")

        if model_path is None:
            filename = "eloftr_outdoor_full.onnx" if model_type == "full" else "eloftr_outdoor_opt.onnx"
            
            logger.info(f"No local model path provided. Fetching '{model_type}' weights from Hugging Face...")
            try:
                model_path = hf_hub_download(repo_id="pankaj-kaushik/efficient-loftr-onnx", filename=filename)
            except Exception as e:
                raise RuntimeError(
                    "Failed to download model weights from Hugging Face. "
                    f"Please check your internet connection. Error: {e}"
                ) from e

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model weights not found at: {model_path}")
        
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        logger.info("Initializing EfficientLoFTR model...")
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
        self.input_names = [i.name for i in self.session.get_inputs()]
        
        # Log the active provider to confirm successful initialization
        active_provider = self.session.get_providers()[0]
        if active_provider != provider:
            logger.warning(
                "Requested execution provider '%s', but ONNX Runtime is using '%s'.",
                provider,
                active_provider,
            )
        else:
            logger.info("Model is running on: %s", active_provider)

    # Run inference
    def match(self, image_a: str | Path | np.ndarray, image_b: str | Path | np.ndarray, max_dim: int | None = 1024) -> MatchResult:
        """
        Takes two raw images, processes them, and returns standardized matches.
        """
        # Pre-process (Resize to % 32, normalize, and get inverse scale factors)
        tensor_a, scale_a = self._preprocess(image_a, max_dim = max_dim)
        tensor_b, scale_b = self._preprocess(image_b, max_dim = max_dim)

        # Pad both tensors to the shared maximum dimensions
        h_a, w_a = tensor_a.shape[2:]
        h_b, w_b = tensor_b.shape[2:]
        max_h = max(h_a, h_b)
        max_w = max(w_a, w_b)

        # We only pad the bottom and right edges of the Height and Width dimensions
        pad_a = ((0, 0), (0, 0), (0, max_h - h_a), (0, max_w - w_a))
        pad_b = ((0, 0), (0, 0), (0, max_h - h_b), (0, max_w - w_b))

        tensor_a = np.pad(tensor_a, pad_a, mode='constant', constant_values=0)
        tensor_b = np.pad(tensor_b, pad_b, mode='constant', constant_values=0)

        # ONNX Inference
        outputs = self.session.run(
            ["mkpts0_f", "mkpts1_f", "mconf"], 
            {
                self.input_names[0]: tensor_a,
                self.input_names[1]: tensor_b
            }
        )
        
        mkpts0_raw, mkpts1_raw, mconf = outputs

        # If no matches were found, return empty arrays safely
        if len(mconf) == 0:
            return MatchResult(
                image_a=image_a, 
                image_b=image_b, 
                keypoints_a=np.empty((0, 2)), 
                keypoints_b=np.empty((0, 2)), 
                confidence=np.empty((0,))
            )

        # Project keypoints back to the original image resolutions
        mkpts0_orig = mkpts0_raw * scale_a
        mkpts1_orig = mkpts1_raw * scale_b

        # 5. Filter out any garbage matches that accidentally landed in the padding area
        valid_mask = (mkpts0_raw[:, 0] < w_a) & (mkpts0_raw[:, 1] < h_a) & (mkpts1_raw[:, 0] < w_b) & (mkpts1_raw[:, 1] < h_b)

        return MatchResult(
                image_a=image_a, 
                image_b=image_b, 
                keypoints_a=mkpts0_orig[valid_mask], 
                keypoints_b=mkpts1_orig[valid_mask], 
                confidence=mconf[valid_mask]
            )

    # pre-process
    def _preprocess(self, img_input: str | Path | np.ndarray, max_dim: int = 1024):
        """
        Ensures grayscale, resizes to multiples of 32, and converts to (1, 1, H, W) float32.
        Returns the tensor and the scaling factors [scale_x, scale_y] to reverse the transformation.
        """

        # Load the image using the utility function
        img = load_image(img_input)

        # Normalize Channels to Grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:    # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:  # BGR/RGB 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        orig_h, orig_w = img.shape

        # Scale down if the largest dimension exceeds max_dim
        scale_factor = 1.0
        if max_dim is not None and max(orig_h, orig_w) > max_dim:
            scale_factor = max_dim / max(orig_h, orig_w)
            inter_h = int(orig_h * scale_factor)
            inter_w = int(orig_w * scale_factor)
            img = cv2.resize(img, (inter_w, inter_h), interpolation=cv2.INTER_AREA)

        # Model architecture requires dimensions divisible by 32
        curr_h, curr_w = img.shape
        new_w = max(32, (curr_w // 32) * 32)
        new_h = max(32, (curr_h // 32) * 32)
        
        # Calculate the projection scale mapping (Original / Resized)
        scale = np.array([orig_w / new_w, orig_h / new_h], dtype=np.float32)

        if (new_w, new_h) != (orig_w, orig_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and add batch & channel dimensions: (1, 1, H, W)
        tensor = img.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, 1))

        return tensor, scale

