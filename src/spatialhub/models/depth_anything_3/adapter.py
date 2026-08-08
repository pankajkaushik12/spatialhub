import logging
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from pathlib import Path

from .DepthAnything3 import normalize_extrinsics, align_poses_umeyama
from .DepthAnything3 import visualize_depth, process_mono_sky_estimation_np, align_nested_depth_np
from .DepthAnything3 import InputProcessor

from ...structures import DepthPrediction

logger = logging.getLogger(__name__)

class DepthAnything3Adapter():

    def __init__(self, model_name: str | list[str] = "da3_base", model_variant: str | None = None, provider: str = "CPUExecutionProvider", align_to_input_ext_scale: bool = True, ransac_view_thresh: int = 10):
        """
        Adapter for Depth Anything 3. Handles ONNX execution, Hugging Face weight fetching, and geometry alignment.
        """

        # Ensure model_name is a list for uniform processing
        model_names = [model_name] if isinstance(model_name, str) else model_name
        
        if len(model_names) > 2:
            raise ValueError("A maximum of 2 models (Main and Metric) can be provided.")

        self.ort_sessions = []

        # Use explicit variant if provided, otherwise auto-parse from filename
        if model_variant is not None:
            self.model_variant = model_variant
        else:
            # Combine names to check variant flags
            combined_names = " ".join([str(n).lower() for n in model_names])
            if "nested" in combined_names or len(model_names) == 2:
                self.model_variant = "metric_nested"
            elif "metric" in combined_names:
                self.model_variant = "metric"
            else:
                self.model_variant = "relative"
            
        logger.info(f"Detected DA3 variant: {self.model_variant}")

        # Fetch and load each session
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        for name in model_names:
            name_str = str(name)

            # If the user requests 'da3metric_large', load the underlying 'da3mono_large.onnx' file instead.
            if "metric" in name_str and "nested" not in name_str:
                file_to_load = name_str.replace("metric", "mono")
                logger.info(f"Optimization: Mapping requested model '{name_str}' to underlying file '{file_to_load}'.")
            else:
                file_to_load = name_str

            # Ensure it ends with .onnx
            filename = file_to_load if file_to_load.endswith(".onnx") else f"{file_to_load}.onnx"
            
            # Check local path first, otherwise fetch from Hugging Face
            local_path = Path(filename)
            if not local_path.exists():
                logger.info(f"No local model found at '{local_path}'. Fetching from Hugging Face...")
                try:
                    # Download the main .onnx file
                    model_path = hf_hub_download(
                        repo_id="pankaj-kaushik/depth-anything-3-onnx", 
                        filename=filename
                    )
                    
                    # Download the .data file if it exists on HF
                    data_filename = f"{filename}.data"
                    try:
                        hf_hub_download(
                            repo_id="pankaj-kaushik/depth-anything-3-onnx", 
                            filename=data_filename
                        )
                        logger.debug(f"Fetched sidecar data file: {data_filename}")
                    except Exception:
                        # Silently pass if no .data file exists (self-contained ONNX model)
                        pass

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download model weights for '{filename}'. "
                        f"Please check internet connection or repo details. Error: {e}"
                    ) from e
            else:
                model_path = str(local_path)
                logger.info(f"Found local model at '{local_path}'.")

            session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
            
            # Provider Verification Logging
            active_provider = session.get_providers()[0]
            if active_provider != provider:
                logger.warning(
                    f"Requested execution provider '{provider}', but ONNX Runtime is using '{active_provider}'."
                )
            else:
                logger.debug(f"Loaded '{filename}' successfully on '{active_provider}'.")
                
            self.ort_sessions.append(session)

        # Preprocessing Configs
        self.input_processor = InputProcessor()
        self.process_res = 504
        self.process_res_method = "upper_bound_resize"
        self.align_to_input_ext_scale = align_to_input_ext_scale
        self.ransac_view_thresh = ransac_view_thresh
    
    def estimate_depth(self, images: list[np.ndarray | str], extrinsics: list[np.ndarray] | None = None, intrinsics: list[np.ndarray] | None = None) -> DepthPrediction:

        np_inputs = self._preprocess(images=images, extrinsics=extrinsics, intrinsics=intrinsics)

        # Branch based on number of loaded models
        if len(self.ort_sessions) == 1:
            raw_output = self._run_inference(self.ort_sessions[0], np_inputs)
            depth, conf, sky, pred_ext, pred_int = self._extract_outputs(raw_output)
            
            # Metric Scaling (Only triggers if variant == "metric")
            depth = self._apply_metric_scaling(depth, np_inputs.get("original_intrinsics"))
            
        elif len(self.ort_sessions) == 2:
            raw_main = self._run_inference(self.ort_sessions[0], np_inputs)
            raw_metric = self._run_inference(self.ort_sessions[1], np_inputs)
            
            main_depth, main_conf, _, pred_ext, pred_int = self._extract_outputs(raw_main)
            metric_depth, _, metric_sky, _, _ = self._extract_outputs(raw_metric)
            
            # Nested Alignment (Applies internal metric scaling natively)
            depth, scale = align_nested_depth_np(
                main_depth=main_depth,
                main_conf=main_conf,
                metric_depth=metric_depth,
                metric_sky=metric_sky,
                intrinsics=pred_int
            )

            sky = metric_sky  # Use the metric sky output for final processing (Only returned by the metric/mono model)
            conf = main_conf  # Use the main model's confidence for final output
            
            # Scale extrinsics if they exist
            if pred_ext is not None:
                pred_ext[:, :3, 3] *= scale
        
        # Process Sky Output
        depth, conf = process_mono_sky_estimation_np(depth, conf, sky)

        # Align prediction to original extrinsics using Umeyama
        orig_extrinsics = np_inputs.get("original_extrinsics")
        depth, pred_ext = self._align_prediction_extrinsics(depth=depth, pred_extrinsics=pred_ext, original_extrinsics=orig_extrinsics)

        return DepthPrediction(
            depth=depth,
            conf=conf,
            extrinsics=pred_ext,
            intrinsics=np_inputs.get("original_intrinsics") if np_inputs.get("original_intrinsics") is not None else pred_int,
            depth_type="metric" if "metric" in self.model_variant else "relative"
        )

    def _preprocess(self, images: list[np.ndarray | str], extrinsics: np.ndarray | None = None, intrinsics: np.ndarray | None = None,) -> dict[str, np.ndarray | None]:
        """
        NumPy Pre-processing.
        Outputs a dictionary of NumPy arrays ready for model inference.
        """
        # Base processing (Returns NumPy arrays due to our previous changes)
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
                                                            images, 
                                                            extrinsics.copy() if extrinsics is not None else None,
                                                            intrinsics.copy() if intrinsics is not None else None,
                                                            self.process_res,
                                                            self.process_res_method,
                                                            sequential=True
                                                        )

        # Prepare inputs for ONNX Runtime
        imgs = np.expand_dims(imgs_cpu, axis=0).astype(np.float32)  # (N, 3, H, W) -> (1, N, 3, H, W)
        ex_t = np.expand_dims(extrinsics, axis=0).astype(np.float32) if extrinsics is not None else None
        in_t = np.expand_dims(intrinsics, axis=0).astype(np.float32) if intrinsics is not None else None

        # Normalize extrinsics
        ex_t_norm = normalize_extrinsics(ex_t.copy() if ex_t is not None else None)

        B, N = imgs.shape[0], imgs.shape[1]

        if ex_t_norm is None:
            ex_t_norm = np.full((B, N, 4, 4), -1.0, dtype=np.float32)

        if in_t is None:
            in_t = np.full((B, N, 3, 3), -1.0, dtype=np.float32)

        return {
            "imgs": imgs,
            "ex_t_norm": ex_t_norm,
            "in_t": in_t,
            "imgs_cpu": imgs_cpu,  # Preserved for post-processing visualization
            "original_extrinsics": extrinsics,
            "original_intrinsics": intrinsics
        }

    def _run_inference(self, session: ort.InferenceSession, np_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Run ONNX inference using the preprocessed NumPy inputs.
        """
        onnx_feed = {
            "image": np_inputs["imgs"].astype(np.float32),
            "extrinsics_in": np_inputs["ex_t_norm"].astype(np.float32),
            "intrinsics_in": np_inputs["in_t"].astype(np.float32)
        }

        # Execute ONNX Inference
        onnx_outputs = session.run(None, onnx_feed)
        
        output_names = [o.name for o in session.get_outputs()]
        return {name: val for name, val in zip(output_names, onnx_outputs)}

    def _extract_outputs(self, raw_output: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Extract and squeeze the outputs from the raw ONNX outputs.
        """
        depth = np.squeeze(raw_output["depth"], axis=0)
        conf = raw_output.get("depth_conf", None)
        if conf is not None and conf.size > 0:
            conf = np.squeeze(conf, axis=0)
            if conf.ndim > depth.ndim:  # Handle (N, 1, H, W) -> (N, H, W)
                conf = np.squeeze(conf, axis=1)
        else:
            conf = None

        sky = raw_output.get("sky", None)
        if sky is not None and sky.size > 0:
            sky = np.squeeze(sky, axis=0)
            if sky.ndim > depth.ndim:  # Handle (N, 1, H, W) -> (N, H, W)
                sky = np.squeeze(sky, axis=1)
        else:
            sky = None

        extrinsics = raw_output.get("extrinsics_out", None)
        if extrinsics is not None and extrinsics.size > 0 and extrinsics.flat[0] != -1.0:
            extrinsics = np.squeeze(extrinsics, axis=0)  # (N, 4, 4)

        intrinsics = raw_output.get("intrinsics_out", None)
        if intrinsics is not None and intrinsics.size > 0 and intrinsics.flat[0] != -1.0:
            intrinsics = np.squeeze(intrinsics, axis=0)  # (N, 3, 3)

        return depth, conf, sky, extrinsics, intrinsics
    
    def _align_prediction_extrinsics(self, depth: np.ndarray, pred_extrinsics: np.ndarray, original_extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Align the extrinsics of the prediction to the original extrinsics using Umeyama alignment.
        """
        if original_extrinsics is None or pred_extrinsics is None:
            return depth, pred_extrinsics

        _, _, scale, aligned_extrinsics = align_poses_umeyama(
                                                            pred_extrinsics,
                                                            original_extrinsics,
                                                            ransac=len(original_extrinsics) >= self.ransac_view_thresh,
                                                            return_aligned=True,
                                                            random_state=42,
                                                        )

        if self.align_to_input_ext_scale:
            pred_extrinsics = original_extrinsics[..., :3, :]
            depth /= scale
        else:
            pred_extrinsics = aligned_extrinsics

        return depth, pred_extrinsics
    
    def _apply_metric_scaling(self, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray: 
        """
        Apply metric scaling to the depth map using the provided intrinsics.
        """
        if self.model_variant != "metric":
            return depth  # No scaling needed for non-metric variants

        if intrinsics is None or intrinsics.shape[-2:] != (3, 3):
            logger.warning("WARNING: Intrinsics missing or invalid. DA3METRIC-LARGE requires (N,3,3) intrinsics for meters. Returning unscaled depth.")
            return depth

        # Calculate average focal length: (fx + fy) / 2
        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        focal = (fx + fy) / 2.0
        
        # Reshape focal to broadcast over (N, H, W) depth map
        focal = focal.reshape(-1, 1, 1) 
        
        # Scale depth to meters
        return focal * depth / 300.0

    def visualize(self, depth: np.ndarray) -> np.ndarray:
        """
        Visualize depth map using a colormap.
        """
        return visualize_depth(depth)
