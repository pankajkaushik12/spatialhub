# Depth Anything 3 (ONNX Engine for SpatialHub)

This module integrates **Depth Anything 3 (DA3)** into the **SpatialHub** library, providing a unified, high-performance, and lightweight inference engine without a PyTorch runtime dependency.

By decoupling the pipeline from deep learning frameworks and re-implementing pre- and post-processing in pure **NumPy** and **OpenCV**, this adapter enables efficient inference for monocular relative depth, metric depth, multi-view camera pose estimation, and nested depth alignment.

---

## 🛠️ Implementation Details

Porting Depth Anything 3 to ONNX Runtime required more than exporting the model graph. Several components from the original implementation were adapted or reimplemented to support framework-agnostic inference. The main implementation changes include:

### 1. Removing PyTorch Runtime Dependencies
* **Pure NumPy & OpenCV Image Preprocessing:** Replaced all `torchvision.transforms` and `PIL` dependencies. Migrated image ingestion, RGB normalization, aspect-ratio scaling, and patch-divisibility transformations to clean, thread-pooled vector operations using OpenCV and NumPy.
* **Vectorized SVD-based Trajectory Alignment:** Replaced the heavy external evaluation library `evo` (and its `PosePath3D` SE(3)/Sim(3) alignment solvers). Implemented a NumPy version of the **Umeyama Sim(3) similarity transformation** using Singular Value Decomposition (`np.linalg.svd`) to align predicted camera trajectories to ground-truth references, complete with an optional robust **RANSAC** outlier filter.

### 2. Resolving ONNX Export Issues
* **Symbolic division failures (`int_truediv`):** Token preparation in the transformer backbone (`prepare_tokens_with_masks`) originally used `einops.rearrange`, which crashed during dynamic view count tracing. Replaced these with native PyTorch `.reshape()` calls to produce clean symbolic graphs.
* **Aspect Ratio Dynamic Scaling:** In DPT/DualDPT heads, computing `aspect_ratio = W / H` using symbolic integers broke the SymPy compiler because it could not mathematically guarantee that $H \neq 0$. Bypassed this by casting dimensions to float tensors before division (`W_t / H_t`).
* **Dynamic Coordinate Grids:** Replaced static coordinate grid generation (`torch.linspace`) inside the DPT head with a custom `dynamic_linspace` function driven by `torch.arange`, preventing the tracer from baking a fixed resolution size into the ONNX graph.
* **Control Flow & In-Place Mutations:** Eliminated dynamic Python shape branch guards (e.g., `if S <= 1`) and in-place tensor slice assignments (e.g., `reorder_indices[:, 0] = b_idx`) in reference view selection, replacing them with functional, trace-safe `torch.where` and `torch.scatter` constructs.
* **2D RoPE Cache Bypassing:** Disabled dynamic sequence length checks and assert statements in `rope.py`. Implemented a pre-computed frequency lookup table with a hardcoded sequence length ceiling (e.g., 4096) to bypass dynamic token caching crashes during export.
* **Giant-Model `xformers` Workaround:** The 1.4-Billion parameter `DA3-GIANT` uses SwiGLU feed-forward networks, which by default route through `xformers` custom C++ kernels, triggering storage pointer capture exceptions (`FakeTensor` / `data_ptr()` crashes). During export, the `xformers` module was temporarily masked from the Python module cache, allowing the exporter to fall back to the mathematically equivalent native PyTorch SwiGLUFFN implementation without affecting ONNX inference performance.

---

## 📌 Supported Inference Pipelines

Depth Anything 3 consists of two distinct model categories: **Monocular Single-View** (relative and metric scaling) and **Multi-View Poses** (incorporating camera encoders and pose decoders). This adapter maps all behaviors to a single unified class.

### 1. Monocular Single-Image Relative Depth (`da3mono_large`)
* **Returns:** A continuous, relative depth map representing spatial boundaries. To prevent clipping artifacts on horizons, the adapter runs a pure NumPy post-processing step (`process_mono_sky_estimation_np`) which identifies sky regions using a predicted sky mask, calculates the 99th percentile depth of non-sky regions, and safely forces the sky pixels to that maximum distance.
* **Required Inputs:** 
  * `images`: A list of image file paths or raw `np.ndarray` objects.

### 2. Monocular Single-Image Metric Depth (`da3metric_large`)
* **Returns:** A depth map representing physical distances in **real-world meters**.
* **Implementation Note:** To reduce storage requirements, the monocular metric pipeline **shares the exact same ONNX weights** as the monocular relative (`da3mono_large.onnx`) model. The adapter transparently reuses the same ONNX weights.
* **Mathematical Scaling:** During post-processing, the adapter computes the average focal length from the camera intrinsics: 
  $$f_{\text{avg}} = \frac{f_x + f_y}{2}$$
  The raw relative depth is then scaled to meters via:
  $$\text{depth}_{\text{metric}} = \frac{f_{\text{avg}} \times \text{depth}_{\text{raw}}}{300.0}$$
* **Required Inputs:**
  * `images`: A list of image file paths or raw `np.ndarray` objects.
  * `intrinsics`: Camera intrinsics matrix $K$ of shape $(N, 3, 3)$. Required for calculating the focal length.

### 3. Multi-View Depth & Camera Pose Estimation (`da3_small` / `da3_base` / `da3_large` / `da3_giant`)
* **Returns:** Multi-view depth maps, a depth confidence map (`depth_conf`), and estimated camera poses (extrinsics and intrinsics). If ground-truth reference camera poses are supplied, the adapter automatically scales, rotates, and translates predicted poses to match your coordinate system using the vectorized Umeyama algorithm.
* **Required Inputs:**
  * `images`: A list of $N$ images.
  * `extrinsics` (Optional): Reference camera extrinsics of shape $(N, 4, 4)$.
  * `intrinsics` (Optional): Reference camera intrinsics of shape $(N, 3, 3)$.

### 4. Advanced Dual-Model Nested Pipeline
* **Returns:** Real-world metric depth maps that combine the high-fidelity relative details of any Any-view model (e.g., `da3_giant`) with the stable, metric scale of a Monocular Metric model (`da3metric_large`).
* **NumPy Least-Squares Alignment:** Instead of baking two massive models into a single, bloated ONNX file (which would crash the ONNX tracer during dynamic slicing), SpatialHub runs inference on both models independently, then uses a **pure NumPy least-squares alignment solver** (`align_nested_depth_np`) to calculate the optimal scaling factor ($s$) between their outputs. The scale factor is then used to align the final depth map and predicted camera extrinsics.
* **Model Nesting Rules:** You can nest **any** Any-view model (e.g., `da3_small`, `da3_base`, `da3_large`, `da3_giant`) with the Monocular Metric model (`da3metric_large`).
* **Required Inputs:**
  * `images`: A list of image file paths or raw `np.ndarray` objects.
  * `intrinsics` (Optional): Camera intrinsics matrix $K$ of shape $(N, 3, 3)$.

---

## ⚙️ API Reference

### `DepthAnything3` Class

```python
class DepthAnything3:
    def __init__(
        self, 
        model_name: str | list[str] = "da3_base", 
        model_variant: str | None = None, 
        provider: str = "CPUExecutionProvider", 
        align_to_input_ext_scale: bool = True, 
        ransac_view_thresh: int = 10
    )
```
Initializes the ONNX engine, manages model-caching directories, resolves local vs. remote weights, and boots the ONNX Runtime session.

* **Parameters:**
  * `model_name` (str | list[str]): Name of the preset (e.g., `"da3_base"`, `"da3mono_large"`, `"da3metric_large"`) or a list of two presets for nested models (e.g. `["da3_base", "da3metric_large"]`). Can also accept a direct absolute path to a local `.onnx` model file.
  * `model_variant` (str | None): Explicitly set the pipeline variant (`"relative"`, `"metric"`, or `"metric_nested"`). If `None`, it is automatically detected.
  * `provider` (str): ONNX Runtime execution provider (e.g., `"CUDAExecutionProvider"`, `"CPUExecutionProvider"`).
  * `align_to_input_ext_scale` (bool): If `True`, aligns predicted depth scale and extrinsics to the scale of user-provided input extrinsics. Default is `True`.
  * `ransac_view_thresh` (int): Minimum view count required to activate RANSAC filtering in Umeyama alignment. Default is 10.

---

### `estimate_depth` Method

```python
def estimate_depth(
    self, 
    images: list[np.ndarray | str], 
    extrinsics: list[np.ndarray] | None = None, 
    intrinsics: list[np.ndarray] | None = None
) -> DepthPrediction
```
The unified entrypoint for running depth estimation. It automatically handles single-model inference, monocular metric scaling, and multi-model nested alignment depending on how the class was initialized.

* **Parameters:**
  * `images` (list): Image paths or NumPy arrays.
  * `extrinsics` (Optional): Ground-truth extrinsics matrices of shape $(N, 4, 4)$.
  * `intrinsics` (Optional): Ground-truth intrinsics matrices of shape $(N, 3, 3)$.
* **Returns:**
  * `DepthPrediction`: A structured dataclass holding:
    * `depth` (`np.ndarray` of shape $[N, H, W]$): Post-processed depth maps.
    * `conf` (`np.ndarray` of shape $[N, H, W]$ or `None`): Confidence scores $[0, 1]$.
    * `extrinsics` (`np.ndarray` of shape $[N, 4, 4]$ or `None`): Camera poses.
    * `intrinsics` (`np.ndarray` of shape $[N, 3, 3]$ or `None`): Intrinsics matrices.
    * `depth_type` (str): `"relative"` or `"metric"`.

---

## 💻 Usage Examples

### Example 1: Single-Image Relative Depth (Base Model)
```python
import cv2
from spatialhub import DepthAnything3

# 1. Initialize the engine (downloads model from Hugging Face automatically on first run)
depth_estimator = DepthAnything3(model_name="da3_base")

# 2. Run depth estimation
result = depth_estimator.estimate_depth(images=["assets/example.png"])

# 3. Visualize using the built-in colormap utility
depth_colored = depth_estimator.visualize(result.depth[0])
cv2.imwrite("depth_relative.png", depth_colored)
```

### Example 2: Single-Image Metric Depth in Meters
```python
import numpy as np
from spatialhub import DepthAnything3

depth_estimator = DepthAnything3(model_name="da3metric_large")

# Provide camera intrinsics to compute real-world metric scale
K = np.array([
    [520.0,   0.0, 252.0],
    [  0.0, 520.0, 252.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

result = depth_estimator.estimate_depth(
    images=["assets/example.png"],
    intrinsics=[K]
)

# Output depths are in physical meters
print("Depth at center pixel (meters):", result.depth[0, 252, 252])
```

### Example 3: Multi-View Depth + Camera Pose Estimation on GPU
```python
import numpy as np
from spatialhub import DepthAnything3

depth_estimator = DepthAnything3(
    model_name="da3_large",
    provider="CUDAExecutionProvider"
)

# A sequence of multi-view camera frames
image_paths = ["frame_00.png", "frame_01.png", "frame_02.png"]

result = depth_estimator.estimate_depth(images=image_paths)

print("Depth shape:", result.depth.shape)         # (3, H, W)
print("Estimated camera poses:", result.extrinsics.shape)  # (3, 4, 4)
```

### Example 4: Dual-Model Nested Pipeline
```python
from spatialhub import DepthAnything3

# Load an Any-view model (da3_base) for details + Metric model (da3metric_large) for metric alignment scale
depth_estimator = DepthAnything3(model_name=["da3_base", "da3metric_large"])

image_paths = [
    "src/spatialhub/models/depth_anything_3/assets/examples/SOH/000.png",
    "src/spatialhub/models/depth_anything_3/assets/examples/SOH/010.png"
]

result = depth_estimator.estimate_depth(images=image_paths)

print("Nested alignment metric depth shape:", result.depth.shape)
print("Aligned camera poses in meters:", result.extrinsics.shape)
```


## Known Limitations
- Training is not supported.
- ONNX Runtime is required.
- Giant model requires the xformers workaround during export.
- Metric depth requires camera intrinsics.

## 📝 Citation & Acknowledgments

`spatialhub` provides an inference adapter and does not claim ownership of the underlying model architecture or trained parameters. 

If you use Depth Anything 3 in your academic work or commercial projects, please cite the original paper and credit the authors:

Project page: [DepthAnything3](https://depth-anything-3.github.io/)

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the Visual Space from Any Views},
  author={Lin, Haotong and Chen, Sili and Liew, Jun Hao and Chen, Donny Y. and Li, Zhenyu and Shi, Guang and Feng, Jiashi and Kang, Bingyi},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}

