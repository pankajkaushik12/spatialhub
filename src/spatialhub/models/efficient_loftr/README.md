# EfficientLoFTR (ONNX Adapter)

This module provides an ONNX Runtime adapter for **EfficientLoFTR**, a local feature matching model for semi-dense feature correspondence.

The adapter runs EfficientLoFTR without requiring PyTorch during inference. It supports multiple ONNX Runtime execution providers, automatic model downloading, and dynamic preprocessing for images of different sizes.

> **Note**
> This adapter provides inference only. Training is not supported.

## Features
- **Dynamic Resolution:** Accepts images of different sizes and aspect ratios by resizing and aligning inputs to the nearest multiple of 32.
- **Lazy Visualization:** Images are only loaded for visualization when `visualize()` is called, reducing unnecessary memory usage.
- **Automatic Model Download:** Downloads the required `.onnx` model from Hugging Face Hub on first use and caches it locally.
- **CPU and GPU Support:** Works with ONNX Runtime execution providers including CPU, CUDA, TensorRT, and ROCm.


## Preprocessing Pipeline
To preserve keypoint coordinates across dynamic resizing and padding operations, the adapter performs the following transformations.

### 1. Dimension Alignment

The Sparse Transformer backbone downsamples feature maps by a factor of **32**. Therefore, the intermediate input dimensions $(W_{curr}, H_{curr})$ are aligned to the nearest multiple of 32:

$$
W_{new} = \max\left(32,\ \left\lfloor\frac{W_{curr}}{32}\right\rfloor \times 32\right)
$$

$$
H_{new} = \max\left(32,\ \left\lfloor\frac{H_{curr}}{32}\right\rfloor \times 32\right)
$$

### 2. Coordinate Projection

Since the input image may be resized during preprocessing, a scale factor vector

$$
S = [S_x, S_y]
$$

is computed to project the predicted keypoints back into the original image coordinate system:

$$
S_x = \frac{W_{orig}}{W_{new}}, \qquad  S_y = \frac{H_{orig}}{H_{new}}
$$

For every keypoint predicted by the ONNX model,

$$
P_{raw} = (x_{raw}, y_{raw}),
$$

the corresponding coordinates in the original image are recovered as

$$
P_{orig} = \left(x_{raw} \cdot S_x,\; y_{raw} \cdot S_y \right)
$$

### 3. Dynamic Padding and Boundary Filtering

When matching two images of different resolutions, both tensors are padded along the bottom and right edges to a shared size:

$$
H_{max} = \max(H_a, H_b), \qquad  W_{max} = \max(W_a, W_b)
$$

To eliminate matches that fall inside the padded regions, the adapter applies a boundary validation mask before returning the final keypoints:

$$
V =
\{
(P_0, P_1)
\mid
x_0 < W_a
\land
y_0 < H_a
\land
x_1 < W_b
\land
y_1 < H_b
\}
$$

## API Reference

### `EfficientLoFTR`
Initializes an EfficientLoFTR inference session using the selected ONNX Runtime execution provider.

```python
from spatialhub import EfficientLoFTR

matcher = EfficientLoFTR(
    provider="CPUExecutionProvider"
)
```

Parameters:
- **`provider`** (`str`): How you want to run the model. Options include:

    | Provider | Description |
    |----------|-------------|
    | `CPUExecutionProvider` | Default CPU execution |
    | `CUDAExecutionProvider` | NVIDIA CUDA |
    | `TensorrtExecutionProvider` | NVIDIA TensorRT |
    | `ROCMExecutionProvider` | AMD ROCm |

### `match`
```python
result = matcher.match(
    image_a, 
    image_b, 
    max_dim=1024
)
```
Matches local features between two input images and returns the results as a `MatchResult`.

**Parameters**:
- **`image_a`** / **`image_b`** (`str | Path | np.ndarray`): The target images. Can be string paths, system paths, or raw NumPy arrays (Grayscale, BGR, or RGBA).
- **`max_dim`** (`int | None`): Caps the largest dimension of the image before preprocessing to prevent Out-Of-Memory (OOM) errors. Defaults to 1024. Passing None runs inference at the raw resolution.

**Returns**:
- `MatchResult`: A structured dataclass holding the matched features.

### `MatchResult`
Represents the output of `match()` and stores the matched keypoints, confidence scores, and references to the original inputs.
- **`image_a`** / **`image_b`** (`str | Path | np.ndarray`): The raw reference image paths or input array tokens.
- **`keypoints_a`** (`np.ndarray of shape [N, 2]`): The verified $[x, y]$ keypoints mapped to the original scale of image_a.
- **`keypoints_b`** (`np.ndarray of shape [N, 2]`): The verified $[x, y]$ keypoints mapped to the original scale of image_b.
- **`confidence`** (`np.ndarray of shape [N]`): The matching confidence scores, ranging from $0.0$ to $1.0$.

### `MatchResult.visualize`
```python
result.visualize(
    conf_thresh=0.5, 
    max_side=800, 
    top_k=None, 
    save_path=None
)
```
Generates a side-by-side visualization of the matched keypoints.

Parameters:
- **`conf_thresh`** (`float`): Drops matches with a confidence score below this threshold. Default is 0.5.
- **`max_side`** (`int`): Caps the longest side of the generated visualization image to preserve screen space. Default is 800.
- **`top_k`** (`int | None`): If specified, automatically filters and displays only the top $k$ highest-confidence matches.
- **`save_path`** (`str | Path | None`): If provided, writes the rendered visualization directly to disk at this path. Safe for headless/Docker environments.

## Usage Examples

### Example 1: GPU Inference
Runs the adapter with GPU acceleration and visualizes the top 50 highest-confidence matches.

```python
from spatialhub import EfficientLoFTR

# Initialize with CUDA Provider
matcher = EfficientLoFTR(provider="CUDAExecutionProvider")

# Perform Match
result = matcher.match("room_left.jpg", "room_right.jpg", max_dim=1280)

# Visualize the 50 most structurally sound matches
result.visualize(top_k=50, save_path="outputs/gpu_matches.png")
```



### Example 2: Headless Environment
Runs EfficientLoFTR in a headless environment such as a Docker container or remote server. The visualization is written directly to disk without opening a GUI window.
```python
import numpy as np
import cv2
from spatialhub import EfficientLoFTR

# Initialize Model (Uses CPU fallback by default)
matcher = EfficientLoFTR()

# Assume we already have pre-loaded OpenCV arrays from a video pipeline
frame_1 = cv2.imread("frame_001.png")
frame_2 = cv2.imread("frame_002.png")

# Run Matching
result = matcher.match(frame_1, frame_2, max_dim=800)

# Render and Save visualization directly without opening a window
result.visualize(
    conf_thresh=0.6, 
    top_k=100, 
    save_path="inference_result.png"
)
```


## Citation

If you use this adapter in your work, please also cite the original EfficientLoFTR paper:

```bibtex
@inproceedings{wang2022efficientloftr,
  title={EfficientLoFTR: Semi-Dense Local Feature Matching with Sparse Transformers},
  author={Wang, Yanzhao and Geng, Yuwei and Jiang, Zheng and Zhao, Yihong and Jin, Shisheng and Lin, Siyu and Han, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

This adapter provides an ONNX Runtime implementation for inference and is not the original training framework.

