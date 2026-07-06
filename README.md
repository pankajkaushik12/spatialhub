# spatialhub

A lightweight, PyTorch-free computer vision hub for Python.

`spatialhub` is built for production and fast execution. By removing heavy PyTorch code and using ONNX Runtime, it gives you reliable, server-safe AI models with automatic downloads and zero memory leaks.

## Why We Built This
Modern AI research is amazing, but running it in production can be hard due to heavy setups, GPU issues, and high memory usage. `spatialhub` fixes this by offering:
* **No PyTorch Needed:** Models are converted to ONNX so they run much faster and are easier to install.
* **Auto-Downloading:** It automatically downloads model files from Hugging Face so your project folder stays clean.
* **Server-Safe:** Built with `opencv-python-headless` so drawing images will never crash your servers or Docker containers.
* **Low Memory:** It only loads images into memory when you actually ask to draw and save them.

## Available Models
Check the specific guides for our current models:

* [**EfficientLoFTR**](./src/spatialhub/models/efficient_loftr/README.md) - Fast and accurate local feature matching.

## Roadmap: What's Next
In the future, `spatialhub` will expand to include more computer vision tools built for real-world use:
* **[Planned]** Depth Estimation (e.g., DepthAnything v2)
* **[Planned]** Pose Estimation & Object Tracking
* **[Planned]** Point Cloud Tools

## Installation

Install via pip (requires Python 3.12+):

```bash
pip install spatialhub

To enable GPU acceleration, install the optional GPU dependencies:
```bash
pip install "spatialhub[gpu]"
```

# Universal Quickstart
All models in spatialhub share a unified, predictable API structure.
```python
from spatialhub import EfficientLoFTR # Or any future model

def main():
    # Initialize the adapter (weights auto-download on first run)
    model = EfficientLoFTR()
    
    # Run inference on raw paths or numpy arrays
    result = model.match("image1.jpg", "image2.jpg", max_dim=1024)

    # Utilize lazy visualization (safely loads and renders without memory bloat)
    result.visualize(top_k=50, save_path="output.png")

if __name__ == "__main__":
    main()
```

# License
The core spatialhub library is licensed under the Apache 2.0 License.

`Note`: Individual model architectures and weights downloaded via the library inherit the licenses of their original authors. Please check the specific model documentation directories for details.

