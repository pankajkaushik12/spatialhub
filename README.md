# spatialhub

[![PyPI version](https://badge.fury.io/py/spatialhub.svg)](https://badge.fury.io/py/spatialhub)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Repo stars](https://img.shields.io/github/stars/pankajkaushik12/spatialhub?style=social)](https://github.com/pankajkaushik12/spatialhub)

A lightweight computer vision library built around ONNX Runtime.

`spatialhub` is a personal project that collects useful computer vision models behind a simple and consistent Python API. The focus is on lightweight inference, minimal dependencies, and easy deployment.

> **Status:** Work in progress. The library is actively being developed by a single maintainer. APIs may change, and only a small number of models are currently available.

---

## Why this project?

Many computer vision models are easy to use in research but harder to deploy because of large dependencies and different APIs.

The goal of `spatialhub` is to provide a simpler alternative by:

- Using **ONNX Runtime** for inference.
- Avoiding a direct **PyTorch** dependency where possible.
- Downloading model weights automatically.
- Providing a consistent interface across different models.
- Supporting headless environments with `opencv-python-headless`.

The project is still growing, and the feature set will expand over time.

---

## Available Models

Currently supported:

- [**EfficientLoFTR**](./src/spatialhub/models/efficient_loftr/README.md) — Semi-dense local feature matching.

---

## Planned

Some models and utilities planned for future releases include:

- Monocular depth estimation
- Pose estimation
- Object tracking
- Point cloud utilities

These are ideas, not promises, and development depends on available time.

---

## Installation

Requires **Python 3.12+**.

Install from PyPI:

```bash
pip install spatialhub
```

Optional GPU support:

```bash
pip install "spatialhub[gpu]"
```

---

## Quick Start

All models follow a similar API.

```python
from spatialhub import EfficientLoFTR

model = EfficientLoFTR()

result = model.match(
    "image1.jpg",
    "image2.jpg",
    max_dim=1024,
)

result.visualize(
    top_k=50,
    save_path="output.png",
)
```

---

## Project Goals

- Simple Python API
- Lightweight inference
- Consistent model interfaces
- Automatic model weight management
- Easy deployment in scripts, servers, and containers

---

## License

The core `spatialhub` library is licensed under the Apache 2.0 License.

Model architectures and pretrained weights keep the licenses provided by their original authors. See the documentation for each model for licensing details.