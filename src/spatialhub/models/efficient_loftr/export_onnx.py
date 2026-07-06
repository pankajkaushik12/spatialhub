import argparse
from src.loftr import LoFTR, full_default_cfg, reparameter
from copy import deepcopy

import onnx
import torch

def load_model(weights_path, device="cpu"):

    _default_cfg = deepcopy(full_default_cfg)
    
    matcher = LoFTR(config=_default_cfg)

    if weights_path is None:
        raise ValueError("weights_path must be provided.")
    
    # Load the weights
    state = torch.load(weights_path, map_location=device, weights_only=False)
    matcher.load_state_dict(state['state_dict'])

    # Reparameterize the model for better performance
    matcher = reparameter(matcher)
    matcher = matcher.eval()

    return matcher

# Create onnx
def export_onnx(matcher, onnx_path = None):

    if onnx_path is None:
        raise ValueError("onnx_path must be provided.")

    # Dummy inputs for ONNX export
    dummy0 = torch.randn(1, 1, 480, 640, dtype=torch.float32)
    dummy1 = torch.randn(1, 1, 480, 640, dtype=torch.float32)

    # Export the model to ONNX format
    with torch.no_grad():
        torch.onnx.export(
                        matcher,
                        (dummy0, dummy1),
                        onnx_path,
                        opset_version=17,
                        input_names=["image0", "image1"],
                        output_names=["mkpts0_f", "mkpts1_f", "mconf"],
                        dynamic_axes={
                            "image0": {2: "height", 3: "width"},
                            "image1": {2: "height", 3: "width"},
                            "mkpts0_f": {0: "num_matches"},
                            "mkpts1_f": {0: "num_matches"},
                            "mconf": {0: "num_matches"},
                        }
                    )
    
    print("ONNX export successful!")

# Check the exported ONNX model
def check_onnx(onnx_path):

    model = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model=model)
        print("The ONNX graph is clean and valid!")
    except onnx.checker.ValidationError as e:
        print(f"Graph validation failed: {e}")

if __name__ == "__main__":

    arg = argparse.ArgumentParser(description="Export EfficientLoFTR model to ONNX format.")
    arg.add_argument("--weights_path", type=str, default="weights/eloftr_outdoor.ckpt", help="Path to the pretrained weights.")
    arg.add_argument("--onnx_path", type=str, default="weights/eloftr_outdoor.onnx", help="Path to save the exported ONNX model.")
    arg.add_argument("--device", type=str, default="cpu", help="Device to use for exporting the model (e.g., 'cpu' or 'cuda').")
    args = arg.parse_args()

    model = load_model(weights_path=args.weights_path, device=args.device)

    export_onnx(model, onnx_path=args.onnx_path)

    check_onnx(onnx_path=args.onnx_path)

