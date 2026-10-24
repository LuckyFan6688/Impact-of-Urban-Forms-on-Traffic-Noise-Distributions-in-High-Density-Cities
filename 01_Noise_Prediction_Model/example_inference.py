"""Test the seven-channel architecture with artificial inputs and random weights.

Inputs: RGB [3,256,256] in 0-255 and four aligned auxiliary channels in
building-height, road-distance, road-mask and road-width order. The sigmoid
output is a normalized network response, not a calibrated sound-level map.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import yaml
import torch
from model_architecture import build_model
from preprocessing import CHANNEL_ORDER, build_seven_channels


def run(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(__file__).resolve().parent
    config = yaml.safe_load((base / "model_config.yaml").read_text("utf-8"))
    expected = {
        "encoder": "efficientnet-b4", "decoder": "UnetPlusPlus",
        "attention": "CBAM", "attention_position": "final_decoder_output",
        "in_channels": 7, "output_channels": 1, "activation": "sigmoid",
    }
    if any(config["model"][key] != value for key, value in expected.items()):
        raise ValueError("Configuration must match the seven-channel B4/U-Net++/CBAM architecture")
    if tuple(config["channels"]) != CHANNEL_ORDER:
        raise ValueError("Configuration channel order does not match input assembly")
    if config["model"]["encoder_initialization_for_example"] != "random":
        raise ValueError("The example requires random initialization")

    rgb = np.load(base / "example_data" / "synthetic_rgb.npy", allow_pickle=False)
    auxiliary = np.load(base / "example_data" / "synthetic_auxiliary.npy", allow_pickle=False)
    channels = build_seven_channels(rgb, *auxiliary)
    shape = (config["preprocessing"]["input_height"], config["preprocessing"]["input_width"])
    if channels.shape != (7, *shape):
        raise ValueError(f"Expected input shape [7,{shape[0]},{shape[1]}]")
    torch.set_num_threads(config["example"]["cpu_threads"])
    torch.manual_seed(config["example"]["seed"])
    model = build_model().eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(channels)[None])[0, 0].cpu().numpy()
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid demonstration output")
    if not ((prediction >= 0).all() and (prediction <= 1).all()):
        raise RuntimeError("Sigmoid output is outside [0,1]")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "synthetic_random_weight_output.npy", prediction, allow_pickle=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    axes[0].imshow(rgb.transpose(1, 2, 0))
    axes[0].set_title("Synthetic RGB")
    axes[1].imshow(auxiliary[0], cmap="viridis")
    axes[1].set_title("Synthetic building height")
    im = axes[2].imshow(prediction, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Random-weight output")
    fig.colorbar(im, ax=axes[2], label="Normalized network response")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("Artificial inputs; untrained architecture test", fontsize=10)
    fig.savefig(output / "synthetic_inference.png", dpi=300,
                metadata={"Software": "Synthetic example"})
    fig.savefig(output / "synthetic_inference.svg",
                metadata={"Date": None, "Creator": "Synthetic example"})
    plt.close(fig)
    record = {
        "data": "synthetic", "weights": "random", "seed": config["example"]["seed"],
        "input_shape": list(channels.shape), "output_shape": list(prediction.shape),
        "output_units": "normalized_network_response", "output_finite": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "paper_validation_results": False,
    }
    (output / "synthetic_inference.json").write_text(json.dumps(record, indent=2), "utf-8")
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/noise_prediction_example"))
    run(parser.parse_args().output)
