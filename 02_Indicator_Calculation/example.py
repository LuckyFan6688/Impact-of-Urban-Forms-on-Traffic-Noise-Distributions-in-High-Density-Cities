"""Calculate indicators from five synthetic aligned raster layers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from urban_form_indicators import calculate_urban_form
from noise_indicators import spatial_metrics
from attenuation_intervals import interval_features, attenuation_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional directory for synthetic CSV outputs.")
    args = parser.parse_args()
    # Channel order: building height (m), road mask, road width (m), land class,
    # and noise level (dBA). All values are artificial and have no georeference.
    layers = np.load(Path(__file__).with_name("example_data") / "synthetic_layers.npy", allow_pickle=False)
    if layers.shape != (5, 32, 32) or not np.isfinite(layers).all():
        raise ValueError("Expected five finite synthetic layers of 32 x 32 pixels.")
    urban = calculate_urban_form(*layers[:4])
    noise = spatial_metrics(layers[4])
    interval = interval_features(layers[4])
    attenuation = attenuation_targets(interval["Lmax"], interval["L_nHM"],
                                      interval["L_nML"], interval["Lmin"])
    print("Synthetic example only: 19 urban-form, spatial noise and attenuation indicators calculated.")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([urban]).to_csv(args.output / "synthetic_urban_indicators.csv", index=False)
        pd.DataFrame([noise]).to_csv(args.output / "synthetic_noise_metrics.csv", index=False)
        pd.DataFrame([{**interval, **attenuation}]).to_csv(args.output / "synthetic_attenuation.csv", index=False)


if __name__ == "__main__":
    main()
