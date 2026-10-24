"""Seven-channel image assembly in the network input order."""
import numpy as np

CHANNEL_ORDER = ("red", "green", "blue", "building_height", "road_distance", "road_mask", "road_width")


def build_seven_channels(rgb, building_height, road_distance, road_mask, road_width):
    """RGB uint8 scale / 255; four auxiliary channels retain their input units."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("RGB must have shape [3,H,W]")
    arrays = [np.asarray(x, dtype=np.float32) for x in (building_height, road_distance, road_mask, road_width)]
    if any(x.shape != rgb.shape[1:] for x in arrays):
        raise ValueError("All channels must be aligned to the same image grid")
    if not np.isfinite(rgb).all() or any(np.isinf(x).any() for x in arrays):
        raise ValueError("Infinite input is invalid; RGB must be finite")
    return np.concatenate([rgb.astype(np.float32) / 255.0] + [np.nan_to_num(x, nan=0)[None] for x in arrays])
