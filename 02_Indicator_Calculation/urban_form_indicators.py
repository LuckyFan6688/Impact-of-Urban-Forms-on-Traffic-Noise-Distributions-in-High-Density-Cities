"""Raster implementations of the 19 urban-form predictors.

Inputs are aligned, single-band arrays. Calculations use a 256 x 256 raster
representation, a land-use validity mask (classes 1--8), boundary-pixel counts,
and skeleton-pixel counts. No physical pixel-length correction is applied.
Land-use shares are fractions, not percentage points. Function names identify
the manuscript abbreviations; the returned fields retain the modelling schema.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.morphology import binary_erosion, skeletonize

LAND_CODES = {"PT": 1, "PG": 2, "PB": 3, "PC": 4,
              "PR": 5, "PI": 6, "PW": 7, "PO": 8}
FIELD_NAMES = {
    "ABH": "mean_height", "GSI": "build_density", "FSI": "FAR_index",
    "BBD": "build_edge_density", "ALSI": "build_lsi",
    "RLI": "road_length_index", "RAD": "road_area_density",
    "RLSI": "road_lsi", "TRLI": "main_road_length_index",
    **{key: f"land_type_{value}" for key, value in LAND_CODES.items()},
    "DLM": "land_mix_entropy", "DFM": "land_mix_norm_0to1"}
INDICATOR_ORDER = ["ABH", "GSI", "FSI", "BBD", "ALSI", "RLI", "RAD",
                   "RLSI", "TRLI", "PC", "PR", "PI", "PT", "PG", "PB",
                   "PW", "PO", "DLM", "DFM"]


def _mask(mask):
    a = np.asarray(mask, dtype=bool)
    if a.ndim != 2:
        raise ValueError("A two-dimensional raster mask is required.")
    return a


def _boundary_count(mask):
    a = _mask(mask)
    return int((a ^ binary_erosion(a)).sum())


def _share(count, valid_mask):
    total = int(_mask(valid_mask).sum())
    return float(count / total) if total else float("nan")


def abh(building_height, building_mask):
    """Mean height over building pixels, in metres (area-weighted)."""
    return float(np.nanmean(np.asarray(building_height)[building_mask])) if np.any(building_mask) else float("nan")


def gsi(building_mask, valid_mask):
    """Building-pixel count divided by valid-pixel count."""
    return _share(_mask(building_mask).sum(), valid_mask)


def fsi(building_height, building_mask, valid_mask, floor_height=3.0):
    """Sum of continuous H/3 storeys clipped to [0,80], divided by valid pixels."""
    if floor_height <= 0:
        raise ValueError("floor_height must be positive.")
    storeys = np.zeros_like(building_height, dtype=np.float32)
    storeys[building_mask] = np.asarray(building_height)[building_mask] / floor_height
    storeys = np.clip(storeys, 0, 80)
    return _share(np.nansum(storeys[building_mask]), valid_mask)


def bbd(building_mask, valid_mask):
    """Boundary-pixel count / valid-pixel count; undefined without buildings."""
    return _share(_boundary_count(building_mask), valid_mask) if np.any(building_mask) else float("nan")


def alsi(building_mask):
    """0.25 times building boundary-pixel count / sqrt(building-pixel count)."""
    area = int(_mask(building_mask).sum())
    return float(0.25 * _boundary_count(building_mask) / np.sqrt(area)) if area else float("nan")


def rli(road_mask, valid_mask):
    """Road skeleton-pixel count divided by valid-pixel count."""
    return _share(int(skeletonize(_mask(road_mask)).sum()), valid_mask)


def rad(road_mask, valid_mask):
    """Road-pixel count divided by valid-pixel count."""
    return _share(_mask(road_mask).sum(), valid_mask)


def rlsi(road_mask):
    """0.25 times road boundary-pixel count / sqrt(road-pixel count)."""
    area = int(_mask(road_mask).sum())
    return float(0.25 * _boundary_count(road_mask) / np.sqrt(area)) if area else float("nan")


def trli(road_mask, road_width, valid_mask, main_width_threshold=15.0):
    """Skeleton pixels of roads with width >=15 m / valid-pixel count."""
    mask = _mask(road_mask) & (np.asarray(road_width) >= main_width_threshold)
    return _share(int(skeletonize(mask).sum()), valid_mask)


def land_share(land_use, valid_mask, class_code):
    """Land-use class fraction among valid class-1--8 pixels."""
    return _share(int(((np.asarray(land_use) == class_code) & valid_mask).sum()), valid_mask)


def pc(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PC"])


def pr(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PR"])


def pi(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PI"])


def pt(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PT"])


def pg(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PG"])


def pb(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PB"])


def pw(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PW"])


def po(land_use, valid_mask):
    return land_share(land_use, valid_mask, LAND_CODES["PO"])


def dlm(land_use, valid_mask):
    """-sum(p * log(p + 1e-12)) over observed valid land-use classes."""
    values = np.asarray(land_use)[valid_mask]
    if values.size == 0:
        return float("nan")
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def dfm(land_use, valid_mask):
    """DLM / log(k), with k the number of observed valid land-use classes.

    For a single class the denominator is zero; the numerical expression is
    preserved and returns -inf. Nonfinite predictors require training-fold
    imputation in the modelling module.
    """
    values = np.asarray(land_use)[valid_mask]
    if values.size == 0:
        return float("nan")
    k = max(1, len(np.unique(values)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(dlm(land_use, valid_mask) / np.log(k))


def _resize(array, size, categorical=False, nearest=False):
    if array is None:
        return None
    a = np.asarray(array, dtype=np.int32 if categorical else np.float32)
    if a.ndim != 2:
        raise ValueError("Provide single-band two-dimensional raster arrays.")
    method = Image.Resampling.NEAREST if categorical or nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(a).resize(tuple(size), method), dtype=a.dtype)


def calculate_urban_form(building_height, road_mask, road_width, land_use,
                         resize_size=(256, 256), floor_height=3.0,
                         minimum_building_height=3.0,
                         main_width_threshold=15.0, abbreviated=False):
    """Return all 19 raster predictors; missing layers yield missing fields.

    Arrays must already cover the same extent and be spatially aligned. Resize
    order is width,height; heights use bilinear interpolation and other layers
    use nearest-neighbour interpolation. No spatial coordinates are required.
    The default layer codes are PT=1,PG=2,PB=3,PC=4,PR=5,PI=6,PW=7,PO=8.
    """
    if len(resize_size) != 2 or any(int(x) <= 0 for x in resize_size):
        raise ValueError("resize_size must contain two positive dimensions.")
    heights = _resize(building_height, resize_size)
    roads = _resize(road_mask, resize_size, nearest=True)
    widths = _resize(road_width, resize_size, nearest=True)
    land = _resize(land_use, resize_size, categorical=True)
    result = dict.fromkeys(INDICATOR_ORDER, float("nan"))
    if land is not None:
        valid = np.isin(land, list(LAND_CODES.values()))
        if valid.any():
            if heights is not None:
                buildings = (heights >= minimum_building_height) & valid
                result.update(ABH=abh(heights, buildings), GSI=gsi(buildings, valid),
                              FSI=fsi(heights, buildings, valid, floor_height),
                              BBD=bbd(buildings, valid), ALSI=alsi(buildings))
            if roads is not None:
                roads = (roads > 0) & valid
                result.update(RLI=rli(roads, valid), RAD=rad(roads, valid), RLSI=rlsi(roads))
                result["TRLI"] = (trli(roads, widths, valid, main_width_threshold)
                                  if widths is not None else (0.0 if not roads.any() else float("nan")))
            result.update({name: land_share(land, valid, code) for name, code in LAND_CODES.items()})
            result.update(DLM=dlm(land, valid), DFM=dfm(land, valid))
    return result if abbreviated else {FIELD_NAMES[name]: result[name] for name in INDICATOR_ORDER}
