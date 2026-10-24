"""Within-unit spatial noise statistics; Ln is not a time-exceedance level."""
import numpy as np

def valid_values(values, nodata=None, invalid_floor=-9999.0):
    """Convert pixels to float32 and remove NoData, nonfinite and floor values."""
    a = np.asarray(values, dtype=np.float32).ravel()
    if nodata is not None:
        a = a[a != nodata]
    return a[np.isfinite(a) & (a > invalid_floor)]


def spatial_ln(values, n):
    """Return Q(100-n), with linear percentile interpolation across pixels."""
    if not 0 <= n <= 100:
        raise ValueError("n must be an exceedance percentage in [0, 100]")
    a = valid_values(values)
    if not len(a):
        raise ValueError("No valid pixels")
    return float(np.percentile(a, 100.0 - n, method="linear"))


def spatial_metrics(values):
    """Arithmetic spatial mean, extrema, spatial Ln and Q05..Q95 quantiles."""
    a = valid_values(values)
    if not len(a):
        raise ValueError("No valid pixels")
    out = {"Lave": float(a.mean()), "Lmax": float(a.max()), "Lmin": float(a.min())}
    out.update({f"L{n}": spatial_ln(a, n) for n in range(10, 100, 10)})
    out.update({f"Q{q:02d}": float(np.percentile(a, q, method="linear"))
                for q in range(5, 100, 5)})
    return out


def zonal_raster_metrics(raster_path, polygons, polygon_crs):
    """Optional rasterio adapter with explicit CRS agreement and pixel centers.

    Polygons are GeoJSON-like mappings. The caller supplies an existing raster
    and polygons in its CRS; no implicit reprojection or resampling is performed.
    """
    import rasterio
    from rasterio.mask import mask
    with rasterio.open(raster_path) as raster:
        if raster.crs is None or raster.crs != rasterio.crs.CRS.from_user_input(polygon_crs):
            raise ValueError("Project polygons explicitly to the raster CRS first")
        out = []
        for polygon in polygons:
            crop, _ = mask(raster, [polygon], crop=True, filled=False, all_touched=False)
            vals = valid_values(crop[0].compressed(), nodata=raster.nodata)
            out.append(spatial_metrics(vals) if len(vals) else None)
        return out


# Alias for callers using the module name as the operation name.
noise_indicators = spatial_metrics
