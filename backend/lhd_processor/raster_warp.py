"""
Mosaic / crop / reproject helper (rasterio + numpy only).

Replaces the handful of ``gdal.Warp`` calls so the pipeline does not need the
system GDAL Python bindings (whose Homebrew build links a second copy of Arrow
and collides with pyarrow's).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


def warp_mosaic(
    src_paths: Sequence[str],
    dst_path: str,
    *,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    bounds_crs=None,
    dst_crs=None,
    size: Optional[Tuple[int, int]] = None,
    resampling: Resampling = Resampling.bilinear,
    compress: str = "LZW",
) -> str:
    """
    Mosaic ``src_paths`` (later files overwrite earlier ones where they overlap)
    onto a single GeoTIFF.

    bounds      (left, bottom, right, top) of the output; defaults to the union
                of the inputs.
    bounds_crs  CRS of ``bounds`` (default: the output CRS).
    dst_crs     output CRS (default: CRS of the first input).
    size        (width, height) in pixels; default keeps the first input's
                resolution and fits it to ``bounds``.
    """
    srcs = [rasterio.open(p) for p in src_paths]
    try:
        first = srcs[0]
        out_crs = CRS.from_user_input(dst_crs) if dst_crs is not None else first.crs

        if bounds is None:
            parts = [transform_bounds(s.crs, out_crs, *s.bounds) for s in srcs]
            bounds = (
                min(p[0] for p in parts), min(p[1] for p in parts),
                max(p[2] for p in parts), max(p[3] for p in parts),
            )
        elif bounds_crs is not None and CRS.from_user_input(bounds_crs) != out_crs:
            bounds = transform_bounds(bounds_crs, out_crs, *bounds)
        left, bottom, right, top = bounds

        if size is not None:
            width, height = size
        else:
            if first.crs == out_crs:
                xres, yres = first.res
            else:
                t, _, _ = calculate_default_transform(
                    first.crs, out_crs, first.width, first.height, *first.bounds
                )
                xres, yres = abs(t.a), abs(t.e)
            # Snap outward to the first input's pixel grid so square source
            # pixels stay square (and, in the source CRS, are copied rather
            # than resampled). Across CRSs there is no grid to align to.
            if first.crs == out_crs:
                ax, ay = first.transform.c, first.transform.f
                eps = 1e-6
                left = ax + math.floor((left - ax) / xres + eps) * xres
                right = ax + math.ceil((right - ax) / xres - eps) * xres
                top = ay - math.floor((ay - top) / yres + eps) * yres
                bottom = ay - math.ceil((ay - bottom) / yres - eps) * yres
            width = max(1, int(round((right - left) / xres)))
            height = max(1, int(round((top - bottom) / yres)))
        transform = from_bounds(left, bottom, right, top, width, height)

        nodata = first.nodata
        fill = nodata if nodata is not None else 0
        data = np.full((first.count, height, width), fill, dtype=first.dtypes[0])
        for s in srcs:
            reproject(
                source=rasterio.band(s, list(range(1, s.count + 1))),
                destination=data,
                dst_transform=transform,
                dst_crs=out_crs,
                src_nodata=s.nodata,
                dst_nodata=nodata,
                resampling=resampling,
                init_dest_nodata=False,
            )

        profile = {
            "driver": "GTiff", "height": height, "width": width,
            "count": first.count, "dtype": first.dtypes[0], "crs": out_crs,
            "transform": transform, "nodata": nodata, "compress": compress,
        }
    finally:
        for s in srcs:
            s.close()

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(dst_path), "w", **profile) as dst:
        dst.write(data)
    return str(dst_path)
