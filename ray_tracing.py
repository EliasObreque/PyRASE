"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""
import numpy as np
import pyvista as pv

def compute_ray_tracing(mesh: pv.PolyData, res_x, res_y):
    hit_points = []
    ray_starts = []
    ray_ends = []
    bounds = mesh.bounds

    bounds_lim = [np.min(bounds), np.max(bounds)]
    x_range = np.linspace(bounds_lim[0] - 10, bounds_lim[1] + 10, res_x)
    y_range = x_range

    z_start = bounds[5] + 50
    pixel_width = abs(x_range[-1] - x_range[0]) / res_x  # in mm
    pixel_height = abs(y_range[-1] - y_range[0]) / res_y

    for y in y_range[:-1]:
        for x in x_range[:-1]:
            start = [x + pixel_width / 2, y + pixel_height / 2, z_start]  # Start above the mesh
            end = [x + pixel_width / 2, y + pixel_height / 2, z_start - 3 * z_start]  # End point
            ray_starts.append(start)
            ray_ends.append(end)
            # Perform ray trace
            points, _ = mesh.ray_trace(start, end, first_point=True)

            if len(points) == 3:
                points = points.reshape(-1, 3)
                hit_points.append(points[0])  # store first hit point
            else:
                hit_points.append(None)  # No intersection

    hit_points = [hit for hit in hit_points if hit is not None]  # gets rid of missed rays
    hit_points = [tuple(coord) for coord in hit_points]  # keeps only coords

    properties = {'x_range': x_range, 'y_range': y_range, 'z_start': z_start, 'pixel_width': pixel_width,
                  'pixel_height': pixel_height, 'hit_points': hit_points, 'res_x': res_x, 'res_y': res_y,
                  'ray_starts': ray_starts, 'ray_ends': ray_ends}
    return properties

def compute_ray_tracing_fast(mesh: pv.PolyData, res_x: int, res_y: int, margin: float = 10.0, z_lift: float = 50.0):
    """
    Batched, orthographic ray casting using PyVista's multi_ray_trace.
    - mesh: pv.PolyData (triangulated) in *mm* (PyVista default)
    - res_x, res_y: grid resolution on the image plane
    - margin: extra padding (mm) around the XY mesh bounds
    - z_lift: height above the mesh max-Z to place the image plane (mm)
    Returns:
      dict with hit points (N,3), ray_ids (N,), cell_ids (N,), pixel size, and grids.
    """
    # Bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # Uniform XY ranges (with padding)
    x_range = np.linspace(xmin - margin, xmax + margin, res_x, dtype=np.float64)
    y_range = np.linspace(ymin - margin, ymax + margin, res_y, dtype=np.float64)

    # Image plane Z above the mesh
    z0 = zmax + z_lift

    # Pixel size (mm)
    pixel_width  = (x_range[-1] - x_range[0]) / (res_x - 1) if res_x > 1 else (xmax - xmin)
    pixel_height = (y_range[-1] - y_range[0]) / (res_y - 1) if res_y > 1 else (ymax - ymin)

    # Build center-of-pixel grid (meshgrid is (rows, cols) = (y, x))
    xx, yy = np.meshgrid(x_range, y_range, indexing="xy")
    xx = xx + 0.5 * (pixel_width  if res_x > 1 else 0.0)
    yy = yy + 0.5 * (pixel_height if res_y > 1 else 0.0)
    zz = np.full_like(xx, z0)
    zz_end = np.full_like(xx, zmin)
    # Starts (N,3) and directions (N,3). Orthographic rays go straight down -Z.
    starts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64)
    dirs   = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (starts.shape[0], 1))
    ends   = np.column_stack((xx.ravel(), yy.ravel(), zz_end.ravel())).astype(np.float64)
    # Batch ray cast. Depending on PyVista version:
    # multi_ray_trace returns (points, ray_ids, cell_ids) OR (points, ray_ids)
    out = mesh.multi_ray_trace(starts, dirs, first_point=True)

    if len(out) == 3:
        points, ray_ids, cell_ids = out
    else:
        points, ray_ids = out
        cell_ids = None  # not all versions return this

    # Convert to numpy arrays
    points  = np.asarray(points)
    ray_ids = np.asarray(ray_ids, dtype=np.int32)
    if cell_ids is not None:
        cell_ids = np.asarray(cell_ids, dtype=np.int32)

    # Filter valid hits (PyVista only returns hits; missing rays are not present)
    # If you need a dense image, you can reconstruct via ray_ids.
    hit_points = points
    properties = {
        "x_range": x_range,
        "y_range": y_range,
        "z_start": z0,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "hit_points": hit_points,
        "ray_ids": ray_ids,
        "cell_ids": cell_ids,
        "res_x": res_x,
        "res_y": res_y,
        "ray_starts": starts,      # optional (can be large; keep only if you need it)
        "ray_dirs": dirs,          # "
        "ray_ends": ends
    }
    return properties


if __name__ == '__main__':
    pass
