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

def compute_ray_tracing_fast(mesh: pv.PolyData, res_x: int, res_y: int):
    """
    Batched, ortográfico (-Z) usando multi_ray_trace.
    Devuelve mismos campos que tu versión, con hits ya filtrados.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    x_width = abs(xmax - xmin)
    y_width = abs(ymax - ymin)
    z_width = abs(zmax - zmin)
    x_range = np.linspace(xmin - x_width*0.1, xmax + x_width*0.1, res_x, dtype=np.float64)
    y_range = np.linspace(ymin - y_width*0.1, ymax + y_width*0.1, res_y, dtype=np.float64)
    z0 = zmax + z_width * 0.6

    pixel_width  = (x_range[-1] - x_range[0]) / (res_x - 1) if res_x > 1 else (xmax - xmin)
    pixel_height = (y_range[-1] - y_range[0]) / (res_y - 1) if res_y > 1 else (ymax - ymin)
    px_area = pixel_height * pixel_width
    # centros de píxel
    XX, YY = np.meshgrid(x_range, y_range, indexing="xy")
    XX = XX + (0.5 * pixel_width  if res_x > 1 else 0.0)
    YY = YY + (0.5 * pixel_height if res_y > 1 else 0.0)
    ZZ     = np.full_like(XX, z0)
    ZZ_end = np.full_like(XX, zmin)

    starts = np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel())).astype(np.float64)
    dirs   = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (starts.shape[0], 1))
    ends   = np.column_stack((XX.ravel(), YY.ravel(), ZZ_end.ravel())).astype(np.float64)

    out = mesh.multi_ray_trace(starts, dirs, first_point=True)
    if len(out) == 3:
        points, ray_ids, cell_ids = out
    else:
        points, ray_ids = out
        cell_ids = None

    points  = np.asarray(points, dtype=np.float64)
    ray_ids = np.asarray(ray_ids, dtype=np.int32)
    if cell_ids is not None:
        cell_ids = np.asarray(cell_ids, dtype=np.int32)

    #t = np.einsum("ij,ij->i", points - starts[ray_ids], dirs[ray_ids])
    #mask = (t >= 0.0)

    # 2) back-face culling simple para rayos en -Z: n_z > 0
    # if cell_ids is not None:
    #     try:
    #         nz = mesh.face_normals[cell_ids, 2]
    #         mask &= (nz > 0.0)
    #     except Exception:
    #         pass
    #
    hit_points = points#[mask]
    # ray_ids    = ray_ids[mask]
    # cell_ids   = cell_ids[mask] if cell_ids is not None else None

    # === Face normals per-hit (vectorized) ===
    if cell_ids is None:
        raise RuntimeError("cell_ids is None; need cell_ids to map hits -> face normals.")
    n = mesh.cell_normals[cell_ids]  # (N,3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)

    ray_dir = dirs[0]
    # cos(theta) between face normal and ray direction
    cos_th = -n @ ray_dir
    # (N,)
    # Safety mask (should already be <0 from culling)
    #mask = cos_th < 1e-9
    #n = n[mask]
    #cos_th = -cos_th[mask]
    #hit_points = hit_points[mask]

    return {
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
        "ray_starts": starts,
        "ray_dirs": dirs,
        "ray_ends": ends,
        "cos_th": cos_th,
        "cell_normal": n
    }

if __name__ == '__main__':
    pass
