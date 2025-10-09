"""
Created by Elias Obreque
Date: 01/10/2025
email: els.obrq@gmail.com
"""
import numpy as np
import pyvista as pv
from scipy.ndimage import label, binary_dilation, generate_binary_structure

from core.monitor import show_ray_tracing_fast

def compute_ray_tracing_fast_optimized(mesh: pv.PolyData, r_source: np.ndarray,
                                       res_x: int, res_y: int, fill_gaps=True):
    """
    Optimized ray tracing with pure NumPy vectorization.

    Parameters:
    -----------
    mesh : pv.PolyData
        Triangulated mesh
    r_source : np.ndarray
        Unit incident direction vector [3,]
    res_x, res_y : int
        Pixel grid resolution
    fill_gaps : bool
        Whether to fill isolated gaps

    Returns:
    --------
    dict with ray tracing information
    """

    # Normalize r_source (vectorized)
    r_source = np.asarray(r_source, dtype=np.float64)
    r_source = r_source / np.linalg.norm(r_source)

    # === 1. Build orthogonal coordinate system (already fast) ===
    if abs(r_source[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    u = np.cross(r_source, arbitrary)
    u = u / np.linalg.norm(u)

    v = np.cross(r_source, u)
    v = v / np.linalg.norm(v)

    # === 2. Determine mesh bounds ===
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.array(mesh.center, dtype=np.float64)

    max_extent = np.sqrt(
        (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
    )

    # === 3. Create pixel plane ===
    plane_distance = max_extent * 1.2
    plane_center = center - r_source * plane_distance
    plane_size = max_extent * 1.1

    u_coords = np.linspace(-plane_size / 2, plane_size / 2, res_x, dtype=np.float64)
    v_coords = np.linspace(-plane_size / 2, plane_size / 2, res_y, dtype=np.float64)

    pixel_width = (u_coords[-1] - u_coords[0]) / (res_x - 1) if res_x > 1 else plane_size
    pixel_height = (v_coords[-1] - v_coords[0]) / (res_y - 1) if res_y > 1 else plane_size
    px_area = pixel_width * pixel_height

    # === 4. Generate ray start points (OPTIMIZED - pure vectorization) ===
    UU, VV = np.meshgrid(u_coords, v_coords, indexing='xy')

    # Shift to pixel centers
    UU += (0.5 * pixel_width if res_x > 1 else 0.0)
    VV += (0.5 * pixel_height if res_y > 1 else 0.0)

    # Flatten
    uu_flat = UU.ravel()
    vv_flat = VV.ravel()

    # Vectorized computation of starts (MUCH faster than loops)
    starts = (plane_center[None, :] +
              uu_flat[:, None] * u[None, :] +
              vv_flat[:, None] * v[None, :]).astype(np.float64)

    # === 5. Ray directions ===
    dirs = np.tile(r_source, (starts.shape[0], 1))

    # === 6. Ray tracing ===
    out = mesh.multi_ray_trace(starts, dirs, first_point=True)

    if len(out) == 3:
        points, ray_ids, cell_ids = out
    else:
        points, ray_ids = out
        cell_ids = None

    points = np.asarray(points, dtype=np.float64)
    ray_ids = np.asarray(ray_ids, dtype=np.int32)

    if cell_ids is not None:
        cell_ids = np.asarray(cell_ids, dtype=np.int32)
    else:
        raise RuntimeError("cell_ids is None")

    # === 7. Calculate normals and cos(theta) (OPTIMIZED - vectorized) ===
    n = mesh.cell_normals[cell_ids]
    # Vectorized normalization
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / norms

    # Vectorized dot product (MUCH faster than loop)
    cos_th = np.einsum('ij,j->i', n, r_source)

    # area projected
    n_proj = mesh.cell_normals
    # Vectorized normalization
    norms = np.linalg.norm(n_proj, axis=1, keepdims=True)
    n_proj = n_proj / norms
    a_fem = mesh.compute_cell_sizes(length=False, area=True, volume=False)['Area']
    a_fem = np.asarray(a_fem, dtype=np.float64)
    cos_th_proj = np.einsum('ij,j->i', n_proj, r_source)
    proj_ids = cos_th_proj <= 0
    area_proj_mesh = np.sum(a_fem[proj_ids] * np.abs(cos_th_proj[proj_ids]))
    # === 8. Filter back-facing triangles (OPTIMIZED - vectorized) ===
    mask = cos_th < -1e-12

    hit_points = points[mask]
    ray_ids_filtered = ray_ids[mask]
    cell_ids_filtered = cell_ids[mask]
    cos_th_filtered = cos_th[mask]
    n_filtered = n[mask]

    # === 9. Package results ===
    res_prop = {
        "x_range": u_coords,
        "y_range": v_coords,
        "plane_center": plane_center,
        "plane_u": u,
        "plane_v": v,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area": px_area,
        "hit_points": hit_points,
        "ray_ids": ray_ids_filtered,
        "cell_ids": cell_ids_filtered,
        'A_fem_proj': area_proj_mesh,
        "res_x": res_x,
        "res_y": res_y,
        "ray_starts": starts,
        "ray_dirs": dirs,
        "r_source": r_source,
        "cos_th": cos_th_filtered,
        "cell_normal": n_filtered
    }

    # === 10. Post-processing (OPTIMIZED) ===
    if fill_gaps:
        res_prop = fill_gaps_by_spatial_interpolation_optimized(res_prop, max_gap_size=10)

    # filtered = filter_edge_artifacts_optimized(
    #     res_prop['hit_points'],
    #     res_prop['cell_normal'],
    #     res_prop['cos_th'],
    #     res_prop['cell_ids'],
    #     res_prop['ray_ids'],
    #     mesh
    # )
    #
    # res_prop['hit_points'] = filtered['hits']
    # res_prop['cell_normal'] = filtered['normals']
    # res_prop['cos_th'] = filtered['cos_th']
    # res_prop['cell_ids'] = filtered['cell_ids']
    # res_prop['ray_ids'] = filtered['ray_ids']

    Area_r = np.abs(px_area / res_prop['cos_th'])
    area_aux = px_area / np.cos(89 * np.pi / 180)
    print(area_aux, 5 * px_area)
    Area_r[Area_r > area_aux] = area_aux
    res_prop['area_proj'] = Area_r
    return res_prop


def filter_edge_artifacts_optimized(hits, cell_normals, cos_th, cell_ids, ray_ids, mesh):
    """Optimized edge artifact filtering with vectorization"""

    mesh_extent = np.max(mesh.bounds) - np.min(mesh.bounds)
    edge_tol = mesh_extent * 0.0001

    bounds = np.array(mesh.bounds, dtype=np.float64).reshape(3, 2)

    # Vectorized boundary detection
    on_boundary_count = np.zeros(len(hits), dtype=np.int32)

    for axis in range(3):
        at_min = np.abs(hits[:, axis] - bounds[axis, 0]) < edge_tol
        at_max = np.abs(hits[:, axis] - bounds[axis, 1]) < edge_tol
        on_boundary_count += (at_min | at_max).astype(np.int32)

    # Keep only hits on exactly 1 face
    on_face_only = on_boundary_count == 1

    n_removed = (~on_face_only).sum()
    print(f"Filtering {n_removed} edge/corner artifacts ({n_removed / len(hits) * 100:.2f}%)")

    return {
        'hits': hits[on_face_only],
        'normals': cell_normals[on_face_only],
        'cos_th': cos_th[on_face_only],
        'cell_ids': cell_ids[on_face_only],
        'ray_ids': ray_ids[on_face_only]
    }


def fill_gaps_by_spatial_interpolation_optimized(res_prop, max_gap_size=2):
    """Optimized gap filling with vectorized projection"""

    hit_points = res_prop['hit_points']
    ray_ids = res_prop['ray_ids']
    cell_ids = res_prop['cell_ids']
    plane_center = res_prop['plane_center']
    plane_u = res_prop['plane_u']
    plane_v = res_prop['plane_v']
    res_x = res_prop['res_x']
    res_y = res_prop['res_y']
    cell_normals = res_prop['cell_normal']
    cos_th = res_prop['cos_th']
    x_range = res_prop['x_range']
    y_range = res_prop['y_range']

    # Vectorized projection (MUCH faster)
    relative_pos = hit_points - plane_center

    # Vectorized dot products
    u_coords = np.dot(relative_pos, plane_u)
    v_coords = np.dot(relative_pos, plane_v)

    # Vectorized searchsorted
    i_indices = np.searchsorted(y_range, v_coords) - 1
    j_indices = np.searchsorted(x_range, u_coords) - 1

    i_indices = np.clip(i_indices, 0, res_y - 1)
    j_indices = np.clip(j_indices, 0, res_x - 1)

    # Create maps
    hit_map = np.zeros((res_y, res_x), dtype=bool)
    point_map = np.full((res_y, res_x, 3), np.nan)
    normal_map = np.full((res_y, res_x, 3), np.nan)
    cos_th_map = np.full((res_y, res_x), np.nan)

    # Populate maps (this is fast with fancy indexing)
    hit_map[i_indices, j_indices] = True
    point_map[i_indices, j_indices] = hit_points
    normal_map[i_indices, j_indices] = cell_normals
    cos_th_map[i_indices, j_indices] = cos_th

    # Find and fill gaps
    miss_map = ~hit_map
    structure = generate_binary_structure(2, 2)
    labeled_gaps, n_gaps = label(miss_map, structure=structure)

    filled_points = []
    filled_normals = []
    filled_cos_th = []
    filled_i_indices = []
    filled_j_indices = []

    for gap_id in range(1, n_gaps + 1):
        gap_mask = (labeled_gaps == gap_id)
        gap_size = gap_mask.sum()

        if 1 <= gap_size <= max_gap_size:
            neighbors_mask = binary_dilation(gap_mask, structure=structure) & ~gap_mask
            n_hit_neighbors = (neighbors_mask & hit_map).sum()

            if n_hit_neighbors >= neighbors_mask.sum() * 0.4:
                gap_positions = np.where(gap_mask)

                for i, j in zip(gap_positions[0], gap_positions[1]):
                    # Get neighbor values using slicing (faster)
                    i_slice = slice(max(0, i - 1), min(res_y, i + 2))
                    j_slice = slice(max(0, j - 1), min(res_x, j + 2))

                    neighbor_hit = hit_map[i_slice, j_slice]

                    if neighbor_hit.sum() >= 3:
                        neighbor_points = point_map[i_slice, j_slice][neighbor_hit]
                        neighbor_normals = normal_map[i_slice, j_slice][neighbor_hit]
                        neighbor_cos = cos_th_map[i_slice, j_slice][neighbor_hit]

                        filled_points.append(np.mean(neighbor_points, axis=0))
                        filled_normals.append(np.mean(neighbor_normals, axis=0))
                        filled_cos_th.append(np.mean(neighbor_cos))
                        filled_i_indices.append(i)
                        filled_j_indices.append(j)

    if len(filled_points) == 0:
        print("No isolated gaps found to fill")
        return res_prop

    print(f"Filled {len(filled_points)} isolated gap pixels by interpolation")

    # Vectorized normalization
    filled_normals = np.array(filled_normals)
    filled_normals = filled_normals / np.linalg.norm(filled_normals, axis=1, keepdims=True)

    new_ray_ids = np.array(filled_i_indices) * res_x + np.array(filled_j_indices)

    # Merge with original data
    res_prop['hit_points'] = np.vstack([hit_points, filled_points])
    res_prop['cell_normal'] = np.vstack([cell_normals, filled_normals])
    res_prop['cos_th'] = np.concatenate([cos_th, filled_cos_th])
    res_prop['ray_ids'] = np.concatenate([ray_ids, new_ray_ids])
    res_prop['cell_ids'] = np.concatenate([cell_ids, np.full(len(filled_points), -1)])

    return res_prop


# ============================================================================
# ADDITIONAL OPTIMIZATIONS - Remove unnecessary functions
# ============================================================================

def compute_ray_tracing_fast_minimal(mesh: pv.PolyData, r_source: np.ndarray,
                                     res_x: int, res_y: int):
    """
    Minimal version without gap filling for maximum speed.
    Use this if you don't need gap filling or edge filtering.
    """

    r_source = np.asarray(r_source, dtype=np.float64)
    r_source = r_source / np.linalg.norm(r_source)

    # Build orthogonal system
    if abs(r_source[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    u = np.cross(r_source, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(r_source, u)
    v = v / np.linalg.norm(v)

    # Mesh bounds
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.array(mesh.center, dtype=np.float64)
    max_extent = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)

    # Pixel plane
    plane_distance = max_extent * 1.2
    plane_center = center - r_source * plane_distance
    plane_size = max_extent * 1.1

    u_coords = np.linspace(-plane_size / 2, plane_size / 2, res_x, dtype=np.float64)
    v_coords = np.linspace(-plane_size / 2, plane_size / 2, res_y, dtype=np.float64)

    pixel_width = (u_coords[-1] - u_coords[0]) / (res_x - 1) if res_x > 1 else plane_size
    pixel_height = (v_coords[-1] - v_coords[0]) / (res_y - 1) if res_y > 1 else plane_size

    # Generate rays (vectorized)
    UU, VV = np.meshgrid(u_coords, v_coords, indexing='xy')
    UU += 0.5 * pixel_width if res_x > 1 else 0.0
    VV += 0.5 * pixel_height if res_y > 1 else 0.0

    uu_flat = UU.ravel()
    vv_flat = VV.ravel()

    starts = (plane_center[None, :] +
              uu_flat[:, None] * u[None, :] +
              vv_flat[:, None] * v[None, :])

    dirs = np.tile(r_source, (starts.shape[0], 1))

    # Ray trace
    out = mesh.multi_ray_trace(starts, dirs, first_point=True)
    points, ray_ids, cell_ids = out if len(out) == 3 else (*out, None)

    if cell_ids is None:
        raise RuntimeError("cell_ids is None")

    # Vectorized normal computation
    n = mesh.cell_normals[cell_ids]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    cos_th = np.einsum('ij,j->i', n, r_source)

    # Filter
    mask = cos_th < -1e-9

    return {
        "x_range": u_coords,
        "y_range": v_coords,
        "plane_center": plane_center,
        "plane_u": u,
        "plane_v": v,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area": pixel_width * pixel_height,
        "hit_points": points[mask],
        "ray_ids": ray_ids[mask],
        "cell_ids": cell_ids[mask],
        "res_x": res_x,
        "res_y": res_y,
        "ray_starts": starts,
        "ray_dirs": dirs,
        "r_source": r_source,
        "cos_th": cos_th[mask],
        "cell_normal": n[mask]
    }


if __name__ == '__main__':
    import time

    mesh = pv.Cube(x_length=2, y_length=2, z_length=2)
    mesh = mesh.triangulate().clean()
    mesh = mesh.subdivide(2, subfilter='linear').clean()
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

    r_ = np.array([-1, -1, 0.0])
    r_ /= np.linalg.norm(r_)

    # Test minimal version (fastest)
    print("Testing minimal version (no gap filling)...")
    start = time.time()
    res = compute_ray_tracing_fast_minimal(mesh, r_, 500, 500)
    elapsed = time.time() - start
    print(f"Minimal version time: {elapsed:.3f} seconds")
    print(f"Hit points: {len(res['hit_points'])}")

    # Test full version with optimizations
    print("\nTesting full version (with gap filling)...")
    start = time.time()
    res = compute_ray_tracing_fast_optimized(mesh, r_, 500, 500)
    elapsed = time.time() - start
    print(f"Full optimized time: {elapsed:.3f} seconds")
    print(f"Hit points: {len(res['hit_points'])}")
    show_ray_tracing_fast(mesh, res, "", show_mesh=True, save_3d=False)