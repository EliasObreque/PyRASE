"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""
import numpy as np
import pyvista as pv
from numba import jit, prange, cuda
import numba as nb
from scipy.ndimage import label, binary_dilation, generate_binary_structure


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


def compute_ray_tracing_fast(mesh: pv.PolyData, r_source: np.ndarray, res_x: int, res_y: int,
                             fill_gaps = True):
    """
    Generalized ray tracing for any incident direction.

    Parameters:
    -----------
    mesh : pv.PolyData
        Triangulated mesh of the spacecraft
    r_source : np.ndarray
        Unit incident direction vector [3,] (e.g., sun or velocity)
        Points FROM source TOWARDS satellite (convention: negative)
    res_x, res_y : int
        Pixel grid resolution

    Returns:
    --------
    dict with ray tracing information including hit points, normals, and cos(theta)
    """

    # Normalize r_source
    r_source = np.asarray(r_source, dtype=np.float64)
    r_source = r_source / np.linalg.norm(r_source)

    # === 1. Build orthogonal coordinate system perpendicular to ray ===
    # Need two vectors perpendicular to r_source to define the plane

    # Choose arbitrary vector that is NOT parallel to r_source
    if abs(r_source[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    # Build orthonormal basis using Gram-Schmidt
    u = np.cross(r_source, arbitrary)
    u = u / np.linalg.norm(u)  # First perpendicular vector

    v = np.cross(r_source, u)
    v = v / np.linalg.norm(v)  # Second perpendicular vector

    # Now {u, v, r_source} form an orthonormal basis

    # === 2. Determine mesh bounds ===
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = mesh.center

    # Maximum radius from center (to ensure complete coverage)
    max_extent = np.sqrt(
        (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
    )

    # === 3. Create pixel plane perpendicular to r_source ===
    # Plane is at a safe distance "behind" the mesh
    plane_distance = max_extent * 1.2
    plane_center = center - r_source * plane_distance

    # Plane size (with 10% padding)
    plane_size = max_extent * 1.1

    # Create grid on the plane (pixel corners)
    u_coords = np.linspace(-plane_size / 2, plane_size / 2, res_x, dtype=np.float64)
    v_coords = np.linspace(-plane_size / 2, plane_size / 2, res_y, dtype=np.float64)

    # Calculate pixel dimensions
    pixel_width = (u_coords[-1] - u_coords[0]) / (res_x - 1) if res_x > 1 else plane_size
    pixel_height = (v_coords[-1] - v_coords[0]) / (res_y - 1) if res_y > 1 else plane_size
    px_area = pixel_width * pixel_height

    # === 4. Generate ray start points at PIXEL CENTERS ===
    UU, VV = np.meshgrid(u_coords, v_coords, indexing='xy')

    # Shift to pixel centers
    UU = UU + (0.5 * pixel_width if res_x > 1 else 0.0)
    VV = VV + (0.5 * pixel_height if res_y > 1 else 0.0)

    # Flatten for ray generation
    uu_flat = UU.ravel()
    vv_flat = VV.ravel()

    # Transform to global coordinates
    # P = plane_center + uu*u + vv*v
    starts = (
            plane_center[None, :] +
            uu_flat[:, None] * u[None, :] +
            vv_flat[:, None] * v[None, :]
    ).astype(np.float64)

    # === 5. Ray directions (all parallel to r_source) ===
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

    if cell_ids is None:
        raise RuntimeError("cell_ids is None; need cell_ids to map hits -> face normals.")

    # === 7. Calculate normals and cos(theta) ===
    n = mesh.cell_normals[cell_ids]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)

    # cos(theta) = n · r_source
    # Negative means the face "sees" the source
    cos_th = n @ r_source

    # === 8. Filter back-facing triangles (optional but recommended) ===
    # Only keep faces that see the source (cos_th < 0)
    mask = cos_th < -1e-9

    hit_points = points[mask]
    ray_ids_filtered = ray_ids[mask]
    cell_ids_filtered = cell_ids[mask]
    cos_th_filtered = cos_th[mask]
    n_filtered = n[mask]

    res_prop = {
        "x_range": u_coords,  # Pixel corner coordinates in local system
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
        "res_x": res_x,
        "res_y": res_y,
        "ray_starts": starts,  # These are at pixel centers
        "ray_dirs": dirs,
        "r_source": r_source,
        "cos_th": cos_th_filtered,  # Positive values
        "cell_normal": n_filtered
    }

    res_prop = fill_gaps_by_spatial_interpolation(res_prop, max_gap_size=10)
    filtered = filter_edge_artifacts(
        res_prop['hit_points'],
        res_prop['cell_normal'],
        res_prop['cos_th'],
        res_prop['cell_ids'],
        res_prop['ray_ids'],
        mesh
    )
    res_prop['hit_points'] = filtered['hits']
    res_prop['cell_normal'] = filtered['normals']
    res_prop['cos_th'] = filtered['cos_th']
    res_prop['cell_ids'] = filtered['cell_ids']
    res_prop['ray_ids'] = filtered['ray_ids']
    return res_prop


def filter_edge_artifacts(hits, cell_normals, cos_th, cell_ids, ray_ids, mesh):
    """
    Remove hits on edges/corners - they're numerical artifacts with no physical area
    """

    mesh_extent = np.max(mesh.bounds) - np.min(mesh.bounds)
    edge_tol = mesh_extent * 0.0001  # 0.5% tolerance

    bounds = np.array(mesh.bounds).reshape(3, 2)

    # Count boundaries each hit is on
    on_boundary_count = 0
    for axis in range(3):
        at_min = np.abs(hits[:, axis] - bounds[axis, 0]) < edge_tol
        at_max = np.abs(hits[:, axis] - bounds[axis, 1]) < edge_tol
        on_boundary_count += (at_min | at_max).astype(int)

    # Keep only hits on exactly 1 face (not on edges/corners)
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


def fill_gaps_by_spatial_interpolation(res_prop, max_gap_size=2):
    """
    Fill gaps using SPATIAL coordinates, not ray_ids
    Projects hit_points back onto pixel plane to find (i,j) positions
    """

    # Extract from res_prop dictionary
    hit_points = res_prop['hit_points']
    ray_ids = res_prop['ray_ids']  # ← AÑADIR ESTA LÍNEA
    cell_ids = res_prop['cell_ids']  # ← AÑADIR ESTA LÍNEA
    plane_center = res_prop['plane_center']
    plane_u = res_prop['plane_u']
    plane_v = res_prop['plane_v']
    pixel_width = res_prop['pixel_width']
    pixel_height = res_prop['pixel_height']
    res_x = res_prop['res_x']
    res_y = res_prop['res_y']
    cell_normals = res_prop['cell_normal']
    cos_th = res_prop['cos_th']

    # Vector from plane center to each hit
    relative_pos = hit_points - plane_center

    # Project onto u and v axes
    u_coords = np.dot(relative_pos, plane_u)
    v_coords = np.dot(relative_pos, plane_v)

    # Convert to pixel indices
    x_range = res_prop['x_range']
    y_range = res_prop['y_range']

    i_indices = np.searchsorted(y_range, v_coords) - 1
    j_indices = np.searchsorted(x_range, u_coords) - 1

    i_indices = np.clip(i_indices, 0, res_y - 1)
    j_indices = np.clip(j_indices, 0, res_x - 1)

    # Create maps using SPATIAL indices
    hit_map = np.zeros((res_y, res_x), dtype=bool)
    point_map = np.full((res_y, res_x, 3), np.nan)
    normal_map = np.full((res_y, res_x, 3), np.nan)
    cos_th_map = np.full((res_y, res_x), np.nan)

    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        hit_map[i, j] = True
        point_map[i, j] = hit_points[k]
        normal_map[i, j] = cell_normals[k]
        cos_th_map[i, j] = cos_th[k]

    # Find isolated gaps
    miss_map = ~hit_map
    structure = generate_binary_structure(2, 2)
    labeled_gaps, n_gaps = label(miss_map, structure=structure)

    # Fill gaps
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
                    neighbor_points = []
                    neighbor_normals = []
                    neighbor_cos = []

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < res_y and 0 <= nj < res_x:
                                if hit_map[ni, nj]:
                                    neighbor_points.append(point_map[ni, nj])
                                    neighbor_normals.append(normal_map[ni, nj])
                                    neighbor_cos.append(cos_th_map[ni, nj])

                    if len(neighbor_points) >= 3:
                        filled_points.append(np.mean(neighbor_points, axis=0))
                        filled_normals.append(np.mean(neighbor_normals, axis=0))
                        filled_cos_th.append(np.mean(neighbor_cos))
                        filled_i_indices.append(i)
                        filled_j_indices.append(j)

    if len(filled_points) == 0:
        print("No isolated gaps found to fill")
        return res_prop

    print(f"Filled {len(filled_points)} isolated gap pixels by interpolation")

    # Normalize interpolated normals
    filled_normals = np.array(filled_normals)
    filled_normals = filled_normals / np.linalg.norm(filled_normals, axis=1, keepdims=True)

    # Create new ray_ids from (i,j) positions
    new_ray_ids = np.array([i * res_x + j for i, j in zip(filled_i_indices, filled_j_indices)])

    # Merge with original data
    res_prop['hit_points'] = np.vstack([hit_points, filled_points])
    res_prop['cell_normal'] = np.vstack([cell_normals, filled_normals])
    res_prop['cos_th'] = np.concatenate([cos_th, filled_cos_th])
    res_prop['ray_ids'] = np.concatenate([ray_ids, new_ray_ids])
    res_prop['cell_ids'] = np.concatenate([cell_ids, np.full(len(filled_points), -1)])

    return res_prop

def correct_edge_normals_minimal(hits, cell_normals, mesh):
    """
    Correct ONLY true edge/corner hits
    """

    needs_correction = detect_true_edges_only(hits, mesh)

    if needs_correction.sum() == 0:
        return cell_normals

    corrected = cell_normals.copy()
    mesh_extent = np.max(mesh.bounds) - np.min(mesh.bounds)
    edge_tol = mesh_extent * 0.005
    bounds = np.array(mesh.bounds).reshape(3, 2)

    # Para cada edge hit, construir normal ideal geométricamente
    edge_indices = np.where(needs_correction)[0]

    for idx in edge_indices:
        point = hits[idx]
        ideal_normal = np.zeros(3)

        # Para cada eje, si está en boundary, contribuye a la normal
        for axis in range(3):
            at_min = np.abs(point[axis] - bounds[axis, 0]) < edge_tol
            at_max = np.abs(point[axis] - bounds[axis, 1]) < edge_tol

            if at_max:
                ideal_normal[axis] = 1.0
            elif at_min:
                ideal_normal[axis] = -1.0

        # Normalizar
        norm = np.linalg.norm(ideal_normal)
        if norm > 0:
            corrected[idx] = ideal_normal / norm

    print(f"Corrected {needs_correction.sum()} edge/corner normals")
    return corrected


def detect_true_edges_only(hits, mesh):
    """
    Detect ONLY actual edge hits (intersection of 2+ faces)
    Not all points near face boundaries
    """

    mesh_extent = np.max(mesh.bounds) - np.min(mesh.bounds)
    edge_tol = mesh_extent * 0.005  # Más estricto: 0.5%

    bounds = np.array(mesh.bounds).reshape(3, 2)

    # Para cada hit, contar cuántas coordenadas están EXACTAMENTE en boundary
    on_boundary_count = np.zeros(len(hits), dtype=int)

    for axis in range(3):
        # Está en el boundary de este eje?
        at_min = np.abs(hits[:, axis] - bounds[axis, 0]) < edge_tol
        at_max = np.abs(hits[:, axis] - bounds[axis, 1]) < edge_tol
        on_this_boundary = at_min | at_max

        on_boundary_count += on_this_boundary.astype(int)

    # TRUE edges: exactamente en 2 boundaries (arista)
    # TRUE corners: exactamente en 3 boundaries (vértice)
    is_edge = on_boundary_count == 2
    is_corner = on_boundary_count == 3

    needs_correction = is_edge | is_corner

    print(f"True edges: {is_edge.sum()}, corners: {is_corner.sum()}")
    print(f"Total to correct: {needs_correction.sum()} ({needs_correction.sum() / len(hits) * 100:.2f}%)")

    return needs_correction

if __name__ == '__main__':
    import pyvista as pv
    from monitor import show_ray_tracing_fast, plot_normals_with_glyph
    model = "models/Aqua+(B).stl"
    
    mesh = pv.read(model)
    mesh = mesh.triangulate().clean()

    mesh = pv.Cube(x_length=2, y_length=2, z_length=2)
    mesh = mesh.triangulate().clean()
    mesh = mesh.subdivide(2, subfilter='linear').clean()
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    #mesh.rotate_x(10, inplace=True)
    #mesh.rotate_y(25, inplace=True)

    r_ = np.array([-1, -1, 0.0])
    r_ /= np.linalg.norm(r_)
    res = compute_ray_tracing_fast(mesh, r_, 500, 500)
    plot_normals_with_glyph(mesh.copy(), res, arrow_scale=0.01)
    show_ray_tracing_fast(mesh, res, save_3d=False, show_mesh=True)

