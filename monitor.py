"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""
import pyvista as pv
import numpy as np


def show_ray_tracing(mesh, prop: dict):
    x_range = prop['x_range']
    y_range = prop['y_range']
    res_x, res_y = prop['res_x'], prop['res_y']
    hit_points = prop['hit_points']
    ray_starts = prop['ray_starts']
    ray_ends = prop['ray_ends']
    z_start = prop['z_start']

    pixel_grid = pv.StructuredGrid()
    pixel_grid.dimensions = (prop['res_x'], prop['res_y'], 1)  # 2D grid
    xx, yy = np.meshgrid(x_range, y_range)
    pixel_grid.points = np.column_stack((xx.ravel(), yy.ravel(), np.full(res_x * res_y, z_start)))

    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.add_mesh(mesh, show_edges=True)  # add mesh
    plotter.add_mesh(pixel_grid, style="wireframe", color="green", line_width=2)
    plotter.add_mesh(np.array(hit_points), color="red", point_size=5)

    # for start, end in zip(ray_starts, ray_ends):
    #     points, _ = mesh.ray_trace(start, end)
    #     if points.size == 0:
    #         line = pv.Line(start, end)
    #         plotter.add_mesh(line, color="yellow", line_width=1, opacity=0.5)

    for start, end in zip(ray_starts, ray_ends):
        points, _ = mesh.ray_trace(start, end)
        if points.size > 0:
            line = pv.Line(start, points[0])
            plotter.add_mesh(line, color="blue", line_width=1)

    # centers = np.array(ray_starts)
    # plotter.add_mesh(centers, color="red", point_size=3)
    plotter.show_grid()
    plotter.show()


def show_ray_tracing_fast(mesh, prop: dict, filename="3d_view.png", show_mesh=False, save_3d=True):
    x_range = prop['x_range']
    y_range = prop['y_range']
    res_x, res_y = prop['res_x'], prop['res_y']
    hit_points = prop['hit_points']
    ray_starts = prop['ray_starts']
    ray_ends = prop['ray_ends']
    ray_dirs = prop['ray_dirs']
    z_start = prop['z_start']
    ray_ids = prop['ray_ids']

    pixel_grid = pv.StructuredGrid()
    pixel_grid.dimensions = (prop['res_x'], prop['res_y'], 1)  # 2D grid
    xx, yy = np.meshgrid(x_range, y_range)
    pixel_grid.points = np.column_stack((xx.ravel(), yy.ravel(), np.full(res_x * res_y, z_start)))

    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.add_mesh(mesh, show_edges=True)  # add mesh
    plotter.add_mesh(pixel_grid, style="wireframe", color="green", line_width=2)
    plotter.add_mesh(np.array(hit_points), color="red", point_size=10)

    hit_starts = ray_starts[ray_ids]
    hit_ends = hit_points

    total_rays = ray_starts.shape[0]
    all_ids = np.arange(total_rays, dtype=np.int32)

    miss_ids = np.setdiff1d(all_ids, ray_ids, assume_unique=False)
    miss_starts = ray_starts[miss_ids]
    miss_ends = ray_ends[miss_ids]

    seg_hits = _segments_to_polydata(hit_starts, hit_ends) if hit_starts.size else None
    seg_miss = _segments_to_polydata(miss_starts, miss_ends) if miss_starts.size else None


    if seg_hits is not None:
        plotter.add_mesh(seg_hits, color="blue", line_width=1)
    if seg_miss is not None:
        plotter.add_mesh(seg_miss, color="yellow", line_width=1, opacity=0.25)

    # centers = np.array(ray_starts)
    # plotter.add_mesh(centers, color="red", point_size=3)
    w = 6 * 300
    h = 6 * 300
    plotter.show_grid()
    dir_cam = np.array([1, 1, 1])
    distance = np.max(x_range) - np.min(x_range)

    pos = np.array((0.0, 0.0, 0.0), float) + 3 * distance * dir_cam

    # Apply to plotter
    plotter.camera.position = tuple(pos)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (1.0, 0.0, 0.0)
    if save_3d:
        plotter.show(auto_close=False, interactive=False)
        plotter.screenshot(filename, window_size=(1800, 1200))
        plotter.close()
    if show_mesh:
        plotter.show()
    else:
        plotter.close()

def _segments_to_polydata(p0: np.ndarray, p1: np.ndarray) -> pv.PolyData:
    assert p0.shape == p1.shape and p0.shape[1] == 3
    n = p0.shape[0]
    pts = np.vstack([p0, p1])
    lines = np.hstack([[2, i, i + n] for i in range(n)]).astype(np.int64)
    poly = pv.PolyData(pts)
    poly.lines = lines
    return poly

if __name__ == '__main__':
    pass
