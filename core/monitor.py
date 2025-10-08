"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",                     # generic family
    "font.serif": ["Times New Roman", "Times"], # try exact TNR first
    "font.size": 18,                            # default text size
    "axes.titlesize": 18,                       # axes title
    "axes.labelsize": 18,                       # x/y labels
    "xtick.labelsize": 18,                      # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    # Math text configured to look like Times
    "mathtext.fontset": "stix",                 # STIX resembles Times
    "mathtext.rm": "Times New Roman",
})


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



def show_ray_tracing_fast(mesh, prop: dict, filename="3d_view.png", show_miss_rays=False,
                          show_mesh=False, save_3d=True):
    """
    Visualization updated for generalized coordinate system

    Parameters:
    -----------
    mesh : pv.PolyData
        Spacecraft mesh
    prop : dict
        Dictionary returned by compute_ray_tracing_fast
    filename : str
        Output filename for screenshot
    show_mesh : bool
        If True, show interactive window
    save_3d : bool
        If True, save screenshot to file
    """
    # Extract properties
    hit_points = prop['hit_points']
    ray_starts = prop['ray_starts']
    ray_dirs = prop['ray_dirs']
    ray_ids = prop['ray_ids']
    r_source = prop['r_source']
    plane_center = prop['plane_center']
    plane_u = prop['plane_u']
    plane_v = prop['plane_v']
    res_x, res_y = prop['res_x'], prop['res_y']

    # === Reconstruct pixel plane grid ===
    x_range = prop['x_range']  # Local coordinates in u
    y_range = prop['y_range']  # Local coordinates in v

    # Create grid in local system
    uu, vv = np.meshgrid(x_range, y_range)

    # Transform to global coordinates
    # P = plane_center + uu*plane_u + vv*plane_v
    pixel_points_global = (
            plane_center[None, None, :] +
            uu[:, :, None] * plane_u[None, None, :] +
            vv[:, :, None] * plane_v[None, None, :]
    )

    # Create StructuredGrid for pixel plane
    pixel_grid = pv.StructuredGrid()
    pixel_grid.dimensions = (res_x, res_y, 1)
    pixel_grid.points = pixel_points_global.reshape(-1, 3)

    # === Setup plotter ===
    plotter = pv.Plotter()
    plotter.show_axes()

    # Spacecraft mesh
    plotter.add_mesh(mesh, show_edges=True, color='lightgray', opacity=1.0)

    # Pixel plane (grid)
    plotter.add_mesh(pixel_grid, style="wireframe", color="green",
                     line_width=2, opacity=0.5)

    # Points where rays hit
    if len(hit_points) > 0:
        plotter.add_mesh(pv.PolyData(hit_points), color="red",
                         point_size=5, render_points_as_spheres=True)

    # === Rays that hit (blue) ===
    if len(ray_ids) > 0:
        hit_starts = ray_starts[ray_ids]
        hit_ends = hit_points
        seg_hits = _segments_to_polydata(hit_starts, hit_ends)
        plotter.add_mesh(seg_hits, color="blue", line_width=0.8, opacity=1)

    # === Rays that miss (yellow, optional) ===
    # This can be heavy, so make it optional or with subsampling
    # show_miss_rays = True  # Change to True to see missed rays

    if show_miss_rays:
        total_rays = ray_starts.shape[0]
        all_ids = np.arange(total_rays, dtype=np.int32)
        miss_ids = np.setdiff1d(all_ids, ray_ids)

        # Subsample missed rays to avoid saturating visualization
        if len(miss_ids) > 1000:
            miss_ids = np.random.choice(miss_ids, 1000, replace=False)

        miss_starts = ray_starts[miss_ids]
        # For missed rays, extend them a fixed distance
        miss_ends = miss_starts + ray_dirs[miss_ids] * np.max(mesh.bounds) * 3

        seg_miss = _segments_to_polydata(miss_starts, miss_ends)
        plotter.add_mesh(seg_miss, color="yellow", line_width=0.5, opacity=0.2)

    # === Add arrow showing r_source direction ===
    # Mesh center
    mesh_center = np.array(mesh.center)
    arrow_start = plane_center - r_source * np.max(mesh.bounds) * 0.8
    arrow_end = plane_center - r_source * np.max(mesh.bounds) * 0.2

    arrow = pv.Arrow(start=arrow_start, direction=r_source,
                     scale=np.linalg.norm(arrow_end - arrow_start))
    plotter.add_mesh(arrow, color='orange', label='Source Direction')

    # === Add text indicators ===
    plotter.add_text(f"Source: [{r_source[0]:.2f}, {r_source[1]:.2f}, {r_source[2]:.2f}]",
                     position='upper_left', font_size=10, color='white')
    plotter.add_text(f"Hits: {len(hit_points)}/{len(ray_starts)}",
                     position='upper_right', font_size=10, color='white')

    # === Configure camera ===
    # Position camera to see both plane and mesh well
    # Camera should be "behind" the pixel plane

    # Camera distance
    mesh_extent = np.max(mesh.bounds) - np.min(mesh.bounds)
    cam_distance = mesh_extent * 2.5

    # Position: from opposite direction to r_source
    camera_pos = mesh_center + r_source * cam_distance * 1.8

    # Add lateral offset for better view
    camera_pos += plane_u * cam_distance * 0.3
    camera_pos += plane_v * cam_distance * 0.3

    plotter.camera.position = tuple(camera_pos)
    plotter.camera.focal_point = tuple(mesh_center)
    plotter.camera.up = tuple(plane_v)  # Up vector perpendicular to ray

    plotter.show_grid()
    plotter.show_bounds(
        xlabel='X [m]',
        ylabel='Y [m]',
        zlabel='Z [m]',
        grid='back',
        location='outer',
        all_edges=True,
        font_size=16,
        color='black'
    )
    # === Save or show ===
    if save_3d:
        plotter.show(auto_close=False, interactive=False)
        plotter.screenshot(filename, window_size=(1800, 1200))
        plotter.close()

    if show_mesh:
        plotter.show()
    else:
        if not save_3d:
            plotter.close()


def _segments_to_polydata(starts, ends):
    """
    Convert pairs of points (start, end) into line PolyData

    Parameters:
    -----------
    starts : np.ndarray
        Start points of segments, shape (N, 3)
    ends : np.ndarray
        End points of segments, shape (N, 3)

    Returns:
    --------
    pv.PolyData with lines
    """
    if len(starts) == 0:
        return None

    n_lines = len(starts)
    points = np.vstack([starts, ends])

    lines = np.empty((n_lines, 3), dtype=np.int32)
    lines[:, 0] = 2  # Each line has 2 points
    lines[:, 1] = np.arange(n_lines)  # Index of start
    lines[:, 2] = np.arange(n_lines) + n_lines  # Index of end

    poly = pv.PolyData(points)
    poly.lines = lines.ravel()

    return poly


def plot_normals_with_glyph(mesh, res_prop, arrow_scale=0.05):
    """
    More efficient for many normals using glyphs
    """

    hit_points = res_prop['hit_points']
    normals = res_prop['cell_normal']

    # Create point cloud with normals as vectors
    cloud = pv.PolyData(hit_points)
    cloud['normals'] = normals

    # Create arrows using glyph
    arrows = cloud.glyph(orient='normals', scale=False,
                         factor=np.max(mesh.bounds) * arrow_scale)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, opacity=0.8, color='white', show_edges=True)
    plotter.add_mesh(arrows, color='blue', opacity=0.8)
    plotter.add_points(cloud, color='red', point_size=3)

    plotter.add_axes(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
    plotter.show()

def plot_force_torque_heatmaps(res_prop, values, value_name="Force", filename="distribution.png"):
    """
    Simple heatmaps for force or torque components
    """

    # Spatial projection to pixel indices
    hit_points = res_prop['hit_points']
    plane_center = res_prop['plane_center']
    plane_u = res_prop['plane_u']
    plane_v = res_prop['plane_v']
    res_x = res_prop['res_x']
    res_y = res_prop['res_y']
    x_range = res_prop['x_range']
    y_range = res_prop['y_range']

    relative_pos = hit_points - plane_center
    u_coords = np.dot(relative_pos, plane_u)
    v_coords = np.dot(relative_pos, plane_v)

    i_indices = np.searchsorted(y_range, v_coords) - 1
    j_indices = np.searchsorted(x_range, u_coords) - 1
    i_indices = np.clip(i_indices, 0, res_y - 1)
    j_indices = np.clip(j_indices, 0, res_x - 1)

    # Create grids with NaN
    grids = [np.full((res_y, res_x), np.nan) for _ in range(3)]

    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        if k < len(values):
            grids[0][i, j] = values[k, 0]
            grids[1][i, j] = values[k, 1]
            grids[2][i, j] = values[k, 2]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    components = ['X', 'Y', 'Z']

    vmax = np.abs(values).max()
    for ax, grid, comp in zip(axes, grids, components):
        masked_grid = np.ma.masked_invalid(grid)


        im = ax.imshow(masked_grid, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        ax.set_title(f'{value_name} Component {comp}', fontsize=16)
        ax.set_xlabel('Pixel Index (u)', fontsize=14)
        ax.set_ylabel('Pixel Index (v)', fontsize=14)


        cbar = plt.colorbar(im, ax=ax, format='%.2e')

    plt.suptitle(f'{value_name} Distribution', fontsize=18)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()

def plot_torque_heatmaps(res_prop, T_s, filename="torque_distribution.png"):
    """
    Create 3 heatmaps showing torque components distribution on pixel plane
    """

    # Extract from res_prop
    hit_points = res_prop['hit_points']
    plane_center = res_prop['plane_center']
    plane_u = res_prop['plane_u']
    plane_v = res_prop['plane_v']
    res_x = res_prop['res_x']
    res_y = res_prop['res_y']
    x_range = res_prop['x_range']
    y_range = res_prop['y_range']

    # Project hit_points to pixel indices using spatial coordinates
    relative_pos = hit_points - plane_center
    u_coords = np.dot(relative_pos, plane_u)
    v_coords = np.dot(relative_pos, plane_v)

    i_indices = np.searchsorted(y_range, v_coords) - 1
    j_indices = np.searchsorted(x_range, u_coords) - 1

    i_indices = np.clip(i_indices, 0, res_y - 1)
    j_indices = np.clip(j_indices, 0, res_x - 1)

    # Create grids - initialize with NaN instead of zero
    T_x_grid = np.full((res_y, res_x), np.nan)
    T_y_grid = np.full((res_y, res_x), np.nan)
    T_z_grid = np.full((res_y, res_x), np.nan)

    # Fill grids with torque values using SPATIAL indices
    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        if k < len(T_s):  # Safety check
            T_x_grid[i, j] = T_s[k, 0]
            T_y_grid[i, j] = T_s[k, 1]
            T_z_grid[i, j] = T_s[k, 2]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Determine color scale limits - EXCLUDE NaN values
    valid_torques = T_s[~np.isnan(T_s).any(axis=1)]
    if len(valid_torques) > 0:
        vmax = max(np.abs(valid_torques).max(), 1e-15)
    else:
        vmax = 1e-15

    components = ['X', 'Y', 'Z']
    grids = [T_x_grid, T_y_grid, T_z_grid]

    for ax, grid, comp in zip(axes, grids, components):
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        # Mask NaN values for display
        masked_grid = np.ma.masked_invalid(grid)

        im = ax.imshow(masked_grid, cmap='RdBu_r', norm=norm,
                       origin='lower', aspect='auto', interpolation='nearest')

        ax.set_title(f'Torque Component {comp} [N·m]', fontsize=16)
        ax.set_xlabel('Pixel Index (u direction)', fontsize=14)
        ax.set_ylabel('Pixel Index (v direction)', fontsize=14)

        cbar = plt.colorbar(im, ax=ax)
        ax.grid(True, alpha=0.3, linewidth=0.5, color='white')

    plt.suptitle('Torque Distribution on Pixel Plane', fontsize=18)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved torque heatmaps to: {filename}")

    plt.show()

    # Print diagnostics - exclude NaN
    print("\n" + "=" * 60)
    print("TORQUE DISTRIBUTION DIAGNOSTICS")
    print("=" * 60)

    for comp, grid in zip(components, grids):
        valid_data = grid[~np.isnan(grid)]
        if len(valid_data) > 0:
            print(f"Torque {comp}: min={valid_data.min():.3e}, max={valid_data.max():.3e}, sum={valid_data.sum():.3e}")
        else:
            print(f"Torque {comp}: No valid data")

    print(f"\nTotal hits mapped: {len(i_indices)}")
    print(f"Pixel grid size: {res_x} × {res_y} = {res_x * res_y} pixels")

    # Count non-NaN pixels
    non_nan_pixels = np.sum(~np.isnan(T_x_grid))
    print(f"Coverage: {non_nan_pixels}/{res_x * res_y} pixels ({non_nan_pixels / (res_x * res_y) * 100:.2f}%)")

    # Symmetry check
    print("\n" + "=" * 60)
    print("SYMMETRY CHECK")
    print("=" * 60)

    for comp, grid in zip(components, grids):
        valid_data = grid[~np.isnan(grid)]
        if len(valid_data) > 0:
            positive = valid_data[valid_data > 0].sum()
            negative = valid_data[valid_data < 0].sum()
            net = positive + negative
            balance = abs(net) / (abs(positive) + abs(negative) + 1e-20) * 100

            print(f"\nComponent {comp}:")
            print(f"  Positive contributions: {positive:+.3e} N·m")
            print(f"  Negative contributions: {negative:+.3e} N·m")
            print(f"  Net (should be ~0):     {net:+.3e} N·m")
            print(f"  Balance metric:         {balance:.4f}% (closer to 0% is better)")


def show_local_coefficient_per_angle(aoa_list, aoa_array,
                                     c_a_sigma_analytic, c_s_sigma_analytic, c_n_sigma_analytic,
                                     c_a_list, c_s_list, c_n_list, sigma_list,
                                     file_name
                                     ):
    colors = {
        0.0: '#6B8CD4',  # Blue
        0.25: '#7CAC9D',  # Teal
        0.5: '#A4B86E',  # Yellow-green
        0.75: '#D8944D',  # Orange
        1.0: '#D87F7F'  # Pink/Red
    }

    # Create figure with WHITE background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')

    # Set text colors for white background
    text_color = 'black'
    ax.tick_params(colors=text_color, which='both')
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['right'].set_color('gray')

    # Grid
    ax.grid(True, color='gray', alpha=0.5, linewidth=0.5)

    # Labels
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]", color=text_color)
    ax.set_ylabel(r"$C_{A,S,N}$ [-]", color=text_color)

    # Plot data - GROUP BY SIGMA VALUE
    legend_handles = []
    legend_labels = []

    for i, sigma in enumerate(sigma_list):
        color = colors.get(sigma, f'C{i}')  # Use predefined colors or default

        # ANALYTICAL (solid lines) - all three coefficients with same color
        line_CA, = ax.plot(aoa_array, c_a_sigma_analytic[i], '-',
                           color=color, linewidth=1., alpha=0.95)
        ax.plot(aoa_array, c_s_sigma_analytic[i], '-',
                color=color, linewidth=1., alpha=0.95)
        ax.plot(aoa_array, c_n_sigma_analytic[i], '-',
                color=color, linewidth=1., alpha=0.95)

        # NUMERICAL/DSMC (dots) - all three coefficients with same color
        ax.plot(aoa_list, c_a_list[i], 'o',
                color=color, markersize=7, markeredgewidth=0.5,
                markeredgecolor='white', alpha=0.8)
        ax.plot(aoa_list, c_s_list[i], 'x',
                color=color, markersize=14, markeredgewidth=0.5,
                markeredgecolor='white', alpha=0.8)
        ax.plot(aoa_list, c_n_list[i], '*',
                color=color, markersize=14, markeredgewidth=0.5,
                markeredgecolor='white', alpha=0.8)

        # Add to legend (only once per sigma)
        legend_handles.append(line_CA)
        legend_labels.append(f'{sigma}')

    # Legend (top right, clean style)
    legend = ax.legend(legend_handles, legend_labels,
                       title=r'$\sigma_{N,T}$',
                       loc='upper right',
                       framealpha=0.95,
                       facecolor='white',
                       edgecolor='gray',
                       shadow=False)
    legend.get_title().set_color(text_color)
    legend.get_title().set_weight('bold')
    for text in legend.get_texts():
        text.set_color(text_color)

    # Add method indicator at bottom center
    ax.text(0.5, -0.22, r'$\circ$ $C_A$,      $\times$ $C_S$,      $\bigstar$ $C_N$,      $-$ Analytic',
        transform=ax.transAxes,
        ha='center', color=text_color,
        style='normal')

    # Title (optional)
    ax.set_title('Rarefied Aerodynamics of a Panel\n' +
                 r'$T_\infty$= 973 K, $V_\infty$= 7500 m/s, $T_W$= 300 K',
                 color=text_color, pad=15)

    # Set axis limits
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 4.5)

    # Make sure x-axis shows key angles
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()

if __name__ == '__main__':
    pass
