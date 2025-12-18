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

# ==========================
# ERROR ANALYSIS
# ==========================
R_EARTH = 6378137.0  # m
def compute_position_error(r_true, r_test):
    """Compute position error magnitude"""
    return np.linalg.norm(r_true - r_test, axis=1)


def compute_velocity_error(v_true, v_test):
    """Compute velocity error magnitude"""
    return np.linalg.norm(v_true - v_test, axis=1)


def compute_relative_error(true, test):
    """Compute relative error"""
    return np.abs(true - test) / (np.abs(true) + 1e-10)


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
        plotter.add_mesh(seg_hits, color="blue", line_width=0.4, opacity=0.4)

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
    plane_v[2]= 1
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
    vmin = 0.0 if np.min(values) >= 0.0 else -vmax
    for ax, grid, comp in zip(axes, grids, components):
        masked_grid = np.ma.masked_invalid(grid)


        im = ax.imshow(masked_grid, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax,
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
        ax.set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(im, ax=ax)
     

    plt.suptitle('Torque Distribution', fontsize=18)
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
                                     file_name, title_name='Rarefied Aerodynamics of a Panel',
                                     x_ticks=[0, 15, 30, 45, 60, 75, 90]
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

        if c_a_sigma_analytic is not None and c_s_sigma_analytic is not None and c_n_sigma_analytic is not None:
            # ANALYTICAL (solid lines) - all three coefficients with same color
            line_CA, = ax.plot(aoa_array, c_a_sigma_analytic[i], '-',
                               color=color, linewidth=1., alpha=0.95)

            ax.plot(aoa_array, c_s_sigma_analytic[i], '-',
                    color=color, linewidth=1., alpha=0.95)
            ax.plot(aoa_array, c_n_sigma_analytic[i], '-',
                    color=color, linewidth=1., alpha=0.95)
        else:
            # NUMERICAL/DSMC (dots) - all three coefficients with same color
            line_CA, = ax.plot(aoa_list, c_a_list[i], '-',
                               color=color, linewidth=1., alpha=0.95)

            ax.plot(aoa_list, c_s_list[i], '-',
                    color=color, linewidth=1., alpha=0.95)
            ax.plot(aoa_list, c_n_list[i], '-',
                    color=color, linewidth=1., alpha=0.95)

        # NUMERICAL/DSMC (dots) - all three coefficients with same color
        ax.plot(aoa_list, c_a_list[i], 'o',
                color=color, markersize=7, markeredgewidth=0.5,
                markeredgecolor='white', alpha=0.8)
        ax.plot(aoa_list, c_s_list[i], 'x',
                color=color, markersize=14, markeredgewidth=2.5,
                markeredgecolor=color, alpha=0.8)
        ax.plot(aoa_list, c_n_list[i], '*',
                color=color, markersize=14, markeredgewidth=0.5,
                markeredgecolor='white', alpha=0.8)

        # Add to legend (only once per sigma)
        legend_handles.append(line_CA)
        legend_labels.append(f'{sigma}')

    # Legend (top right, clean style)
    legend = ax.legend(legend_handles, legend_labels,
                       title=r'$\sigma_{N,T}$',
                       loc='best',
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
    ax.set_title(title_name,
                 color=text_color, pad=15)

    # Set axis limits
    ax.set_xlim(0, 90)

    # Make sure x-axis shows key angles
    ax.set_xticks(x_ticks)

    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight',
                facecolor='white')


def show_error_local_coefficient_per_angle(aoa_list, error_c_a_list, error_c_s_list, error_c_n_list, sigma_list,
                                           file_name, x_ticks=[0, 15, 30, 45, 60, 75, 90],
                                           title_name='Rarefied Aerodynamics error of a Panel'):
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
    ax.set_ylabel(r"Error $C_{A,S,N}$ [%]", color=text_color)

    # Plot data - GROUP BY SIGMA VALUE
    legend_handles = []
    legend_labels = []

    for i, sigma in enumerate(sigma_list):
        color = colors.get(sigma, f'C{i}')  # Use predefined colors or default

        line_CA, = ax.plot(aoa_list, error_c_a_list[i], '-',
                           color=color, linewidth=1., alpha=0.95)
        ax.plot(aoa_list, error_c_s_list[i], '-',
                color=color, linewidth=1., alpha=0.95)
        ax.plot(aoa_list, error_c_n_list[i], '-',
                color=color, linewidth=1., alpha=0.95)

        # NUMERICAL/DSMC (dots) - all three coefficients with same color
        ax.plot(aoa_list, error_c_a_list[i], 'o',
                color=color, markersize=7, markeredgewidth=0.5,
                markeredgecolor=color, alpha=0.8)
        ax.plot(aoa_list, error_c_s_list[i], 'x',
                color=color, markersize=14, markeredgewidth=1.5,
                markeredgecolor=color, alpha=0.8)
        ax.plot(aoa_list, error_c_n_list[i], '*',
                color=color, markersize=14, markeredgewidth=0.5,
                markeredgecolor=color, alpha=0.8)

        # Add to legend (only once per sigma)
        legend_handles.append(line_CA)
        legend_labels.append(f'{sigma}')

    # Legend (top right, clean style)
    legend = ax.legend(legend_handles, legend_labels,
                       title=r'$\sigma_{N,T}$',
                       loc='best',
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
    ax.set_title(title_name, color=text_color, pad=15)

    all_errors = []
    for i in range(len(sigma_list)):
        all_errors.extend(error_c_a_list[i])
        all_errors.extend(error_c_s_list[i])
        all_errors.extend(error_c_n_list[i])


    # Set axis limits
    error_min = min(all_errors)
    error_max = max(all_errors)
    margin = (error_max - error_min) * 0.1  # 10% margin
    ax.set_ylim(error_min - margin, error_max + margin)
    ax.set_xticks(x_ticks)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()


def show_torque_drag_per_angle(aoa_list, sigma_list, torque_list,
                               file_name, title_name="Torque on Panel",
                               x_ticks=[0, 15, 30, 45, 60, 75, 90]):
    colors = {
        0.0: '#6B8CD4',  # Blue
        0.25: '#7CAC9D',  # Teal
        0.5: '#A4B86E',  # Yellow-green
        0.75: '#D8944D',  # Orange
        1.0: '#D87F7F'  # Pink/Red
    }

    # Create figure with WHITE background
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, facecolor='white')

    # Configure each subplot
    text_color = 'black'
    component_labels = ['x', 'y', 'z']

    legend_handles = []
    legend_labels = []

    for i in range(3):
        ax[i].set_facecolor('white')
        ax[i].grid(True, color='gray', alpha=0.5, linewidth=0.5)
        ax[i].set_ylabel(f"$\\tau_{{{component_labels[i]}}}$ [mN·m]", color=text_color)
        ax[i].tick_params(colors=text_color, which='both')

        # Set spines color
        for spine in ax[i].spines.values():
            spine.set_color('gray')

        # Scientific notation on y-axis
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax[i].yaxis.get_offset_text().set_color(text_color)

    # Plot data
    for i, sigma in enumerate(sigma_list):
        color = colors.get(sigma, f'C{i}')

        for j in range(3):
            line, = ax[j].plot(aoa_list, np.array(torque_list[i])[:, j] * 1000,
                               '-o', color=color, linewidth=1.5, alpha=0.95,
                               markersize=5, markeredgewidth=0.5)

        # Add to legend only once
        if i == 0:
            legend_handles.append(line)
            legend_labels.append(f'{sigma}')
        else:
            legend_handles.append(line)
            legend_labels.append(f'{sigma}')

    # X-axis label (only on bottom subplot)
    ax[2].set_xlabel(r"Angle of attack $\alpha$ [deg]", color=text_color)
    ax[2].set_xticks(x_ticks)

    # Title
    fig.suptitle(title_name, color=text_color, y=0.98)

    # Single legend OUTSIDE the plot, right and centered
    fig.legend(legend_handles, legend_labels,
               title=r'$\sigma_{N,T}$',
               loc='center right',
               bbox_to_anchor=(1.001, 0.5),  # Right and centered vertically
               framealpha=0.95,
               facecolor='white',
               edgecolor='gray',
               shadow=False)

    # Adjust layout to make room for legend
    plt.tight_layout()  # Leave space on right for legend

    plt.subplots_adjust(
        left=0.13,  # Left margin
        bottom=0.15,  # Bottom margin
        right=0.85,  # Right margin (leave space for legend: 0.85-0.88)
        top=0.90,  # Top margin (leave space for title)
        wspace=0.1,  # Width space between subplots (not needed for single column)
        hspace=0.25  # Height space between subplots (IMPORTANT for 3 rows)
    )

    plt.savefig(file_name, dpi=300, bbox_inches='tight',
                facecolor='white')
    return fig, ax


def show_force_drag_per_angle(aoa_list, sigma_list, torque_list,
                               file_name, title_name="Torque on Panel",
                               x_ticks=[0, 15, 30, 45, 60, 75, 90]):
    colors = {
        0.0: '#6B8CD4',  # Blue
        0.25: '#7CAC9D',  # Teal
        0.5: '#A4B86E',  # Yellow-green
        0.75: '#D8944D',  # Orange
        1.0: '#D87F7F'  # Pink/Red
    }

    # Create figure with WHITE background
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, facecolor='white')

    # Configure each subplot
    text_color = 'black'
    component_labels = ['x', 'y', 'z']

    legend_handles = []
    legend_labels = []

    for i in range(3):
        ax[i].set_facecolor('white')
        ax[i].grid(True, color='gray', alpha=0.5, linewidth=0.5)
        ax[i].set_ylabel(f"$F_{{{component_labels[i]}}}$ [mN]", color=text_color)
        ax[i].tick_params(colors=text_color, which='both')

        # Set spines color
        for spine in ax[i].spines.values():
            spine.set_color('gray')

        # Scientific notation on y-axis
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax[i].yaxis.get_offset_text().set_color(text_color)

    # Plot data
    for i, sigma in enumerate(sigma_list):
        color = colors.get(sigma, f'C{i}')

        for j in range(3):
            line, = ax[j].plot(aoa_list, np.array(torque_list[i])[:, j] * 1000,
                               '-o', color=color, linewidth=1.5, alpha=0.95,
                               markersize=5, markeredgewidth=0.5)

        # Add to legend only once
        if i == 0:
            legend_handles.append(line)
            legend_labels.append(f'{sigma}')
        else:
            legend_handles.append(line)
            legend_labels.append(f'{sigma}')

    # X-axis label (only on bottom subplot)
    ax[2].set_xlabel(r"Angle of attack $\alpha$ [deg]", color=text_color)
    ax[2].set_xticks(x_ticks)

    # Title
    fig.suptitle(title_name, color=text_color, y=0.98)

    # Single legend OUTSIDE the plot, right and centered
    fig.legend(legend_handles, legend_labels,
               title=r'$\sigma_{N,T}$',
               loc='center right',
               bbox_to_anchor=(1.001, 0.5),  # Right and centered vertically
               framealpha=0.95,
               facecolor='white',
               edgecolor='gray',
               shadow=False)

    # Adjust layout to make room for legend
    plt.tight_layout()  # Leave space on right for legend

    plt.subplots_adjust(
        left=0.13,  # Left margin
        bottom=0.15,  # Bottom margin
        right=0.85,  # Right margin (leave space for legend: 0.85-0.88)
        top=0.90,  # Top margin (leave space for title)
        wspace=0.1,  # Width space between subplots (not needed for single column)
        hspace=0.25  # Height space between subplots (IMPORTANT for 3 rows)
    )

    plt.savefig(file_name, dpi=300, bbox_inches='tight',
                facecolor='white')
    return fig, ax


def plot_sphere_distribution(mesh, data_mesh, columns_data, filename, show=False):
    # Create plotter with 3 columns
    plotter = pv.Plotter(shape=(1, 3), border=False, window_size=[1600, 500],
                         off_screen=True if not show else False
                         )

    # Convert lists to numpy arrays for easier handling
    r_vectors = np.array(data_mesh["r_"])


    # Scale factor for vector origin (adjust as needed)
    origin_scale = 5

    origins = -r_vectors * origin_scale
    for col, values, title, scalar_name, view_ in columns_data:
        plotter.subplot(0, col)
        
        # Add mesh
        plotter.add_mesh(mesh, opacity=1, color='lightgray', 
                         show_edges=True, edge_color='black', line_width=1)
        
        points = pv.PolyData(-r_vectors * origin_scale)
        
        # Add colored points
        plotter.add_mesh(points, scalars=values, point_size=10, 
                         render_points_as_spheres=True, 
                         opacity=0.7,
                         cmap='coolwarm',
                         scalar_bar_args={
                             'title': scalar_name,
                             'title_font_size': 30,
                             'label_font_size': 30,
                             'vertical': True, 
                             'position_x': 0.8,
                             'position_y': 0.1,
                             'width': 0.05,
                             'fmt': '%.2f', 
                             'height': 0.8
                         })
        
        plotter.add_title(title, font_size=14)
        plotter.add_axes()
        #for i in range(len(r_vectors)):
        #    arrow = pv.Arrow(start=origins[i], direction=r_vectors[i], 
        #                     scale='auto', tip_length=0.3, shaft_radius=0.01)
        #    plotter.add_mesh(arrow, color='black', opacity=0.3)
            
        plotter.camera_position = view_

    # Link cameras for synchronized rotation
    # plotter.link_views()

    plotter.show(
        screenshot=filename
        )


def plot_predictions_by_axis(P_real, Y_real, force, torque, title_prefix=""):
    """
    Create separate plots for forces and torques.
    Forces: 3 rows (Fx, Fy, Fz)
    Torques: 3 rows (Tx, Ty, Tz)
    
    Args:
        P_real: Predictions array (N, 6) - [Fx, Fy, Fz, Tx, Ty, Tz]
        Y_real: Ground truth array (N, 6) - [Fx, Fy, Fz, Tx, Ty, Tz]
        title_prefix: Prefix for plot titles
    """
    # Output is: [Fx, Fy, Fz, Tx, Ty, Tz]
    _indices = [0, 1, 2]   # Fx, Fy, Fz

    axis_labels = ['x', 'y', 'z']
    fig1, fig2 = None, None
    # ==================== FORCES PLOT ====================
    if force and not torque:
        fig1, axes1 = plt.subplots(1, 3, figsize=(12, 5))
        fig1.suptitle(f'{title_prefix} - Force Predictions vs Ground Truth (PyRase)', fontweight='bold')

        for i, (idx, label) in enumerate(zip(_indices, axis_labels)):
            ax = axes1[i]

            # Scatter plot: Predicted vs Actual
            ax.scatter(Y_real[:, idx], P_real[:, idx], alpha=0.5, s=20, label='Predictions')

            # Perfect prediction line
            min_val = min(Y_real[:, idx].min(), P_real[:, idx].min())
            max_val = max(Y_real[:, idx].max(), P_real[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Calculate metrics
            mae = np.mean(np.abs(P_real[:, idx] - Y_real[:, idx]))
            rmse = np.sqrt(np.mean((P_real[:, idx] - Y_real[:, idx])**2))
            r2 = 1 - np.sum((Y_real[:, idx] - P_real[:, idx])**2) / np.sum((Y_real[:, idx] - Y_real[:, idx].mean())**2)

            unit_text = r"$m^2$"
            ax.set_xlabel(f'PyRase {label}-axis ({unit_text})')
            symb_ = r"$\tilde{\mathbf{F}}$" + fr"$_{label}$"
            ax.set_ylabel(f'Predicted {symb_} ({unit_text})')
            ax.set_title(f'MAE: {mae:.3f} {unit_text} \n RMSE: {rmse:.3f} {unit_text} |R²: {r2:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

    if torque and not force:
        # ==================== TORQUES PLOT ====================
        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
        fig2.suptitle(f'{title_prefix} - Torque Predictions vs Ground Truth (PyRase)', fontweight='bold')

        for i, (idx, label) in enumerate(zip(_indices, axis_labels)):
            ax = axes2[i]

            # Scatter plot: Predicted vs Actual
            ax.scatter(Y_real[:, idx], P_real[:, idx], alpha=0.5, s=20, color='orange', label='Predictions')

            # Perfect prediction line
            min_val = min(Y_real[:, idx].min(), P_real[:, idx].min())
            max_val = max(Y_real[:, idx].max(), P_real[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Calculate metrics
            mae = np.mean(np.abs(P_real[:, idx] - Y_real[:, idx]))
            rmse = np.sqrt(np.mean((P_real[:, idx] - Y_real[:, idx])**2))
            r2 = 1 - np.sum((Y_real[:, idx] - P_real[:, idx])**2) / np.sum((Y_real[:, idx] - Y_real[:, idx].mean())**2)

            unit_text = r"$m^3$"
            ax.set_xlabel(f'PyRase {label}-axis ({unit_text})')
            symb_ = r"$\tilde{\boldsymbol{\tau}}$" + fr"$_{label}$"
            ax.set_ylabel(f'Predicted {symb_} ({unit_text})')
            ax.set_title(f'MAE: {mae:.3f} {unit_text} \n RMSE: {rmse:.3f} {unit_text} |R²: {r2:.3f}')

            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
    
    return fig1, fig2


def plot_orbit_comparison(results_dict, t_eval, save_path='orbit_comparison.png'):
    """
    Plot orbit comparison between models

    Parameters:
    -----------
    results_dict : dict
        {
            'ground_truth': solution,
            'spherical': solution,
            'ann': solution
        }
    t_eval : np.ndarray
        Time vector [s]
    save_path : str
        Output file path
    """
    # Figure 1: 3D orbit + error components (2x2 layout)
    fig = plt.figure(figsize=(8, 7))

    colors = {
        'ground_truth': 'blue',
        'spherical': 'red',
        'ann': 'green'
    }

    labels = {
        'ground_truth': 'Ground Truth (Ray Tracing)',
        'spherical': 'Spherical Model',
        'ann': 'ANN Model'
    }

    # 3D orbit plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    for model_name, sol in results_dict.items():
        r = sol.sol(t_eval)[:3, :].T / 1000  # Convert to km
        ax1.plot(r[:, 0], r[:, 1], r[:, 2],
                 color=colors[model_name],
                 label=labels[model_name],
                 linewidth=1.5, alpha=0.8)

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = R_EARTH / 1000 * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH / 1000 * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH / 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

    ax1.set_xlabel('X [km]', fontsize=10)
    ax1.set_ylabel('Y [km]', fontsize=10)
    ax1.set_zlabel('Z [km]', fontsize=10)
    ax1.set_title('Orbit Trajectories', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Position error vs time
    ax2 = fig.add_subplot(2, 2, 2)

    r_true = results_dict['ground_truth'].sol(t_eval)[:3, :].T

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            r_test = results_dict[model_name].sol(t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax2.plot(t_eval / 3600, pos_error / 1000,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=1.5)

    ax2.set_xlabel('Time [hours]', fontsize=10)
    ax2.set_ylabel('Position Error [km]', fontsize=10)
    ax2.set_title('Position Error vs Time', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=9)

    # Velocity error vs time
    ax3 = fig.add_subplot(2, 2, 3)

    v_true = results_dict['ground_truth'].sol(t_eval)[3:, :].T

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            v_test = results_dict[model_name].sol(t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax3.plot(t_eval / 3600, vel_error,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=1.5)

    ax3.set_xlabel('Time [hours]', fontsize=10)
    ax3.set_ylabel('Velocity Error [m/s]', fontsize=10)
    ax3.set_title('Velocity Error vs Time', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.tick_params(labelsize=9)

    # Error components
    ax4 = fig.add_subplot(2, 2, 4)

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            r_test = results_dict[model_name].sol(t_eval)[:3, :].T
            error_vec = r_test - r_true

            # Radial, along-track, cross-track errors
            for i, component in enumerate(['X', 'Y', 'Z']):
                ax4.plot(t_eval / 3600, error_vec[:, i] / 1000,
                         linestyle='--' if model_name == 'ann' else '-',
                         label=f'{labels[model_name]} - {component}',
                         linewidth=1.2, alpha=0.7)

    ax4.set_xlabel('Time [hours]', fontsize=10)
    ax4.set_ylabel('Position Error Components [km]', fontsize=10)
    ax4.set_title('Position Error Components', fontsize=11)
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Orbit comparison plot saved to: {save_path}")

    # Figure 2: Position and Velocity Errors (1x2 layout)
    fig2, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Position error
    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            r_test = results_dict[model_name].sol(t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax_pos.plot(t_eval / 3600, pos_error / 1000,
                        color=colors[model_name],
                        label=labels[model_name],
                        linewidth=1.5)

    ax_pos.set_xlabel('Time [hours]', fontsize=10)
    ax_pos.set_ylabel('Position Error [km]', fontsize=10)
    ax_pos.set_title('Position Error', fontsize=11)
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_yscale('log')
    ax_pos.tick_params(labelsize=9)

    # Velocity error
    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            v_test = results_dict[model_name].sol(t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax_vel.plot(t_eval / 3600, vel_error,
                        color=colors[model_name],
                        label=labels[model_name],
                        linewidth=1.5)

    ax_vel.set_xlabel('Time [hours]', fontsize=10)
    ax_vel.set_ylabel('Velocity Error [m/s]', fontsize=10)
    ax_vel.set_title('Velocity Error', fontsize=11)
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_yscale('log')
    ax_vel.tick_params(labelsize=9)

    plt.tight_layout()

    # Save second figure
    save_path_2 = save_path.replace('.png', '_errors.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Error comparison plot saved to: {save_path_2}")


def plot_error_statistics(results_dict, t_eval, save_path='error_statistics.png'):
    """
    Plot detailed error statistics
    """
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))

    r_true = results_dict['ground_truth'].sol(t_eval)[:3, :].T
    v_true = results_dict['ground_truth'].sol(t_eval)[3:, :].T

    models = ['spherical', 'ann']
    colors = {'spherical': 'red', 'ann': 'green'}
    labels = {'spherical': 'Spherical', 'ann': 'ANN'}

    for idx, model_name in enumerate(models):
        if model_name not in results_dict:
            continue

        r_test = results_dict[model_name].sol(t_eval)[:3, :].T
        v_test = results_dict[model_name].sol(t_eval)[3:, :].T

        # Position error components
        ax = axes[idx, 0]
        error_r = r_test - r_true
        for i, comp in enumerate(['X', 'Y', 'Z']):
            ax.plot(t_eval / 3600, error_r[:, i] / 1000,
                    label=f'{comp}', linewidth=1.5)
        ax.set_xlabel('Time [hours]', fontsize=9)
        ax.set_ylabel('Position Error [km]', fontsize=9)
        ax.set_title(f'{labels[model_name]} - Position Error Components', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Velocity error components
        ax = axes[idx, 1]
        error_v = v_test - v_true
        for i, comp in enumerate(['Vx', 'Vy', 'Vz']):
            ax.plot(t_eval / 3600, error_v[:, i],
                    label=f'{comp}', linewidth=1.5)
        ax.set_xlabel('Time [hours]', fontsize=9)
        ax.set_ylabel('Velocity Error [m/s]', fontsize=9)
        ax.set_title(f'{labels[model_name]} - Velocity Error Components', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Total error magnitude
        ax = axes[idx, 2]
        pos_error = compute_position_error(r_true, r_test)
        vel_error = compute_velocity_error(v_true, v_test)

        ax_twin = ax.twinx()
        line1 = ax.plot(t_eval / 3600, pos_error / 1000,
                        'b-', label='Position', linewidth=1.5)
        line2 = ax_twin.plot(t_eval / 3600, vel_error,
                             'r-', label='Velocity', linewidth=1.5)

        ax.set_xlabel('Time [hours]', fontsize=9)
        ax.set_ylabel('Position Error [km]', color='b', fontsize=9)
        ax_twin.set_ylabel('Velocity Error [m/s]', color='r', fontsize=9)
        ax.set_title(f'{labels[model_name]} - Total Error', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b', labelsize=8)
        ax_twin.tick_params(axis='y', labelcolor='r', labelsize=8)

        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Error statistics plot saved to: {save_path}")

if __name__ == '__main__':
    pass
