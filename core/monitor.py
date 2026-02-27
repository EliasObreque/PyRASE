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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

mpl.rcParams.update({
    "font.family": "serif",                     # generic family
    "font.serif": ["Times New Roman", "Times"], # try exact TNR first
    "font.size": 14,                            # default text size
    "axes.titlesize": 14,                       # axes title
    "axes.labelsize": 14,                       # x/y labels
    "xtick.labelsize": 14,                      # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
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


def get_hill_frame_ON(rc_, vc_):
    vec_or = rc_ / np.linalg.norm(rc_)
    h = np.cross(rc_, vc_)
    f_dot = np.linalg.norm(h) / np.linalg.norm(rc_) ** 2
    vec_oh = h / np.linalg.norm(h)
    vec_ot = np.cross(vec_oh, vec_or)
    on_ = np.array([vec_or, vec_ot, vec_oh])
    return on_


def plot_orbit(results_dict, t_eval):
    fig = plt.figure(figsize=(12, 7))
    colors = {
        'nominal_j2': 'black',
        'ground_truth': 'blue',
        'spherical': 'red',
        'ann': 'green'
    }

    labels = {
        'nominal_j2': 'Nominal J2',
        'ground_truth': 'Ground Truth (Ray Tracing)',
        'spherical': 'Spherical Model',
        'ann': 'ANN Model'
    }

    # 3D orbit plot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    for model_name, sol in results_dict.items():
        if model_name == 't_eval':
            continue
        r = sol.sol(t_eval)[:3, :].T / 1000  # Convert to km

        ax1.plot(r[:, 0], r[:, 1], r[:, 2],
                 color=colors[model_name],
                 label=labels[model_name],
                 linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    ax1.set_title('Orbit Trajectories')
    ax1.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

def plot_altitude(results_dict, t_eval, save_path='altitude_comparison.png'):
    R_EARTH = 6378137.0  # m

    fig = plt.figure(figsize=(10, 5))
    colors = {
        'nominal_j2': 'black',
        'ground_truth': 'blue',
        'spherical': 'red',
        'ann': 'green'
    }

    labels = {
        'nominal_j2': 'Nominal J2',
        'ground_truth': 'Ground Truth (Ray Tracing)',
        'spherical': 'Spherical Model',
        'ann': 'ANN Model'
    }
    plt.xlabel("Time [h]")
    plt.ylabel("Altitude [km]")
    for model_name, sol in results_dict.items():
        if model_name == 't_eval':
            continue
        r = sol.sol(t_eval)[:3, :].T / 1000  # Convert to km
        r_alt = np.linalg.norm(r, axis=1) - R_EARTH /1000

        plt.plot(t_eval/ 3600, r_alt,
                 label=labels[model_name],
                 linewidth=1.5, alpha=0.8)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')


def _get_data(sol, t_eval):
    """Get state data from OdeResult"""
    return sol.sol(t_eval)


def plot_orbit_comparison(results_dict, t_eval, save_path='orbit_comparison.png'):
    """One plot showing all altitudes together"""
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'])
    colors_alt = {'300': '#1f77b4', '400': '#ff7f0e', '500': '#2ca02c'}
    
    # Figure 1: 2x2 grid
    fig = plt.figure(figsize=(14, 10))

    # 3D orbit - ALL models on ONE plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        
        # Plot ground truth for this altitude
        if 'ground_truth' in alt_results:
            r = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T / 1000
            ax1.plot(r[:, 0], r[:, 1], r[:, 2], '-',
                    color=colors_alt[alt], linewidth=2, alpha=0.8,
                    label=f'GT {alt}km')
        
        # Plot spherical for this altitude
        if 'spherical' in alt_results:
            r = alt_results['spherical'].sol(alt_t_eval)[:3, :].T / 1000
            ax1.plot(r[:, 0], r[:, 1], r[:, 2], '--',
                    color=colors_alt[alt], linewidth=1.5, alpha=0.6,
                    label=f'Sph {alt}km')
        
        # Plot ANN for this altitude
        if 'ann' in alt_results:
            r = alt_results['ann'].sol(alt_t_eval)[:3, :].T / 1000
            ax1.plot(r[:, 0], r[:, 1], r[:, 2], ':',
                    color=colors_alt[alt], linewidth=1.5, alpha=0.6,
                    label=f'ANN {alt}km')

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = R_EARTH / 1000 * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH / 1000 * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH / 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    ax1.set_title('Orbit Trajectories')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Position error vs time - Spherical
    ax2 = fig.add_subplot(2, 2, 2)
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            r_test = alt_results['spherical'].sol(alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax2.plot(alt_t_eval / 3600, pos_error / 1000, color=colors_alt[alt],
                    label=f'{alt} km', linewidth=2)
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Position Error [km]')
    ax2.set_title('Position Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Velocity error vs time - Spherical
    ax3 = fig.add_subplot(2, 2, 3)
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            v_test = alt_results['spherical'].sol(alt_t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax3.plot(alt_t_eval / 3600, vel_error, color=colors_alt[alt],
                    label=f'{alt} km', linewidth=2)
    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('Velocity Error [m/s]')
    ax3.set_title('Velocity Error vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Position error components
    ax4 = fig.add_subplot(2, 2, 4)
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            
            if 'spherical' in alt_results:
                r_test = alt_results['spherical'].sol(alt_t_eval)[:3, :].T
                error_vec = r_test - r_true
                for i, comp in enumerate(['X', 'Y', 'Z']):
                    ax4.plot(alt_t_eval / 3600, error_vec[:, i] / 1000, '-',
                            color=colors_alt[alt], linewidth=1.2, alpha=0.7,
                            label=f'Sph-{comp} {alt}km' if i == 0 else '')
            
            if 'ann' in alt_results:
                r_test = alt_results['ann'].sol(alt_t_eval)[:3, :].T
                error_vec = r_test - r_true
                for i, comp in enumerate(['X', 'Y', 'Z']):
                    ax4.plot(alt_t_eval / 3600, error_vec[:, i] / 1000, '--',
                            color=colors_alt[alt], linewidth=1.2, alpha=0.7,
                            label=f'ANN-{comp} {alt}km' if i == 0 else '')
    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Position Error Components [km]')
    ax4.set_title('Position Error Components')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Orbit comparison saved to: {save_path}")


def plot_pos_vel_error(results_dict, t_eval, save_path='orbit_comparison_error.png', log_scale=False):
    """Position and velocity errors with twin axes"""
    from matplotlib.lines import Line2D
    
    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(6, 6))
    
    # for drag use: 
    colors_alt = {'300': '#D87F7F', '400': '#A4B86E', '500': '#6B8CD4'}
    # for srp use:
    #colors_alt = {'500': '#D87F7F', '1000': '#A4B86E', '1500': '#6B8CD4'}
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'], key=int)
    
    # Position error
    
    for i, alt in enumerate(altitudes):
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            r_test = alt_results['ann'].sol(alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax_pos.plot(alt_t_eval / 3600, pos_error, '-', #label="ANN" if i == 0 else None,
                       color=colors_alt[alt], linewidth=1.5)

        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            r_test = alt_results['spherical'].sol(alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax_pos.plot(alt_t_eval / 3600, pos_error, '--', #label="Spherical" if i == 0 else None,
                        color=colors_alt[alt], linewidth=1.5)
    
    ax_pos.set_xlabel('Time [hours]')
    ax_pos.set_ylabel('Position Error [m]')
    if log_scale:
        ax_pos.set_yscale("log")
        ax_pos.set_ylim(0.0001, ax_pos.get_ylim()[1])

    ax_pos.minorticks_on()
    ax_pos.grid(which='major', linestyle='-', alpha=0.8)
    ax_pos.grid(which='minor', linestyle=':', alpha=0.5)
    legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='ANN'),
    Line2D([0], [0], color='black', linestyle='--', label='Spherical')
    ]
    ax_pos.legend(handles=legend_elements, loc="lower right")

    # Velocity error
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            v_test = alt_results['ann'].sol(alt_t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax_vel.plot(alt_t_eval / 3600, vel_error, '-', #label="ANN" if i == 0 else None,
                       color=colors_alt[alt], linewidth=1.5)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            v_test = alt_results['spherical'].sol(alt_t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax_vel.plot(alt_t_eval / 3600, vel_error, '--', #label="Spherical" if i == 0 else None,
                        color=colors_alt[alt], linewidth=1.5)
    
    ax_vel.set_xlabel('Time [hours]')
    ax_vel.set_ylabel('Velocity Error [m/s]')
    if log_scale:
        ax_vel.set_yscale("log")
        ax_vel.set_ylim(1e-7, ax_vel.get_ylim()[1])

    ax_vel.minorticks_on()
    ax_vel.grid(which='major', linestyle='-', alpha=0.8)
    ax_vel.grid(which='minor', linestyle=':', alpha=0.5)


    legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='ANN'),
    Line2D([0], [0], color='black', linestyle='--', label='Spherical')
    ]
    ax_vel.legend(handles=legend_elements, loc="lower right")

    
    # Single legend at top with 3 columns
    legend_elements = [Line2D([0], [0], color=colors_alt[alt], lw=2, label=f'{alt} km') 
                       for alt in altitudes]

    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 0.14), frameon=True, title='Altitude')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.22)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Error comparison plot saved to: {save_path}")

def plot_pos_vel_error_(results_dict, t_eval, save_path='orbit_comparison_error.png'):
    """Position and velocity errors with two legend boxes"""
    from matplotlib.lines import Line2D
    
    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(5, 6))
    colors_alt = {'300': '#D87F7F', '400': '#A4B86E', '500': '#6B8CD4'}
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'])
    
    # Position error
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            r_test = alt_results['spherical'].sol(alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax_pos.plot(alt_t_eval / 3600, pos_error / 1000, '--',
                       color=colors_alt[alt], linewidth=1.5)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            r_test = alt_results['ann'].sol(alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax_pos.plot(alt_t_eval / 3600, pos_error / 1000, '-',
                       color=colors_alt[alt], linewidth=1.5)
    
    ax_pos.set_xlabel('Time [hours]')
    ax_pos.set_ylabel('Position Error [km]')
    ax_pos.set_title('Position Error')
    #ax_pos.set_yscale('log')
    ax_pos.grid(True, alpha=0.3)
    
    # Create two legend boxes for position
    legend1_elements = [Line2D([0], [0], color=colors_alt[alt], lw=2, label=f'{alt} km') 
                       for alt in altitudes]
    legend2_elements = [Line2D([0], [0], color='black', lw=2, linestyle='-', label='ANN'),
                       Line2D([0], [0], color='black', lw=2, linestyle='--', label='Spherical')]
    
    
    # Velocity error
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            v_test = alt_results['spherical'].sol(alt_t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax_vel.plot(alt_t_eval / 3600, vel_error, '--',
                       color=colors_alt[alt], linewidth=1.5)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            v_test = alt_results['ann'].sol(alt_t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax_vel.plot(alt_t_eval / 3600, vel_error, '-',
                       color=colors_alt[alt], linewidth=1.5)
    
    ax_vel.set_xlabel('Time [hours]')
    ax_vel.set_ylabel('Velocity Error [m/s]')
    ax_vel.set_title('Velocity Error')
    #ax_vel.set_yscale('log')
    ax_vel.grid(True, alpha=0.3)
    
    # Create two legend boxes for velocity
    # For position subplot
    legend1 = ax_pos.legend(handles=legend1_elements, loc='center left', 
                            bbox_to_anchor=(1, 0.65), title='Altitude')
    ax_pos.add_artist(legend1)
    ax_pos.legend(handles=legend2_elements, loc='center left', 
                    bbox_to_anchor=(1, 0.35), title='Model')

    # For velocity subplot
    legend1 = ax_vel.legend(handles=legend1_elements, loc='center left', 
                            bbox_to_anchor=(1, 0.65), title='Altitude')
    ax_vel.add_artist(legend1)
    ax_vel.legend(handles=legend2_elements, loc='center left', 
                    bbox_to_anchor=(1, 0.35), title='Model')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Error comparison plot saved to: {save_path}")


def plot_orbit_hill_frame(results_dict, t_eval, save_path='hill_orbit_comparison.png'):
    """Hill frame 3D plot - relative position with time colormap"""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'])
    colors_alt = {'300': '#1f77b4', '400': '#ff7f0e', '500': '#2ca02c'}
    
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Relative Position - Hill Frame")
    altitude_hatch = {'300': '*', '400': '-', '500': 'o'}
    for i, model_name in enumerate(['spherical', 'ann']):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        
        for alt in altitudes:
            alt_results = results_dict[alt]
            alt_t_eval = alt_results.get('t_eval', t_eval)
            
            if 'ground_truth' not in alt_results or model_name not in alt_results:
                continue
            
            r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
            v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T
            r_test = alt_results[model_name].sol(alt_t_eval)[:3, :].T
            
            rel_pos = r_test - r_true
            rho_list = np.array([get_hill_frame_ON(r_c, v_c) @ r_d 
                                for r_d, r_c, v_c in zip(rel_pos, r_true, v_true)]) / 1000
            
            points = rho_list.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap='inferno', linewidth=1.5, hatch=altitude_hatch[alt])
            lc.set_array(alt_t_eval[:-1] / 3600)
            ax.add_collection(lc)
        
        labels_model = {'spherical': 'Spherical Model', 'ann': 'ANN Model'}
        ax.set_title(labels_model[model_name])
        ax.set_xlabel('Radial [km]')
        ax.set_ylabel('Along Track [km]')
        ax.set_zlabel('Cross Track [km]')
    
    plt.subplots_adjust(right=0.98)
    
    # Add colorbar
    cbar = fig.colorbar(lc, ax=fig.get_axes(), shrink=0.8, aspect=20)
    cbar.set_label('Time [h]', rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Hill frame saved to: {save_path}")

def plot_error_statistics(results_dict, t_eval, save_path='error_statistics.png'):
    """
    Error statistics - all altitudes together
    """
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'])
    colors_alt = {'300': '#1f77b4', '400': '#ff7f0e', '500': '#2ca02c'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Spherical - Position components
    ax = axes[0, 0]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            r_true = _get_data(alt_results['ground_truth'], alt_t_eval)[:3, :].T
            r_test = _get_data(alt_results['spherical'], alt_t_eval)[:3, :].T
            error_r = r_test - r_true
            ax.plot(alt_t_eval / 3600, np.linalg.norm(error_r, axis=1) / 1000,
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('Spherical - Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spherical - Velocity components
    ax = axes[0, 1]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            v_true = _get_data(alt_results['ground_truth'], alt_t_eval)[3:, :].T
            v_test = _get_data(alt_results['spherical'], alt_t_eval)[3:, :].T
            error_v = v_test - v_true
            ax.plot(alt_t_eval / 3600, np.linalg.norm(error_v, axis=1),
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Velocity Error [m/s]')
    ax.set_title('Spherical - Velocity Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spherical - Combined
    ax = axes[0, 2]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'spherical' in alt_results:
            r_true = _get_data(alt_results['ground_truth'], alt_t_eval)[:3, :].T
            r_test = _get_data(alt_results['spherical'], alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax.plot(alt_t_eval / 3600, pos_error / 1000,
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('Spherical - Total Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ANN - Position components
    ax = axes[1, 0]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            r_true = _get_data(alt_results['ground_truth'], alt_t_eval)[:3, :].T
            r_test = _get_data(alt_results['ann'], alt_t_eval)[:3, :].T
            error_r = r_test - r_true
            ax.plot(alt_t_eval / 3600, np.linalg.norm(error_r, axis=1) / 1000,
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('ANN - Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ANN - Velocity components
    ax = axes[1, 1]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            v_true = _get_data(alt_results['ground_truth'], alt_t_eval)[3:, :].T
            v_test = _get_data(alt_results['ann'], alt_t_eval)[3:, :].T
            error_v = v_test - v_true
            ax.plot(alt_t_eval / 3600, np.linalg.norm(error_v, axis=1),
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Velocity Error [m/s]')
    ax.set_title('ANN - Velocity Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ANN - Combined
    ax = axes[1, 2]
    for alt in altitudes:
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        if 'ground_truth' in alt_results and 'ann' in alt_results:
            r_true = _get_data(alt_results['ground_truth'], alt_t_eval)[:3, :].T
            r_test = _get_data(alt_results['ann'], alt_t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax.plot(alt_t_eval / 3600, pos_error / 1000,
                   color=colors_alt[alt], label=f'{alt} km', linewidth=1.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('ANN - Total Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Error statistics saved to: {save_path}")

if __name__ == '__main__':
    pass
