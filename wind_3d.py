"""
ComIn plugin for computing time-averaged wind speed at multiple heights using
3D wind field interpolation.

This plugin interpolates wind components (u, v) from ICON's 3D model levels
to specified heights above ground level (50m, 100m, 120m), then computes
wind speed and applies temporal averaging.

NOTE: Works only for AES physics.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import sys
import comin
from datetime import datetime
import numpy as np
from mpi4py import MPI


comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

# Configuration constants
COUNT = 0  # Count to keep track for the averaging
CUTIN = 5.0  # Cut-in speed in m/s
CUTIN_VALUE = np.nan  # Value to assign when wind < CUTIN
AVG_INTERVAL = 300  # Averaging interval in seconds
HEIGHT_LEVELS = [50, 100, 120]  # Target heights above ground in meters

# Domain
jg = 1
domain = comin.descrdata_get_domain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)
nlev = domain.nlev  # Number of vertical levels

# Register output variables for wind speed at different heights
for height in HEIGHT_LEVELS:
    heightstr = str(height) + "m"
    vd_wind_comin = ("wind_" + heightstr, jg)
    comin.var_request_add(vd_wind_comin, lmodexclusive=False)
    comin.metadata_set(vd_wind_comin,
                       hgrid_id=1,
                       zaxis_id=comin.COMIN_ZAXIS_2D,
                       standard_name='wind_speed_at_' + heightstr,
                       long_name='Wind speed at ' + heightstr + ' above ground (interpolated from 3D fields)',
                       units='m s-1')

# Variable for tracking accumulation interval
vd_timer = ("wind_avg_timer", jg)
comin.var_request_add(vd_timer, lmodexclusive=False)
comin.metadata_set(vd_timer,
                   hgrid_id=1,
                   zaxis_id=comin.COMIN_ZAXIS_2D,
                   standard_name='wind_averaging_timer',
                   long_name='Timer for wind averaging interval',
                   units='s')


def interpolate_wind_to_height(u_3d, v_3d, z_agl, target_height, mask_2d):
    """
    Interpolate u and v wind components to a target height above ground level.

    Parameters
    ----------
    u_3d : np.ndarray
        Eastward wind component (nproma, nlev)
    v_3d : np.ndarray
        Northward wind component (nproma, nlev)
    z_agl : np.ndarray
        Height above ground level for each model level (nproma, nlev)
        Note: levels are ordered from top (index 0) to bottom (index nlev-1)
    target_height : float
        Target height above ground in meters
    mask_2d : np.ndarray
        Boolean mask for valid (prognostic) cells

    Returns
    -------
    wind_speed : np.ndarray
        Wind speed at target height (nproma,)
    """
    nproma = u_3d.shape[0]
    u_interp = np.zeros(nproma)
    v_interp = np.zeros(nproma)

    # Loop over horizontal points
    # Note: ICON levels are ordered top-to-bottom, so level 0 is highest, nlev-1 is lowest
    for jc in range(nproma):
        if mask_2d[jc]:
            # Skip masked (halo) cells
            u_interp[jc] = np.nan
            v_interp[jc] = np.nan
            continue

        # Find the two levels bracketing the target height
        # z_agl[jc, :] goes from high altitude (index 0) to low altitude (index nlev-1)
        z_col = z_agl[jc, :]

        # Check if target height is within the model domain
        if target_height > z_col[0]:
            # Target is above highest model level - extrapolate from top two levels
            k_upper = 0
            k_lower = 1
        elif target_height < z_col[-1]:
            # Target is below lowest model level - extrapolate from bottom two levels
            k_upper = len(z_col) - 2
            k_lower = len(z_col) - 1
        else:
            # Find bracketing levels
            # We want k_upper where z_col[k_upper] > target_height > z_col[k_lower]
            for k in range(len(z_col) - 1):
                if z_col[k] >= target_height >= z_col[k + 1]:
                    k_upper = k
                    k_lower = k + 1
                    break

        # Linear interpolation weight
        z_upper = z_col[k_upper]
        z_lower = z_col[k_lower]

        if abs(z_upper - z_lower) < 1e-6:
            # Levels too close, just use upper level
            weight = 1.0
        else:
            weight = (target_height - z_lower) / (z_upper - z_lower)

        # Interpolate u and v
        u_interp[jc] = weight * u_3d[jc, k_upper] + (1.0 - weight) * u_3d[jc, k_lower]
        v_interp[jc] = weight * v_3d[jc, k_upper] + (1.0 - weight) * v_3d[jc, k_lower]

    # Compute wind speed
    wind_speed = np.sqrt(u_interp**2 + v_interp**2)

    return wind_speed


def interpolate_wind_to_height_vectorized(u_2d, v_2d, z_agl, target_height, mask_1d):
    """
    Vectorized version: Interpolate u and v wind components to a target height.

    Parameters
    ----------
    u_2d : np.ndarray
        Eastward wind component, shape (ncells, nlev)
    v_2d : np.ndarray
        Northward wind component, shape (ncells, nlev)
    z_agl : np.ndarray
        Height above ground level, shape (ncells, nlev)
        Levels ordered top-to-bottom (index 0 = highest)
    target_height : float
        Target height in meters above ground
    mask_1d : np.ndarray
        Boolean mask, shape (ncells,). True = masked/invalid cell

    Returns
    -------
    wind_speed : np.ndarray
        Wind speed at target height, shape (ncells,)
    """
    ncells, nlev_local = u_2d.shape

    # Initialize output arrays
    u_interp = np.full(ncells, np.nan)
    v_interp = np.full(ncells, np.nan)

    # Only process valid (non-masked) cells
    valid_mask = ~mask_1d

    if not np.any(valid_mask):
        return np.full(ncells, np.nan)

    # Create a boolean array: True where z_agl >= target_height
    above_target = z_agl >= target_height  # (ncells, nlev)

    # Count how many levels are above target for each column
    n_above = np.sum(above_target, axis=1)  # (ncells,)

    # k_lower is the first level below target
    k_lower = np.clip(n_above, 1, nlev_local - 1)
    k_upper = k_lower - 1

    # Handle edge cases
    above_all = n_above == 0
    k_upper[above_all] = 0
    k_lower[above_all] = 1

    below_all = n_above >= nlev_local
    k_upper[below_all] = nlev_local - 2
    k_lower[below_all] = nlev_local - 1

    # Get heights at bracketing levels using advanced indexing
    idx = np.arange(ncells)
    z_upper = z_agl[idx, k_upper]
    z_lower = z_agl[idx, k_lower]

    # Compute interpolation weights
    dz = z_upper - z_lower
    dz = np.where(np.abs(dz) < 1e-6, 1.0, dz)
    weight = (target_height - z_lower) / dz
    weight = np.clip(weight, 0.0, 1.0)

    # Get wind components at bracketing levels
    u_upper = u_2d[idx, k_upper]
    u_lower = u_2d[idx, k_lower]
    v_upper = v_2d[idx, k_upper]
    v_lower = v_2d[idx, k_lower]

    # Interpolate
    u_interp[valid_mask] = (weight * u_upper + (1.0 - weight) * u_lower)[valid_mask]
    v_interp[valid_mask] = (weight * v_upper + (1.0 - weight) * v_lower)[valid_mask]

    # Compute wind speed
    wind_speed = np.sqrt(u_interp**2 + v_interp**2)

    return wind_speed


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def wind_3d_constructor():
    """
    Access ICON variables needed for wind interpolation.
    """
    global wind_50m, wind_100m, wind_120m, wind_avg_timer
    global u_3d, v_3d, z_mc, topography_c
    global COUNT, AVG_INTERVAL

    # Output variables (wind at different heights)
    wind_50m = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("wind_50m", jg), flag=comin.COMIN_FLAG_WRITE)
    wind_100m = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("wind_100m", jg), flag=comin.COMIN_FLAG_WRITE)
    wind_120m = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("wind_120m", jg), flag=comin.COMIN_FLAG_WRITE)

    # Timer for averaging
    wind_avg_timer = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("wind_avg_timer", jg), flag=comin.COMIN_FLAG_WRITE)

    # Input variables from ICON (3D wind components and geometry)
    u_3d = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("u", jg), flag=comin.COMIN_FLAG_READ)
    v_3d = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("v", jg), flag=comin.COMIN_FLAG_READ)
    z_mc = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("z_mc", jg), flag=comin.COMIN_FLAG_READ)
    topography_c = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("topography_c", jg), flag=comin.COMIN_FLAG_READ)

    if rank == 0:
        print(f"ComIn - wind_3d.py: Constructor completed. nlev={nlev}, heights={HEIGHT_LEVELS}",
              file=sys.stderr)


@comin.register_callback(comin.EP_ATM_PHYSICS_AFTER)
def accumulate_wind():
    """
    Interpolate wind to target heights and accumulate for time averaging.
    """
    global COUNT

    # Mask for prognostic cells (decomp_domain != 0 means halo/boundary)
    # decomp_domain_np has shape (nproma, nblks) for 2D fields
    mask_2d = (decomp_domain_np != 0)

    # Get physics time step
    dt_phy_sec = comin.descrdata_get_timesteplength(jg)

    # Get timer as numpy array - flatten to 1D
    timer_raw = np.asarray(wind_avg_timer)
    timer = np.squeeze(timer_raw).flatten()
    mask_1d = mask_2d.flatten()

    # Get output wind arrays - flatten to 1D
    wind_50m_np = np.squeeze(np.asarray(wind_50m)).flatten()
    wind_100m_np = np.squeeze(np.asarray(wind_100m)).flatten()
    wind_120m_np = np.squeeze(np.asarray(wind_120m)).flatten()

    # Get 3D input arrays and squeeze out singleton dimensions
    # ICON shape is (nproma, nlev, nblks, 1, 1) - squeeze to (nproma, nlev, nblks)
    u_np = np.squeeze(np.asarray(u_3d))
    v_np = np.squeeze(np.asarray(v_3d))
    z_mc_np = np.squeeze(np.asarray(z_mc))
    topo_np = np.squeeze(np.asarray(topography_c))

    # After squeeze, shapes should be:
    # - 3D fields: (nproma, nlev, nblks)
    # - 2D fields: (nproma, nblks)
    # We need to reshape to (ncells, nlev) where ncells = nproma * nblks

    if u_np.ndim == 3:
        nproma, nlev_local, nblks = u_np.shape
        # Transpose to (nproma, nblks, nlev) then reshape to (ncells, nlev)
        u_2d = u_np.transpose(0, 2, 1).reshape(-1, nlev_local)
        v_2d = v_np.transpose(0, 2, 1).reshape(-1, nlev_local)
        z_mc_2d = z_mc_np.transpose(0, 2, 1).reshape(-1, nlev_local)
    elif u_np.ndim == 2:
        # Already 2D (nproma, nlev) - single block case
        u_2d = u_np
        v_2d = v_np
        z_mc_2d = z_mc_np
    else:
        raise ValueError(f"Unexpected u_np dimensions: {u_np.ndim}")

    # Flatten topography to 1D (ncells,)
    # topo_np should be (nproma, nblks) after squeeze
    topo_1d = topo_np.flatten()

    # Compute height above ground level
    # z_mc_2d is (ncells, nlev), topo_1d is (ncells,)
    z_agl = z_mc_2d - topo_1d[:, np.newaxis]

    # Check if we need to reset (where timer >= AVG_INTERVAL)
    reset_mask = timer >= AVG_INTERVAL
    if np.any(reset_mask):
        wind_50m_np[reset_mask] = 0.0
        wind_100m_np[reset_mask] = 0.0
        wind_120m_np[reset_mask] = 0.0
        timer[reset_mask] = 0.0
        COUNT = 0

    # Update timer (write back to original array)
    timer[:] = timer[:] + dt_phy_sec
    # Write timer back to the ComIn variable
    timer_raw.flat[:] = timer

    # Interpolate wind to each target height
    wind_50m_instant = interpolate_wind_to_height_vectorized(u_2d, v_2d, z_agl, 50.0, mask_1d)
    wind_100m_instant = interpolate_wind_to_height_vectorized(u_2d, v_2d, z_agl, 100.0, mask_1d)
    wind_120m_instant = interpolate_wind_to_height_vectorized(u_2d, v_2d, z_agl, 120.0, mask_1d)

    # Accumulate (handle NaN by treating as 0 for accumulation)
    wind_50m_instant = np.nan_to_num(wind_50m_instant, nan=0.0)
    wind_100m_instant = np.nan_to_num(wind_100m_instant, nan=0.0)
    wind_120m_instant = np.nan_to_num(wind_120m_instant, nan=0.0)

    wind_50m_np[:] = wind_50m_np[:] + wind_50m_instant
    wind_100m_np[:] = wind_100m_np[:] + wind_100m_instant
    wind_120m_np[:] = wind_120m_np[:] + wind_120m_instant

    # Write back to ComIn variables
    np.asarray(wind_50m).flat[:] = wind_50m_np
    np.asarray(wind_100m).flat[:] = wind_100m_np
    np.asarray(wind_120m).flat[:] = wind_120m_np

    COUNT = COUNT + 1


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def average_wind():
    """
    Compute time-averaged wind speed before writing output.
    """
    global COUNT

    current_time = comin.current_get_datetime()
    current_datetime = datetime.fromisoformat(current_time)
    seconds = current_datetime.minute * 60 + current_datetime.second

    # Check if we are at an output time
    if seconds % AVG_INTERVAL == 0 and COUNT > 0:

        # Get output arrays as flat views
        wind_50m_raw = np.asarray(wind_50m)
        wind_100m_raw = np.asarray(wind_100m)
        wind_120m_raw = np.asarray(wind_120m)

        # Compute averages (in-place using flat view)
        wind_50m_raw.flat[:] = wind_50m_raw.flat[:] / COUNT
        wind_100m_raw.flat[:] = wind_100m_raw.flat[:] / COUNT
        wind_120m_raw.flat[:] = wind_120m_raw.flat[:] / COUNT

        # Apply cut-in threshold
        cutin_mask_50 = wind_50m_raw.flat[:] < CUTIN
        cutin_mask_100 = wind_100m_raw.flat[:] < CUTIN
        cutin_mask_120 = wind_120m_raw.flat[:] < CUTIN

        if np.any(cutin_mask_50):
            wind_50m_raw.flat[cutin_mask_50] = CUTIN_VALUE
        if np.any(cutin_mask_100):
            wind_100m_raw.flat[cutin_mask_100] = CUTIN_VALUE
        if np.any(cutin_mask_120):
            wind_120m_raw.flat[cutin_mask_120] = CUTIN_VALUE


@comin.register_callback(comin.EP_DESTRUCTOR)
def wind_3d_destructor():
    """
    Cleanup callback when simulation ends.
    """
    if rank == 0:
        print(f"ComIn - wind_3d.py: Plugin finished.", file=sys.stderr)
