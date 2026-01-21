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
CUTIN = 0.0  # Cut-in speed in m/s
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

# =============================================================================
# Precomputed arrays (initialized in constructor, used every timestep)
# These are constant for the entire simulation and computed once for efficiency
# =============================================================================
PRECOMPUTED = {
    'initialized': False,
    'mask_1d': None,           # Boolean mask for halo cells (ncells,)
    'valid_mask': None,        # Inverse of mask_1d (ncells,)
    'dt_phy_sec': None,        # Physics timestep [s]
    'topo_1d': None,           # Flattened topography (ncells,)
    'z_agl': None,             # Height above ground level (ncells, nlev)
    'ncells': None,            # Total number of cells
    'nlev_local': None,        # Number of vertical levels
    'idx': None,               # Index array for advanced indexing (ncells,)
    'is_3d': None,             # Whether input arrays are 3D
    'nproma': None,            # Cells per block
    'nblks': None,             # Number of blocks
    # Per-height precomputed interpolation data
    'interp_50m': None,        # Dict with k_upper, k_lower, weight for 50m
    'interp_100m': None,       # Dict with k_upper, k_lower, weight for 100m
    'interp_120m': None,       # Dict with k_upper, k_lower, weight for 120m
}


def precompute_interpolation_indices(z_agl, target_height, ncells, nlev_local):
    """
    Precompute the bracketing level indices and interpolation weights for a target height.

    These values are constant because z_agl (model level heights above ground)
    and target_height don't change during the simulation.

    Parameters
    ----------
    z_agl : np.ndarray
        Height above ground level, shape (ncells, nlev)
    target_height : float
        Target height in meters
    ncells : int
        Number of horizontal cells
    nlev_local : int
        Number of vertical levels

    Returns
    -------
    dict with keys: k_upper, k_lower, weight
    """
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

    return {
        'k_upper': k_upper,
        'k_lower': k_lower,
        'weight': weight,
    }


def interpolate_wind_with_precomputed(u_2d, v_2d, interp_data, valid_mask, idx, ncells):
    """
    Interpolate wind using precomputed indices and weights.

    This is much faster than recomputing the bracketing levels every timestep.

    Parameters
    ----------
    u_2d : np.ndarray
        Eastward wind component, shape (ncells, nlev)
    v_2d : np.ndarray
        Northward wind component, shape (ncells, nlev)
    interp_data : dict
        Precomputed interpolation data with k_upper, k_lower, weight
    valid_mask : np.ndarray
        Boolean mask for valid cells (ncells,)
    idx : np.ndarray
        Index array (ncells,)
    ncells : int
        Number of cells

    Returns
    -------
    wind_speed : np.ndarray
        Wind speed at target height, shape (ncells,)
    """
    k_upper = interp_data['k_upper']
    k_lower = interp_data['k_lower']
    weight = interp_data['weight']

    # Initialize output arrays
    u_interp = np.full(ncells, np.nan)
    v_interp = np.full(ncells, np.nan)

    if not np.any(valid_mask):
        return np.full(ncells, np.nan)

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
    Access ICON variables and precompute constant arrays.

    This function runs once at the beginning of the simulation and precomputes
    all arrays that don't change during the simulation:
    - Mask arrays (based on domain decomposition)
    - Physics timestep
    - Topography
    - Height above ground level for all model levels
    - Interpolation indices and weights for each target height
    """
    global wind_50m, wind_100m, wind_120m, wind_avg_timer
    global u_3d, v_3d, z_mc, topography_c
    global COUNT, AVG_INTERVAL, PRECOMPUTED

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

    # =========================================================================
    # PRECOMPUTE CONSTANT ARRAYS
    # =========================================================================

    # 1. Mask arrays (constant - based on domain decomposition)
    mask_2d = (decomp_domain_np != 0)
    PRECOMPUTED['mask_1d'] = mask_2d.flatten()
    PRECOMPUTED['valid_mask'] = ~PRECOMPUTED['mask_1d']

    # 2. Physics timestep (constant for the simulation)
    PRECOMPUTED['dt_phy_sec'] = comin.descrdata_get_timesteplength(jg)

    # 3. Get geometry arrays and determine shapes
    z_mc_np = np.squeeze(np.asarray(z_mc))
    topo_np = np.squeeze(np.asarray(topography_c))

    if z_mc_np.ndim == 3:
        nproma, nlev_local, nblks = z_mc_np.shape
        PRECOMPUTED['is_3d'] = True
        PRECOMPUTED['nproma'] = nproma
        PRECOMPUTED['nblks'] = nblks
        PRECOMPUTED['nlev_local'] = nlev_local
        PRECOMPUTED['ncells'] = nproma * nblks

        # Reshape z_mc to (ncells, nlev)
        z_mc_2d = z_mc_np.transpose(0, 2, 1).reshape(-1, nlev_local)
        # Flatten topography to (ncells,)
        topo_1d = topo_np.flatten()
    else:
        # 2D case (single block)
        PRECOMPUTED['is_3d'] = False
        PRECOMPUTED['nlev_local'] = z_mc_np.shape[1] if z_mc_np.ndim == 2 else 1
        PRECOMPUTED['ncells'] = z_mc_np.shape[0] if z_mc_np.ndim >= 1 else 1
        z_mc_2d = z_mc_np
        topo_1d = topo_np.flatten()

    PRECOMPUTED['topo_1d'] = topo_1d

    # 4. Compute height above ground level (constant - model levels don't move)
    PRECOMPUTED['z_agl'] = z_mc_2d - topo_1d[:, np.newaxis]

    # 5. Index array for advanced indexing
    PRECOMPUTED['idx'] = np.arange(PRECOMPUTED['ncells'])

    # 6. Precompute interpolation indices and weights for each target height
    ncells = PRECOMPUTED['ncells']
    nlev_local = PRECOMPUTED['nlev_local']
    z_agl = PRECOMPUTED['z_agl']

    PRECOMPUTED['interp_50m'] = precompute_interpolation_indices(z_agl, 50.0, ncells, nlev_local)
    PRECOMPUTED['interp_100m'] = precompute_interpolation_indices(z_agl, 100.0, ncells, nlev_local)
    PRECOMPUTED['interp_120m'] = precompute_interpolation_indices(z_agl, 120.0, ncells, nlev_local)

    PRECOMPUTED['initialized'] = True

    if rank == 0:
        print(f"ComIn - wind_3d.py: Constructor completed. nlev={nlev}, heights={HEIGHT_LEVELS}",
              file=sys.stderr)
        print(f"ComIn - wind_3d.py: Precomputed arrays initialized. ncells={ncells}, "
              f"is_3d={PRECOMPUTED['is_3d']}", file=sys.stderr)


@comin.register_callback(comin.EP_ATM_PHYSICS_AFTER)
def accumulate_wind():
    """
    Interpolate wind to target heights and accumulate for time averaging.

    This function uses precomputed arrays for efficiency. Only the wind
    components (u, v) are read fresh each timestep.
    """
    global COUNT

    # Retrieve precomputed constants
    mask_1d = PRECOMPUTED['mask_1d']
    valid_mask = PRECOMPUTED['valid_mask']
    dt_phy_sec = PRECOMPUTED['dt_phy_sec']
    idx = PRECOMPUTED['idx']
    ncells = PRECOMPUTED['ncells']
    is_3d = PRECOMPUTED['is_3d']
    nlev_local = PRECOMPUTED['nlev_local']

    # Get timer as numpy array - flatten to 1D
    timer_raw = np.asarray(wind_avg_timer)
    timer = np.squeeze(timer_raw).flatten()

    # Get output wind arrays - flatten to 1D
    wind_50m_np = np.squeeze(np.asarray(wind_50m)).flatten()
    wind_100m_np = np.squeeze(np.asarray(wind_100m)).flatten()
    wind_120m_np = np.squeeze(np.asarray(wind_120m)).flatten()

    # Get wind components (these change every timestep)
    u_np = np.squeeze(np.asarray(u_3d))
    v_np = np.squeeze(np.asarray(v_3d))

    # Reshape wind arrays to (ncells, nlev)
    if is_3d:
        nproma = PRECOMPUTED['nproma']
        nblks = PRECOMPUTED['nblks']
        u_2d = u_np.transpose(0, 2, 1).reshape(-1, nlev_local)
        v_2d = v_np.transpose(0, 2, 1).reshape(-1, nlev_local)
    else:
        u_2d = u_np
        v_2d = v_np

    # Check if we need to reset (where timer >= AVG_INTERVAL)
    reset_mask = timer >= AVG_INTERVAL
    if np.any(reset_mask):
        wind_50m_np[reset_mask] = 0.0
        wind_100m_np[reset_mask] = 0.0
        wind_120m_np[reset_mask] = 0.0
        timer[reset_mask] = 0.0
        COUNT = 0

    # Update timer
    timer[:] = timer[:] + dt_phy_sec
    timer_raw.flat[:] = timer

    # Interpolate wind using precomputed indices and weights
    wind_50m_instant = interpolate_wind_with_precomputed(
        u_2d, v_2d, PRECOMPUTED['interp_50m'], valid_mask, idx, ncells)
    wind_100m_instant = interpolate_wind_with_precomputed(
        u_2d, v_2d, PRECOMPUTED['interp_100m'], valid_mask, idx, ncells)
    wind_120m_instant = interpolate_wind_with_precomputed(
        u_2d, v_2d, PRECOMPUTED['interp_120m'], valid_mask, idx, ncells)

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
