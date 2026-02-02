"""
ComIn plugin for extracting 5-minutes precipitation datasets for AES physics.
AES does not have a built-in algorithm for accumulating precipitation, so we
do it here.

This version includes support for geographic bounding box filtering, allowing
users to specify a lat/lon box to restrict output to a specific region.

NOTE: Works only for AES.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import sys
import comin
import argparse
from datetime import datetime
import numpy as np
from mpi4py import MPI

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

EPSILON = 1e-6 # Tolerance for the floor

parser = argparse.ArgumentParser()

parser.add_argument("--interval", type=int, default=None,
                    help="Specify the accumulation interval in seconds.")
parser.add_argument("--floor", type=float, default=None,
                    help="Set a floor in kg/m2 at or below which the accumulated precipitation is masked (set to NaN or zero).")
parser.add_argument("--floor_to_zero", action="store_true", default=False,
                    help="Set the values below the floor to zero. Default: set to NaN.")
parser.add_argument("--no_land_mask", action="store_true", default=False,
                    help="Disable land masking. By default, ocean cells are masked out before output.")
parser.add_argument("--lon_min", type=float, default=None,
                    help="Western boundary of bounding box (degrees, -180 to 180).")
parser.add_argument("--lon_max", type=float, default=None,
                    help="Eastern boundary of bounding box (degrees, -180 to 180).")
parser.add_argument("--lat_min", type=float, default=None,
                    help="Southern boundary of bounding box (degrees, -90 to 90).")
parser.add_argument("--lat_max", type=float, default=None,
                    help="Northern boundary of bounding box (degrees, -90 to 90).")

args = parser.parse_args(comin.current_get_plugin_info().args)

if args.interval is None:
    accumulation_interval = 300
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: No precip interval is specified. Using default value of 300 seconds.",
              file=sys.stderr)
else:
    accumulation_interval = args.interval
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: precipitation interval set to {accumulation_interval} seconds.",
              file=sys.stderr)

if args.floor is None:
    floor = 1e-5
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: No floor is specified. Using default value of 1e-5 kg/m2.",
              file=sys.stderr)
else:
    floor = args.floor
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: Setting precipitation floor to {floor} kg/m2.",
              file=sys.stderr)

floor += EPSILON

use_land_mask = not args.no_land_mask
if rank == 0:
    if use_land_mask:
        print(f"ComIn - precipitation_accumulation_aes.py: Land mask enabled. Ocean cells will be masked out.", file=sys.stderr)
    else:
        print(f"ComIn - precipitation_accumulation_aes.py: Land mask disabled. All cells will be included.", file=sys.stderr)

if args.floor_to_zero:
    floor_value = 0.0
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: Values below {floor} kg/m2 will be set to {floor_value}.",
              file=sys.stderr)
else:
    floor_value = np.nan
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: Values below {floor} kg/m2 will be set to {floor_value}.",
              file=sys.stderr)

# Bounding box configuration
bbox_args = [args.lon_min, args.lon_max, args.lat_min, args.lat_max]
bbox_specified = [arg is not None for arg in bbox_args]

if any(bbox_specified) and not all(bbox_specified):
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: ERROR - Bounding box requires all four corners "
              f"(--lon_min, --lon_max, --lat_min, --lat_max). Only partial specification provided.",
              file=sys.stderr)
    comin.finish("precipitation_accumulation_aes.py", "Incomplete bounding box specification")

use_bounding_box = all(bbox_specified)
if use_bounding_box:
    lon_min, lon_max, lat_min, lat_max = args.lon_min, args.lon_max, args.lat_min, args.lat_max
    # Validate latitude range
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        if rank == 0:
            print(f"ComIn - precipitation_accumulation_aes.py: ERROR - Latitude must be between -90 and 90 degrees.",
                  file=sys.stderr)
        comin.finish("precipitation_accumulation_aes.py", "Invalid latitude range")
    if lat_min >= lat_max:
        if rank == 0:
            print(f"ComIn - precipitation_accumulation_aes.py: ERROR - lat_min ({lat_min}) must be less than lat_max ({lat_max}).",
                  file=sys.stderr)
        comin.finish("precipitation_accumulation_aes.py", "Invalid latitude range: lat_min >= lat_max")
    # Check for date line crossing (lon_min > lon_max means box crosses 180° meridian)
    crosses_dateline = lon_min > lon_max
    if rank == 0:
        if crosses_dateline:
            print(f"ComIn - precipitation_accumulation_aes.py: Bounding box enabled (crosses date line): "
                  f"lon=[{lon_min}, 180] U [-180, {lon_max}], lat=[{lat_min}, {lat_max}]",
                  file=sys.stderr)
        else:
            print(f"ComIn - precipitation_accumulation_aes.py: Bounding box enabled: "
                  f"lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]",
                  file=sys.stderr)
else:
    lon_min = lon_max = lat_min = lat_max = None
    crosses_dateline = False
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: Bounding box disabled. All cells will be included.",
              file=sys.stderr)

# domain
jg = 1
domain = comin.descrdata_get_domain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)

# Create bounding box mask from cell coordinates
if use_bounding_box:
    # Cell coordinates are in radians, convert to degrees for comparison
    clon_rad = np.asarray(domain.cells.clon)
    clat_rad = np.asarray(domain.cells.clat)
    clon_deg = np.rad2deg(clon_rad)
    clat_deg = np.rad2deg(clat_rad)

    # Create latitude mask (always straightforward)
    lat_mask = (clat_deg >= lat_min) & (clat_deg <= lat_max)

    # Create longitude mask (handle date line crossing)
    if crosses_dateline:
        # Box crosses 180° meridian: include lon >= lon_min OR lon <= lon_max
        lon_mask = (clon_deg >= lon_min) | (clon_deg <= lon_max)
    else:
        # Normal box: include lon_min <= lon <= lon_max
        lon_mask = (clon_deg >= lon_min) & (clon_deg <= lon_max)

    # Combined bounding box mask (True = inside the box)
    bbox_mask = lat_mask & lon_mask

else:
    bbox_mask = None

# accumulated precipitation variable
vd_prec_accumulated = ("tot_prec_acc", jg)
comin.var_request_add(vd_prec_accumulated, lmodexclusive=False)
comin.metadata_set(vd_prec_accumulated, 
                   hgrid_id=1,
                   zaxis_id=comin.COMIN_ZAXIS_2D,
                   standard_name='accumulated_precipitation',
                   long_name='Total accumulated precipitation',
                   units='kg m-2')

# variable for tracking accumulation interval
vd_prec_timer = ("prec_timer", jg)
comin.var_request_add(vd_prec_timer, lmodexclusive=False)
comin.metadata_set(vd_prec_timer,
                   hgrid_id=1,
                   zaxis_id=comin.COMIN_ZAXIS_2D,
                   standard_name='precipitation_timer',
                   long_name='Timer for precipitation accumulation interval',
                   units='s')

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def precipitation_constructor():
    # access the variables
    global tot_prec_acc, pr_var, prec_timer, sftlf_var
    tot_prec_acc = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("tot_prec_acc", jg), flag=comin.COMIN_FLAG_WRITE)
    pr_var = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("pr", jg), flag=comin.COMIN_FLAG_READ)
    prec_timer = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("prec_timer", jg), flag=comin.COMIN_FLAG_WRITE)
    if use_land_mask:
        sftlf_var = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("sftlf", jg), flag=comin.COMIN_FLAG_READ)

    # Initialize timer and accumulator to zero
    prec_timer_np = np.squeeze(np.asarray(prec_timer))
    prec_timer_np[:] = 0.0

    tot_prec_acc_np = np.squeeze(np.asarray(tot_prec_acc))
    tot_prec_acc_np[:] = 0.0


@comin.register_callback(comin.EP_ATM_PHYSICS_AFTER)
def accumulate_precipitation():
    """
    Accumulate precipitation using pr flux and reset every accumulation_interval seconds
    """
    # Mask for prognostic cells
    mask_2d = (decomp_domain_np != 0)

    # Get physics time step
    dt_phy_sec = comin.descrdata_get_timesteplength(jg)

    # Get variables as numpy arrays
    pr_flux = np.ma.masked_array(np.squeeze(pr_var), mask=mask_2d)
    timer = np.ma.masked_array(np.squeeze(prec_timer), mask=mask_2d)
    # Accumulated precipitation
    tot_prec_acc_np = np.squeeze(np.asarray(tot_prec_acc))

    # Check if we need to reset (where timer >= accumulation_interval)
    reset_mask = timer >= accumulation_interval
    if np.any(reset_mask):
        # Only reset cells that are within the active region (inside bbox if enabled)
        # This preserves NaN values for masked-out cells
        if use_bounding_box and bbox_mask is not None:
            active_reset_mask = reset_mask & bbox_mask
        else:
            active_reset_mask = reset_mask
        tot_prec_acc_np[active_reset_mask] = 0.0
        timer[reset_mask] = 0.0

    timer[:] = timer[:] + dt_phy_sec
    prec_increment = pr_flux * dt_phy_sec
    tot_prec_acc_np[:] = tot_prec_acc_np[:] + prec_increment

@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def precipitation_floor():
    """
    Apply geographic masking (bounding box, land mask) and precipitation floor before writing output.

    Masking order:
    1. Bounding box mask (if enabled) - mask cells outside the specified region
    2. Land mask (if enabled) - mask ocean cells
    3. Floor threshold - set values below threshold to floor_value
    """
    current_time = comin.current_get_datetime()
    current_datetime = datetime.fromisoformat(current_time)
    seconds = current_datetime.minute*60 + current_datetime.second

    # Are we writing output now?
    if seconds % accumulation_interval == 0:
        tot_prec_acc_np = np.squeeze(np.asarray(tot_prec_acc))

        # Apply bounding box mask first (mask cells outside the box)
        if use_bounding_box and bbox_mask is not None:
            tot_prec_acc_np[~bbox_mask] = np.nan

        # Apply land mask (mask ocean cells)
        if use_land_mask:
            sftlf_np = np.squeeze(np.asarray(sftlf_var))
            land_mask = sftlf_np > 0.0
            tot_prec_acc_np[~land_mask] = np.nan

        # Clamp negative values to zero BEFORE applying floor
        # (handles floating-point noise in precipitation flux)
        negative_mask = tot_prec_acc_np < 0.0
        tot_prec_acc_np[negative_mask] = 0.0

        # Apply floor threshold
        floor_mask = tot_prec_acc_np <= floor
        if np.any(floor_mask):
            tot_prec_acc_np[floor_mask] = floor_value

@comin.register_callback(comin.EP_DESTRUCTOR)
def precipitation_destructor():
    if 'tot_prec_acc' in globals() and tot_prec_acc is not None:
        if rank == 0:
            print(f"ComIn - precipitation_accumulation_aes.py: Plugin finished.", 
                  file=sys.stderr)
