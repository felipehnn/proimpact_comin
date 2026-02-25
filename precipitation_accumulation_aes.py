"""
ComIn plugin for extracting 5-minutes precipitation datasets for AES physics.
AES does not have a built-in algorithm for accumulating precipitation, so we
do it here.

This version includes support for geographic bounding box filtering, allowing
users to specify a lat/lon box to restrict output to a specific region.

NOTE: Works only for AES.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import comin
import numpy as np
from datetime import datetime
from comin_utils import (
    PluginContext, PluginLogger, PluginArgumentParser,
    register_variable, LandMask, to_numpy, to_masked,
    get_interval_with_default
)

# =============================================================================
# Plugin Setup
# =============================================================================

ctx = PluginContext(jg=1)
logger = PluginLogger("precipitation_accumulation_aes.py", ctx)

EPSILON = 1e-6  # Tolerance for the floor

# Argument parsing
parser = PluginArgumentParser()
parser.add_common_args(interval=True, land_mask=True)
parser.add_argument("--floor", type=float, default=None,
                    help="Set a floor in kg/m2 at or below which the accumulated precipitation is masked.")
parser.add_argument("--floor_to_zero", action="store_true", default=False,
                    help="Set the values below the floor to zero. Default: set to NaN.")
parser.add_argument("--no_temperature", action="store_true", default=False,
                    help="Disable output of near-surface temperature at precipitation locations.")
parser.add_argument("--lon_min", type=float, default=None,
                    help="Western boundary of bounding box (degrees, -180 to 180).")
parser.add_argument("--lon_max", type=float, default=None,
                    help="Eastern boundary of bounding box (degrees, -180 to 180).")
parser.add_argument("--lat_min", type=float, default=None,
                    help="Southern boundary of bounding box (degrees, -90 to 90).")
parser.add_argument("--lat_max", type=float, default=None,
                    help="Northern boundary of bounding box (degrees, -90 to 90).")

args = parser.parse()

# Temperature output
store_temperature = not args.no_temperature
if store_temperature:
    logger.info("Temperature output enabled. Use --no_temperature to disable.")
else:
    logger.info("Temperature output disabled.")

# Accumulation interval
accumulation_interval = get_interval_with_default(
    args.interval, default=300, logger=logger, param_name="precip interval"
)

# Floor configuration
if args.floor is None:
    floor = 1e-5
    logger.info(f"No floor specified. Using default value of {floor} kg/m2.")
else:
    floor = args.floor
    logger.info(f"Setting precipitation floor to {floor} kg/m2.")

floor += EPSILON

# Floor value (NaN or zero)
floor_value = 0.0 if args.floor_to_zero else np.nan
logger.info(f"Values below {floor} kg/m2 will be set to {floor_value}.")

# Land mask
land_mask = LandMask(ctx, enabled=not args.no_land_mask, logger=logger)

# =============================================================================
# Bounding Box Configuration
# =============================================================================

bbox_args = [args.lon_min, args.lon_max, args.lat_min, args.lat_max]
bbox_specified = [arg is not None for arg in bbox_args]

if any(bbox_specified) and not all(bbox_specified):
    logger.error("Bounding box requires all four corners "
                 "(--lon_min, --lon_max, --lat_min, --lat_max). Only partial specification provided.")
    comin.finish("precipitation_accumulation_aes.py", "Incomplete bounding box specification")

use_bounding_box = all(bbox_specified)
if use_bounding_box:
    lon_min, lon_max, lat_min, lat_max = args.lon_min, args.lon_max, args.lat_min, args.lat_max

    # Validate latitude range
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        logger.error("Latitude must be between -90 and 90 degrees.")
        comin.finish("precipitation_accumulation_aes.py", "Invalid latitude range")
    if lat_min >= lat_max:
        logger.error(f"lat_min ({lat_min}) must be less than lat_max ({lat_max}).")
        comin.finish("precipitation_accumulation_aes.py", "Invalid latitude range: lat_min >= lat_max")

    # Check for date line crossing (lon_min > lon_max means box crosses 180° meridian)
    crosses_dateline = lon_min > lon_max
    if crosses_dateline:
        logger.info(f"Bounding box enabled (crosses date line): "
                    f"lon=[{lon_min}, 180] U [-180, {lon_max}], lat=[{lat_min}, {lat_max}]")
    else:
        logger.info(f"Bounding box enabled: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")

    # Create bounding box mask from cell coordinates
    lat_mask_arr = (ctx.clat_deg >= lat_min) & (ctx.clat_deg <= lat_max)
    if crosses_dateline:
        lon_mask_arr = (ctx.clon_deg >= lon_min) | (ctx.clon_deg <= lon_max)
    else:
        lon_mask_arr = (ctx.clon_deg >= lon_min) & (ctx.clon_deg <= lon_max)
    bbox_mask = lat_mask_arr & lon_mask_arr
else:
    lon_min = lon_max = lat_min = lat_max = None
    crosses_dateline = False
    bbox_mask = None
    logger.info("Bounding box disabled. All cells will be included.")

# =============================================================================
# Variable Registration
# =============================================================================

register_variable("tot_prec_acc", ctx.jg,
                  standard_name='accumulated_precipitation',
                  long_name='Total accumulated precipitation',
                  units='kg m-2')

register_variable("prec_timer", ctx.jg,
                  standard_name='precipitation_timer',
                  long_name='Timer for precipitation accumulation interval',
                  units='s')

if store_temperature:
    register_variable("tas_prec", ctx.jg,
                      standard_name='air_temperature_at_precipitation',
                      long_name='Near-surface air temperature at precipitation locations',
                      units='K')

# =============================================================================
# Callbacks
# =============================================================================

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def precipitation_constructor():
    global tot_prec_acc, pr_var, prec_timer, tas_var, tas_prec
    tot_prec_acc = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("tot_prec_acc", ctx.jg), flag=comin.COMIN_FLAG_WRITE)
    pr_var = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("pr", ctx.jg), flag=comin.COMIN_FLAG_READ)
    prec_timer = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("prec_timer", ctx.jg), flag=comin.COMIN_FLAG_WRITE)

    # Initialize land mask variable
    land_mask.init_sftlf_var([comin.EP_ATM_PHYSICS_AFTER])

    # Initialize timer and accumulator to zero
    to_numpy(prec_timer)[:] = 0.0
    to_numpy(tot_prec_acc)[:] = 0.0

    if store_temperature:
        tas_var  = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("tas", ctx.jg), flag=comin.COMIN_FLAG_READ)
        tas_prec = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("tas_prec", ctx.jg), flag=comin.COMIN_FLAG_WRITE)
        to_numpy(tas_prec)[:] = np.nan
    else:
        tas_var = tas_prec = None


@comin.register_callback(comin.EP_ATM_PHYSICS_AFTER)
def accumulate_precipitation():
    """Accumulate precipitation using pr flux and reset every accumulation_interval seconds."""
    pr_flux = to_masked(pr_var, ctx.mask_2d)
    timer = to_masked(prec_timer, ctx.mask_2d)
    tot_prec_acc_np = to_numpy(tot_prec_acc)

    # Check if we need to reset (where timer >= accumulation_interval)
    reset_mask = timer >= accumulation_interval
    if np.any(reset_mask):
        # Only reset cells within the active region (inside bbox if enabled)
        if use_bounding_box and bbox_mask is not None:
            active_reset_mask = reset_mask & bbox_mask
        else:
            active_reset_mask = reset_mask
        tot_prec_acc_np[active_reset_mask] = 0.0
        timer[reset_mask] = 0.0

    timer[:] = timer[:] + ctx.dt
    prec_increment = pr_flux * ctx.dt
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
    seconds = current_datetime.minute * 60 + current_datetime.second

    # Are we writing output now?
    if seconds % accumulation_interval == 0:
        tot_prec_acc_np = to_numpy(tot_prec_acc)

        # Apply bounding box mask first (mask cells outside the box)
        if use_bounding_box and bbox_mask is not None:
            tot_prec_acc_np[~bbox_mask] = np.nan

        # Apply land mask (mask ocean cells)
        land_mask.apply(tot_prec_acc_np)

        # Clamp negative values to zero BEFORE applying floor
        # (handles floating-point noise in precipitation flux)
        negative_mask = tot_prec_acc_np < 0.0
        tot_prec_acc_np[negative_mask] = 0.0

        # Apply floor threshold
        floor_mask_arr = tot_prec_acc_np <= floor
        if np.any(floor_mask_arr):
            tot_prec_acc_np[floor_mask_arr] = floor_value

        # Store temperature where precipitation is not masked
        if store_temperature:
            tas_prec_np = to_numpy(tas_prec)
            tas_prec_np[:] = to_numpy(tas_var)[:]
            tas_prec_np[np.isnan(tot_prec_acc_np)] = np.nan


@comin.register_callback(comin.EP_DESTRUCTOR)
def precipitation_destructor():
    if 'tot_prec_acc' in globals() and tot_prec_acc is not None:
        logger.finished()
