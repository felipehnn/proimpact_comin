"""
ComIn plugin for extracting 5-minutes precipitation datasets for NWP physics.

NWP physics has a built-in algorithm for accumulating precipitation, so the
plugin only needs to compute the difference between successive tot_prec values.

This version includes support for geographic bounding box filtering, allowing
users to specify a lat/lon box to restrict output to a specific region.

NOTE: Works only for NWP. Other physics packages may require to set the
accumulation algorithm in the plugin itself.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import comin
import numpy as np
from datetime import datetime
from comin_utils import (
    PluginContext, PluginLogger, PluginConfig,
    register_variable, LandMask, to_numpy, to_masked
)

# =============================================================================
# Plugin Setup
# =============================================================================

ctx = PluginContext(jg=1)
logger = PluginLogger("5_min_prec", ctx)

EPSILON = 1e-6  # Tolerance for the floor

# Config loading
config = PluginConfig("5_min_prec", logger=logger)
args = config.load(defaults={
    "interval": 300,
    "floor": 1e-5,
    "floor_to_zero": False,
    "no_land_mask": False,
    "no_temperature": False,
    "lon_min": None,
    "lon_max": None,
    "lat_min": None,
    "lat_max": None,
})

# Temperature output
store_temperature = not args.no_temperature
if store_temperature:
    logger.info("Temperature output enabled.")
else:
    logger.info("Temperature output disabled.")

# Accumulation interval
accumulation_interval = args.interval
logger.info(f"Accumulation interval: {accumulation_interval} seconds.")

# Floor configuration
floor = args.floor
logger.info(f"Precipitation floor: {floor} kg/m2.")
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
    comin.finish("5_min_prec", "Incomplete bounding box specification")

use_bounding_box = all(bbox_specified)
if use_bounding_box:
    lon_min, lon_max, lat_min, lat_max = args.lon_min, args.lon_max, args.lat_min, args.lat_max

    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        logger.error("Latitude must be between -90 and 90 degrees.")
        comin.finish("5_min_prec", "Invalid latitude range")
    if lat_min >= lat_max:
        logger.error(f"lat_min ({lat_min}) must be less than lat_max ({lat_max}).")
        comin.finish("5_min_prec", "Invalid latitude range: lat_min >= lat_max")

    crosses_dateline = lon_min > lon_max
    if crosses_dateline:
        logger.info(f"Bounding box enabled (crosses date line): "
                    f"lon=[{lon_min}, 180] U [-180, {lon_max}], lat=[{lat_min}, {lat_max}]")
    else:
        logger.info(f"Bounding box enabled: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")

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

register_variable("tot_prec_5min", ctx.jg,
                  standard_name='5_minute_precipitation',
                  long_name='Precipitation accumulated over 5 minutes',
                  units='kg m-2')

register_variable("previous_prec", ctx.jg,
                  standard_name='previous_precipitation',
                  long_name='Placeholder for the previous precipitation accumulation',
                  units='kg m-2')

if store_temperature:
    register_variable("tas_prec", ctx.jg,
                      standard_name='air_temperature_at_precipitation',
                      long_name='Near-surface air temperature at precipitation locations',
                      units='K')

# =============================================================================
# Callbacks
# =============================================================================

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def prec_constructor():
    global tot_prec_5min, tot_prec, previous_prec, tas_var, tas_prec
    tot_prec_5min = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tot_prec_5min", ctx.jg), flag=comin.COMIN_FLAG_WRITE)
    tot_prec      = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tot_prec", ctx.jg),       flag=comin.COMIN_FLAG_READ)
    previous_prec = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_AFTER],  ("previous_prec", ctx.jg),  flag=comin.COMIN_FLAG_WRITE)

    land_mask.init_sftlf_var([comin.EP_ATM_WRITE_OUTPUT_BEFORE])

    if store_temperature:
        tas_var  = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", ctx.jg), flag=comin.COMIN_FLAG_READ)
        tas_prec = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas_prec", ctx.jg), flag=comin.COMIN_FLAG_WRITE)
    else:
        tas_var = tas_prec = None


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_total_prec():
    """
    Compute interval precipitation as tot_prec - previous_prec.
    Apply masking (bounding box, land mask) and floor threshold before writing output.
    """
    current_time = comin.current_get_datetime()
    current_datetime = datetime.fromisoformat(current_time)
    seconds = current_datetime.minute * 60 + current_datetime.second

    if seconds % accumulation_interval == 0:
        tot_prec_np = to_masked(tot_prec, ctx.mask_2d)
        previous_prec_np = to_masked(previous_prec, ctx.mask_2d)
        tot_prec_5min_np = to_numpy(tot_prec_5min)

        tot_prec_5min_np[:] = tot_prec_np - previous_prec_np

        # Apply bounding box mask first (mask cells outside the box)
        if use_bounding_box and bbox_mask is not None:
            tot_prec_5min_np[~bbox_mask] = np.nan

        # Apply land mask (mask ocean cells)
        land_mask.apply(tot_prec_5min_np)

        # Clamp negative values to zero BEFORE applying floor
        negative_mask = tot_prec_5min_np < 0.0
        tot_prec_5min_np[negative_mask] = 0.0

        # Apply floor threshold
        floor_mask_arr = tot_prec_5min_np <= floor
        if np.any(floor_mask_arr):
            tot_prec_5min_np[floor_mask_arr] = floor_value

        # Store temperature where precipitation is not masked
        if store_temperature:
            tas_prec_np = to_numpy(tas_prec)
            tas_prec_np[:] = to_numpy(tas_var)[:]
            tas_prec_np[np.isnan(tot_prec_5min_np)] = np.nan


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_AFTER)
def prev_prec_callback():
    """Store the total precipitation from previous step."""
    current_time = comin.current_get_datetime()
    current_datetime = datetime.fromisoformat(current_time)
    seconds = current_datetime.minute * 60 + current_datetime.second

    if seconds % accumulation_interval == 0:
        tot_prec_np = to_masked(tot_prec, ctx.mask_2d)
        previous_prec_np = to_numpy(previous_prec)
        previous_prec_np[:] = tot_prec_np


@comin.register_callback(comin.EP_DESTRUCTOR)
def prec_destructor():
    if 'tot_prec_5min' in globals() and tot_prec_5min is not None:
        logger.finished()