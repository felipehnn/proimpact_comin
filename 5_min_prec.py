"""
Simple ComIn plugin for extracting 5-minutes precipitation datasets.
Actually, this plugin works for any precipitation accumulation interval
different than that set for ICON's io_nml value of precip_interval with
small changes, but it was created for 5min precipitation in mind.

NOTE: Works only for NWP. Other physics packages may require to set the
accumulation algorithm in the plugin itself.

TODO: 1) Consider adding a threshold for the precipitation under which it is
         set to zero, focusing only on extreme precipitation events. Combined
         with Zarr compression, this might save a big chunk of disk space.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import comin
import numpy as np

# domain
jg = 1
domain = comin.descrdata_get_domain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)

# five minute precipitation variable
vd_5min_prec = ("tot_prec_5min", jg)
comin.var_request_add(vd_5min_prec, lmodexclusive=False)
comin.metadata_set(vd_5min_prec, zaxis_id = comin.COMIN_ZAXIS_2D,
                   standard_name = '5_minute_precipitation',
                   long_name = 'Precipitation accumulated over 5 minutes',
                   units = 'kg m-2')

# placeholder for the previous precipitation
vd_prev_prec = ("previous_prec", jg)
comin.var_request_add(vd_prev_prec, lmodexclusive=False)
comin.metadata_set(vd_prev_prec, zaxis_id = comin.COMIN_ZAXIS_2D,
                   standard_name = 'previous precipitation',
                   long_name = 'Placeholder for the previous precipitation accumulation',
                   units = 'kg m-2')

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    # access the variables 
    global tot_prec_5min, tot_prec, previous_prec
    tot_prec_5min = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tot_prec_5min", jg),  flag=comin.COMIN_FLAG_WRITE)
    tot_prec      = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tot_prec", jg) ,  flag=comin.COMIN_FLAG_READ)
    previous_prec = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_AFTER], ("previous_prec", jg) ,  flag=comin.COMIN_FLAG_WRITE)


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_total_prec():
    """
    Get the total precipitation.

    TODO: Adds a zero to places where
          mean (precipitation rate) < threshold
          where the threshold is defined by the user.
    """

    # mask for prognostic cells
    mask_2d = (decomp_domain_np != 0)

    # masked variables
    tot_prec_np = np.ma.masked_array(np.squeeze(tot_prec), mask = mask_2d)
    previous_prec_np = np.ma.masked_array(np.squeeze(previous_prec), mask = mask_2d)
    # 5min precipitation
    tot_prec_5min_np = np.squeeze(tot_prec_5min)
    tot_prec_5min_np[:] = tot_prec_np - previous_prec_np


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_AFTER)
def prev_prec_callback():
    """Store the total precipitation from previous step"""
    # mask for prognostic cells
    mask_2d = (decomp_domain_np != 0)

    tot_prec_np = np.ma.masked_array(np.squeeze(tot_prec), mask = mask_2d)
    previous_prec_np = np.squeeze(previous_prec)
    previous_prec_np[:] = tot_prec_np
