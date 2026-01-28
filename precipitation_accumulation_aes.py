"""
ComIn plugin for extracting 5-minutes precipitation datasets for AES physics.
AES does not have a built-in algorithm for accumulating precipitation, so we
do it here.

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

parser = argparse.ArgumentParser()

parser.add_argument("--interval", type=int, default=None,
                    help="Specify the accumulation interval in seconds.")
parser.add_argument("--floor", type=float, default=None,
                    help="Set a floor in kg/m2 below which the accumulated precipitation is set to zero.")
parser.add_argument("--floor_to_zero", action="store_true", default=False,
                    help="Set the values below the floor to zero. Default: set to NaN.")
parser.add_argument("--no_land_mask", action="store_true", default=False,
                    help="Disable land masking. By default, ocean cells are masked out before output.")

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
    floor = 0.0
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: No floor is specified. Using default value of 0.0 kg/m2 (no floor).",
              file=sys.stderr)
else:
    floor = args.floor
    if rank == 0:
        print(f"ComIn - precipitation_accumulation_aes.py: Setting precipitation floor to {floor} kg/m2.",
              file=sys.stderr)

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

# domain
jg = 1
domain = comin.descrdata_get_domain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)

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
    reset_mask = timer > accumulation_interval
    if np.any(reset_mask):
        tot_prec_acc_np[reset_mask] = 0.0
        timer[reset_mask] = 0.0

    timer[:] = timer[:] + dt_phy_sec
    prec_increment = pr_flux * dt_phy_sec
    tot_prec_acc_np[:] = tot_prec_acc_np[:] + prec_increment

@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def precipitation_floor():
    """
    Apply the precipitation floor before writing the output
    """
    current_time = comin.current_get_datetime()
    current_datetime = datetime.fromisoformat(current_time)
    seconds = current_datetime.minute*60 + current_datetime.second

    # Are we writing output now?
    if seconds % accumulation_interval == 0:
        mask_2d = (decomp_domain_np != 0)
        tot_prec_acc_np = np.squeeze(np.asarray(tot_prec_acc))
        if use_land_mask:
            sftlf_np = np.squeeze(np.asarray(sftlf_var))
            land_mask = sftlf_np > 0.0
            tot_prec_acc_np[~land_mask] = np.nan
        floor_mask = tot_prec_acc_np < floor
        if np.any(floor_mask):
            tot_prec_acc_np[floor_mask] = floor_value

@comin.register_callback(comin.EP_DESTRUCTOR)
def precipitation_destructor():
    if 'tot_prec_acc' in globals() and tot_prec_acc is not None:
        if rank == 0:
            print(f"ComIn - precipitation_accumulation_aes.py: Plugin finished.", 
                  file=sys.stderr)
