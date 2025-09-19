"""
ComIn plugin for extracting 5-minutes precipitation datasets for AES physics.
AES does not have a built-in algorithm for accumulating precipitation, so we
do it here.

NOTE: Works only for AES.

TODO: 1) Consider adding a threshold for the precipitation under which it is
         set to zero, focusing only on extreme precipitation events. Combined
         with Zarr compression, this might save a big chunk of disk space.

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""
import sys
import comin
import argparse
import numpy as np
from mpi4py import MPI

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=int, default=None,
                    help="Specify the accumulation interval in seconds")
args = parser.parse_args(comin.current_get_plugin_info().args)

if args.interval is None:
    accumulation_interval = 300
    if rank == 0:
        print(f"No precip interval is specified, using default value of 300 seconds.",
              file=sys.stderr)
else:
    accumulation_interval = args.interval
    if rank == 0:
        print(f"ComIn Plugin: precipitation interval set to {accumulation_interval} seconds.",
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
    global tot_prec_acc, pr_var, prec_timer
    tot_prec_acc = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("tot_prec_acc", jg), flag=comin.COMIN_FLAG_WRITE)
    pr_var = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("pr", jg), flag=comin.COMIN_FLAG_READ)
    prec_timer = comin.var_get([comin.EP_ATM_PHYSICS_AFTER], ("prec_timer", jg), flag=comin.COMIN_FLAG_WRITE)


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
    tot_prec_acc_np = np.squeeze(tot_prec_acc)
    
    # Check if we need to reset (where timer >= accumulation_interval)
    reset_mask = timer > accumulation_interval
    if np.any(reset_mask):
        tot_prec_acc_np[reset_mask] = 0.0
        timer[reset_mask] = 0.0

    timer[:] = timer[:] + dt_phy_sec
    prec_increment = pr_flux * dt_phy_sec
    tot_prec_acc_np[:] = tot_prec_acc_np[:] + prec_increment

@comin.register_callback(comin.EP_DESTRUCTOR)
def precipitation_destructor():
    if 'tot_prec_acc' in globals() and tot_prec_acc is not None:
        print(f"Precipitation accumulation ComIn plugin finished.", 
              file=sys.stderr)
