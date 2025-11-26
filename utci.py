"""
Test Plugin for computing the UTCI during model runtime.

NOTE: Works for AES physics only.
"""

import comin
import argparse
import numpy as np
from numba import jit

SIGMA_SB = 5.67e-8

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

parser = argparser.ArgumentParser()

parser.add_argument("--interval", type=int, default=1,
                    help="Specify the desired time interval to compute the UTCI in hours")

args = parser.parse_args(comin.current_get_plugin_info().args)

jg = 1
domain = comin.descrdata_get_comain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)

# register the variable
vd_utci = ("utci", jg)
comin.var_request_add(vd_utci,
                      hgrid_id=1
                      zaxis_id=comin.COMIN_ZAXIS_2D,
                      standard_name="UTCI",
                      long_name="Universal Thermal Climate Index",
                      units="C")

def compute_mrt(rlds, rlus, rsds, rsus, rsds_diff, fp, i_star):
    mrt = np.power((1/SIGMA_SB) * (
        0.5*rlds + 0.5*rlus + (0.7/0.97)*(0.5*rsds_diff + 0.5*rsus + fp*i_star)
    ), 0.25
    )
    return mrt

@jit(nopython=True)
def _compute_utci():
    """Polynomial that computes the UTCI"""

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def utci_constructor():
    # access the variables
    global utci, tas, d, cosmu0, rlds, rlus, rsds, rsus
    utci = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("utci", jg), flag=comin.COMIN_FLAG_WRITE)
    tas = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", jg), flag=comin.COMIN_FLAG_READ)
    d = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("d", jg), flag=comin.COMIN_FLAG_READ)
    cosmu0 = comin.var_get([comin.ET_ATM_WRITE_OUTPUT_BEFORE], ("cosmu0", jg), flag=comin.COMIN_FLAG_READ)
    rlds = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rlds", jg), flag=comin.COMIN_FLAG_READ) 
    rlus = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rlus", jg), flag=comin.COMIN_FLAG_READ)
    rsds = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rsds", jg), flag=comin.COMIN_FLAG_READ)
    rsus = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rsus", jg), flag=comin.COMIN_FLAG_READ)

@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def compute_utci():
    mask_2d = (decomp_domain_np != 0)
    cosmu0_np = np.ma.masked_array(np.squeeze(cosmu0), mask=mask_2d)
    rlds_np = np.ma.masked_array(np.squeeze(rlds), mask=mask_2d)
    rlus_np = np.ma.masked_array(np.squeeze(rlus), mask=mask_2d)
    rsds_np = np.ma.masked_array(np.squeeze(rsds), mask=mask_2d)
    rsus_np = np.ma.masked_array(np.squeeze(rsus), mask=mask_2d)
    tas_np = np.ma.masked_array(np.squeeze(tas), masl=mask_2d)
    d_np = np.ma.masked_array(np.squeeze(d), mask=mask_2d)

    gamma = np.arcsin(cosmu0_np)
    fp = 0.308 * np.cos(0.988*gamma - gamma**2/50000)

    s_star = rsds*((1367*cosmu0_np * d**-2)**-1)
    s_star = np.where(s_star > 0.85, 0.85, s_star)
    fdir_ratio = np.exp(3 - 1.34*s_star - 1.65 * (s_star**-1))
    fdir_ratio = np.where(fdir_ratio > 0.9, 0.9, fdir_ratio)

    rsds_dir = fdir_ratio * rsds
    rsds_diff = rsds_np - rsds_dir

    i_star = np.where(cosmu0 > 0.001, rsds_direct/cosmu0, 0)

    mrt = compute_mrt(rlds_np, rlus_np, rsds_np, rsus_np, rsds_diff, fp, i_star)

    utci = _compute_utci(tas_np, mrt)
