"""
Test Plugin for computing the UTCI during model runtime.

NOTE: Works for AES physics only.
"""

import sys
import comin
import argparse
import numpy as np
from numba import jit
from mpi4py import MPI
from datetime import datetime

SIGMA_SB = 5.67e-8
_first_call_done = False

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

parser = argparse.ArgumentParser()

parser.add_argument("--interval", type=int, default=1,
                    help="Specify the desired time interval to compute the UTCI in hours")

args = parser.parse_args(comin.current_get_plugin_info().args)

jg = 1
domain = comin.descrdata_get_domain(jg)
decomp_domain_np = np.asarray(domain.cells.decomp_domain)

# register the variables
vd_utci = ("utci", jg)
comin.var_request_add(vd_utci, lmodexclusive=False)
comin.metadata_set(vd_utci,
                      hgrid_id=1,
                      zaxis_id=comin.COMIN_ZAXIS_2D,
                      standard_name="UTCI",
                      long_name="Universal Thermal Climate Index",
                      units="C")

vd_mrt = ("mrt", jg)
comin.var_request_add(vd_mrt, lmodexclusive=False)
comin.metadata_set(vd_mrt,
                      hgrid_id=1,
                      zaxis_id=comin.COMIN_ZAXIS_2D,
                      standard_name="Tmrt",
                      long_name="Mean Radiant Temperature",
                      units="C")

def calc_sat_pres_water(temp_K):
    """
    Calculate saturation vapor pressure over liquid water.

    Parameters:
    -----------
    temp_K : array
        Temperature in Kelvin

    Returns:
    --------
    array
        Saturation vapor pressure in Pa
    """
    # Constants from ICON (mo_aes_thermo.f90)
    c1es = 610.78      # Pa
    c3les = 17.269     # dimensionless
    c4les = 35.86      # K
    tmelt = 273.15     # K

    return c1es * np.exp(c3les * (temp_K - tmelt) / (temp_K - c4les))

def calc_sat_pres_ice(temp_K):
    """
    Calculate saturation vapor pressure over ice.

    Parameters:
    -----------
    temp_K : array
        Temperature in Kelvin

    Returns:
    --------
    array
        Saturation vapor pressure in Pa
    """
    # Constants from ICON (mo_aes_thermo.f90)
    c1es = 610.78      # Pa
    c3ies = 21.875     # dimensionless
    c4ies = 7.66       # K
    tmelt = 273.15     # K

    return c1es * np.exp(c3ies * (temp_K - tmelt) / (temp_K - c4ies))

def calc_sat_pres_mixed(temp_K):
    """Calculate saturation vapor pressure with mixed phase."""
    tmelt = 273.15

    esat_water = calc_sat_pres_water(temp_K)
    esat_ice = calc_sat_pres_ice(temp_K)

    # Simple linear blend (ICON uses more sophisticated schemes)
    # You can adjust this based on your needs
    esat = np.where(temp_K >= tmelt, esat_water, esat_ice)

    return esat

def compute_mrt(rlds, rlus, rsds, rsus, rsds_diff, fp, i_star):
    _mrt = np.power((1/SIGMA_SB) * (
        0.5*rlds + 0.5*rlus + (0.7/0.97)*(0.5*rsds_diff + 0.5*rsus + fp*i_star)
    ), 0.25
    )
    return _mrt

@jit(nopython=True)
def _compute_utci(t_2m, sfcwind, _mrt, wvp):
    """
    Polynomial that computes the UTCI.

    Taken from xclim's utci function, which in turn is taken from
    http://www.utci.org/public/UTCI%20Program%20Code/UTCI_a002.f90
    by Peter Bröde.
    
    
    Parameters:
    -----------
    t_2m : array
        Temperature in Celsius
    sfcwind : array
        Surface wind speed in m/s
    _mrt : array
        Mean radian temperature
    wvp : array
        Water vapor pressure in kPa

    Returns:
    --------
    array
        UTCI in Celsius
    """
    dt = _mrt - t_2m  # temperature delta
    utci = (t_2m
            + 6.07562052e-1
            + -2.27712343e-2 * t_2m
            + 8.06470249e-4 * t_2m * t_2m
            + -1.54271372e-4 * t_2m * t_2m * t_2m
            + -3.24651735e-6 * t_2m * t_2m * t_2m * t_2m
            + 7.32602852e-8 * t_2m * t_2m * t_2m * t_2m * t_2m
            + 1.35959073e-9 * t_2m * t_2m * t_2m * t_2m * t_2m * t_2m
            + -2.25836520e0 * sfcwind
            + 8.80326035e-2 * t_2m * sfcwind
            + 2.16844454e-3 * t_2m * t_2m * sfcwind
            + -1.53347087e-5 * t_2m * t_2m * t_2m * sfcwind
            + -5.72983704e-7 * t_2m * t_2m * t_2m * t_2m * sfcwind
            + -2.55090145e-9 * t_2m * t_2m * t_2m * t_2m * t_2m * sfcwind
            + -7.51269505e-1 * sfcwind * sfcwind
            + -4.08350271e-3 * t_2m * sfcwind * sfcwind
            + -5.21670675e-5 * t_2m * t_2m * sfcwind * sfcwind
            + 1.94544667e-6 * t_2m * t_2m * t_2m * sfcwind * sfcwind
            + 1.14099531e-8 * t_2m * t_2m * t_2m * t_2m * sfcwind * sfcwind
            + 1.58137256e-1 * sfcwind * sfcwind * sfcwind
            + -6.57263143e-5 * t_2m * sfcwind * sfcwind * sfcwind
            + 2.22697524e-7 * t_2m * t_2m * sfcwind * sfcwind * sfcwind
            + -4.16117031e-8 * t_2m * t_2m * t_2m * sfcwind * sfcwind * sfcwind
            + -1.27762753e-2 * sfcwind * sfcwind * sfcwind * sfcwind
            + 9.66891875e-6 * t_2m * sfcwind * sfcwind * sfcwind * sfcwind
            + 2.52785852e-9 * t_2m * t_2m * sfcwind * sfcwind * sfcwind * sfcwind
            + 4.56306672e-4 * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind
            + -1.74202546e-7 * t_2m * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind
            + -5.91491269e-6 * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind
            + 3.98374029e-1 * dt
            + 1.83945314e-4 * t_2m * dt
            + -1.73754510e-4 * t_2m * t_2m * dt
            + -7.60781159e-7 * t_2m * t_2m * t_2m * dt
            + 3.77830287e-8 * t_2m * t_2m * t_2m * t_2m * dt
            + 5.43079673e-10 * t_2m * t_2m * t_2m * t_2m * t_2m * dt
            + -2.00518269e-2 * sfcwind * dt
            + 8.92859837e-4 * t_2m * sfcwind * dt
            + 3.45433048e-6 * t_2m * t_2m * sfcwind * dt
            + -3.77925774e-7 * t_2m * t_2m * t_2m * sfcwind * dt
            + -1.69699377e-9 * t_2m * t_2m * t_2m * t_2m * sfcwind * dt
            + 1.69992415e-4 * sfcwind * sfcwind * dt
            + -4.99204314e-5 * t_2m * sfcwind * sfcwind * dt
            + 2.47417178e-7 * t_2m * t_2m * sfcwind * sfcwind * dt
            + 1.07596466e-8 * t_2m * t_2m * t_2m * sfcwind * sfcwind * dt
            + 8.49242932e-5 * sfcwind * sfcwind * sfcwind * dt
            + 1.35191328e-6 * t_2m * sfcwind * sfcwind * sfcwind * dt
            + -6.21531254e-9 * t_2m * t_2m * sfcwind * sfcwind * sfcwind * dt
            + -4.99410301e-6 * sfcwind * sfcwind * sfcwind * sfcwind * dt
            + -1.89489258e-8 * t_2m * sfcwind * sfcwind * sfcwind * sfcwind * dt
            + 8.15300114e-8 * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind * dt
            + 7.55043090e-4 * dt * dt
            + -5.65095215e-5 * t_2m * dt * dt
            + -4.52166564e-7 * t_2m * t_2m * dt * dt
            + 2.46688878e-8 * t_2m * t_2m * t_2m * dt * dt
            + 2.42674348e-10 * t_2m * t_2m * t_2m * t_2m * dt * dt
            + 1.54547250e-4 * sfcwind * dt * dt
            + 5.24110970e-6 * t_2m * sfcwind * dt * dt
            + -8.75874982e-8 * t_2m * t_2m * sfcwind * dt * dt
            + -1.50743064e-9 * t_2m * t_2m * t_2m * sfcwind * dt * dt
            + -1.56236307e-5 * sfcwind * sfcwind * dt * dt
            + -1.33895614e-7 * t_2m * sfcwind * sfcwind * dt * dt
            + 2.49709824e-9 * t_2m * t_2m * sfcwind * sfcwind * dt * dt
            + 6.51711721e-7 * sfcwind * sfcwind * sfcwind * dt * dt
            + 1.94960053e-9 * t_2m * sfcwind * sfcwind * sfcwind * dt * dt
            + -1.00361113e-8 * sfcwind * sfcwind * sfcwind * sfcwind * dt * dt
            + -1.21206673e-5 * dt * dt * dt
            + -2.18203660e-7 * t_2m * dt * dt * dt
            + 7.51269482e-9 * t_2m * t_2m * dt * dt * dt
            + 9.79063848e-11 * t_2m * t_2m * t_2m * dt * dt * dt
            + 1.25006734e-6 * sfcwind * dt * dt * dt
            + -1.81584736e-9 * t_2m * sfcwind * dt * dt * dt
            + -3.52197671e-10 * t_2m * t_2m * sfcwind * dt * dt * dt
            + -3.36514630e-8 * sfcwind * sfcwind * dt * dt * dt
            + 1.35908359e-10 * t_2m * sfcwind * sfcwind * dt * dt * dt
            + 4.17032620e-10 * sfcwind * sfcwind * sfcwind * dt * dt * dt
            + -1.30369025e-9 * dt * dt * dt * dt
            + 4.13908461e-10 * t_2m * dt * dt * dt * dt
            + 9.22652254e-12 * t_2m * t_2m * dt * dt * dt * dt
            + -5.08220384e-9 * sfcwind * dt * dt * dt * dt
            + -2.24730961e-11 * t_2m * sfcwind * dt * dt * dt * dt
            + 1.17139133e-10 * sfcwind * sfcwind * dt * dt * dt * dt
            + 6.62154879e-10 * dt * dt * dt * dt * dt
            + 4.03863260e-13 * t_2m * dt * dt * dt * dt * dt
            + 1.95087203e-12 * sfcwind * dt * dt * dt * dt * dt
            + -4.73602469e-12 * dt * dt * dt * dt * dt * dt
            + 5.12733497e0 * wvp
            + -3.12788561e-1 * t_2m * wvp
            + -1.96701861e-2 * t_2m * t_2m * wvp
            + 9.99690870e-4 * t_2m * t_2m * t_2m * wvp
            + 9.51738512e-6 * t_2m * t_2m * t_2m * t_2m * wvp
            + -4.66426341e-7 * t_2m * t_2m * t_2m * t_2m * t_2m * wvp
            + 5.48050612e-1 * sfcwind * wvp
            + -3.30552823e-3 * t_2m * sfcwind * wvp
            + -1.64119440e-3 * t_2m * t_2m * sfcwind * wvp
            + -5.16670694e-6 * t_2m * t_2m * t_2m * sfcwind * wvp
            + 9.52692432e-7 * t_2m * t_2m * t_2m * t_2m * sfcwind * wvp
            + -4.29223622e-2 * sfcwind * sfcwind * wvp
            + 5.00845667e-3 * t_2m * sfcwind * sfcwind * wvp
            + 1.00601257e-6 * t_2m * t_2m * sfcwind * sfcwind * wvp
            + -1.81748644e-6 * t_2m * t_2m * t_2m * sfcwind * sfcwind * wvp
            + -1.25813502e-3 * sfcwind * sfcwind * sfcwind * wvp
            + -1.79330391e-4 * t_2m * sfcwind * sfcwind * sfcwind * wvp
            + 2.34994441e-6 * t_2m * t_2m * sfcwind * sfcwind * sfcwind * wvp
            + 1.29735808e-4 * sfcwind * sfcwind * sfcwind * sfcwind * wvp
            + 1.29064870e-6 * t_2m * sfcwind * sfcwind * sfcwind * sfcwind * wvp
            + -2.28558686e-6 * sfcwind * sfcwind * sfcwind * sfcwind * sfcwind * wvp
            + -3.69476348e-2 * dt * wvp
            + 1.62325322e-3 * t_2m * dt * wvp
            + -3.14279680e-5 * t_2m * t_2m * dt * wvp
            + 2.59835559e-6 * t_2m * t_2m * t_2m * dt * wvp
            + -4.77136523e-8 * t_2m * t_2m * t_2m * t_2m * dt * wvp
            + 8.64203390e-3 * sfcwind * dt * wvp
            + -6.87405181e-4 * t_2m * sfcwind * dt * wvp
            + -9.13863872e-6 * t_2m * t_2m * sfcwind * dt * wvp
            + 5.15916806e-7 * t_2m * t_2m * t_2m * sfcwind * dt * wvp
            + -3.59217476e-5 * sfcwind * sfcwind * dt * wvp
            + 3.28696511e-5 * t_2m * sfcwind * sfcwind * dt * wvp
            + -7.10542454e-7 * t_2m * t_2m * sfcwind * sfcwind * dt * wvp
            + -1.24382300e-5 * sfcwind * sfcwind * sfcwind * dt * wvp
            + -7.38584400e-9 * t_2m * sfcwind * sfcwind * sfcwind * dt * wvp
            + 2.20609296e-7 * sfcwind * sfcwind * sfcwind * sfcwind * dt * wvp
            + -7.32469180e-4 * dt * dt * wvp
            + -1.87381964e-5 * t_2m * dt * dt * wvp
            + 4.80925239e-6 * t_2m * t_2m * dt * dt * wvp
            + -8.75492040e-8 * t_2m * t_2m * t_2m * dt * dt * wvp
            + 2.77862930e-5 * sfcwind * dt * dt * wvp
            + -5.06004592e-6 * t_2m * sfcwind * dt * dt * wvp
            + 1.14325367e-7 * t_2m * t_2m * sfcwind * dt * dt * wvp
            + 2.53016723e-6 * sfcwind * sfcwind * dt * dt * wvp
            + -1.72857035e-8 * t_2m * sfcwind * sfcwind * dt * dt * wvp
            + -3.95079398e-8 * sfcwind * sfcwind * sfcwind * dt * dt * wvp
            + -3.59413173e-7 * dt * dt * dt * wvp
            + 7.04388046e-7 * t_2m * dt * dt * dt * wvp
            + -1.89309167e-8 * t_2m * t_2m * dt * dt * dt * wvp
            + -4.79768731e-7 * sfcwind * dt * dt * dt * wvp
            + 7.96079978e-9 * t_2m * sfcwind * dt * dt * dt * wvp
            + 1.62897058e-9 * sfcwind * sfcwind * dt * dt * dt * wvp
            + 3.94367674e-8 * dt * dt * dt * dt * wvp
            + -1.18566247e-9 * t_2m * dt * dt * dt * dt * wvp
            + 3.34678041e-10 * sfcwind * dt * dt * dt * dt * wvp
            + -1.15606447e-10 * dt * dt * dt * dt * dt * wvp
            + -2.80626406e0 * wvp * wvp
            + 5.48712484e-1 * t_2m * wvp * wvp
            + -3.99428410e-3 * t_2m * t_2m * wvp * wvp
            + -9.54009191e-4 * t_2m * t_2m * t_2m * wvp * wvp
            + 1.93090978e-5 * t_2m * t_2m * t_2m * t_2m * wvp * wvp
            + -3.08806365e-1 * sfcwind * wvp * wvp
            + 1.16952364e-2 * t_2m * sfcwind * wvp * wvp
            + 4.95271903e-4 * t_2m * t_2m * sfcwind * wvp * wvp
            + -1.90710882e-5 * t_2m * t_2m * t_2m * sfcwind * wvp * wvp
            + 2.10787756e-3 * sfcwind * sfcwind * wvp * wvp
            + -6.98445738e-4 * t_2m * sfcwind * sfcwind * wvp * wvp
            + 2.30109073e-5 * t_2m * t_2m * sfcwind * sfcwind * wvp * wvp
            + 4.17856590e-4 * sfcwind * sfcwind * sfcwind * wvp * wvp
            + -1.27043871e-5 * t_2m * sfcwind * sfcwind * sfcwind * wvp * wvp
            + -3.04620472e-6 * sfcwind * sfcwind * sfcwind * sfcwind * wvp * wvp
            + 5.14507424e-2 * dt * wvp * wvp
            + -4.32510997e-3 * t_2m * dt * wvp * wvp
            + 8.99281156e-5 * t_2m * t_2m * dt * wvp * wvp
            + -7.14663943e-7 * t_2m * t_2m * t_2m * dt * wvp * wvp
            + -2.66016305e-4 * sfcwind * dt * wvp * wvp
            + 2.63789586e-4 * t_2m * sfcwind * dt * wvp * wvp
            + -7.01199003e-6 * t_2m * t_2m * sfcwind * dt * wvp * wvp
            + -1.06823306e-4 * sfcwind * sfcwind * dt * wvp * wvp
            + 3.61341136e-6 * t_2m * sfcwind * sfcwind * dt * wvp * wvp
            + 2.29748967e-7 * sfcwind * sfcwind * sfcwind * dt * wvp * wvp
            + 3.04788893e-4 * dt * dt * wvp * wvp
            + -6.42070836e-5 * t_2m * dt * dt * wvp * wvp
            + 1.16257971e-6 * t_2m * t_2m * dt * dt * wvp * wvp
            + 7.68023384e-6 * sfcwind * dt * dt * wvp * wvp
            + -5.47446896e-7 * t_2m * sfcwind * dt * dt * wvp * wvp
            + -3.59937910e-8 * sfcwind * sfcwind * dt * dt * wvp * wvp
            + -4.36497725e-6 * dt * dt * dt * wvp * wvp
            + 1.68737969e-7 * t_2m * dt * dt * dt * wvp * wvp
            + 2.67489271e-8 * sfcwind * dt * dt * dt * wvp * wvp
            + 3.23926897e-9 * dt * dt * dt * dt * wvp * wvp
            + -3.53874123e-2 * wvp * wvp * wvp
            + -2.21201190e-1 * t_2m * wvp * wvp * wvp
            + 1.55126038e-2 * t_2m * t_2m * wvp * wvp * wvp
            + -2.63917279e-4 * t_2m * t_2m * t_2m * wvp * wvp * wvp
            + 4.53433455e-2 * sfcwind * wvp * wvp * wvp
            + -4.32943862e-3 * t_2m * sfcwind * wvp * wvp * wvp
            + 1.45389826e-4 * t_2m * t_2m * sfcwind * wvp * wvp * wvp
            + 2.17508610e-4 * sfcwind * sfcwind * wvp * wvp * wvp
            + -6.66724702e-5 * t_2m * sfcwind * sfcwind * wvp * wvp * wvp
            + 3.33217140e-5 * sfcwind * sfcwind * sfcwind * wvp * wvp * wvp
            + -2.26921615e-3 * dt * wvp * wvp * wvp
            + 3.80261982e-4 * t_2m * dt * wvp * wvp * wvp
            + -5.45314314e-9 * t_2m * t_2m * dt * wvp * wvp * wvp
            + -7.96355448e-4 * sfcwind * dt * wvp * wvp * wvp
            + 2.53458034e-5 * t_2m * sfcwind * dt * wvp * wvp * wvp
            + -6.31223658e-6 * sfcwind * sfcwind * dt * wvp * wvp * wvp
            + 3.02122035e-4 * dt * dt * wvp * wvp * wvp
            + -4.77403547e-6 * t_2m * dt * dt * wvp * wvp * wvp
            + 1.73825715e-6 * sfcwind * dt * dt * wvp * wvp * wvp
            + -4.09087898e-7 * dt * dt * dt * wvp * wvp * wvp
            + 6.14155345e-1 * wvp * wvp * wvp * wvp
            + -6.16755931e-2 * t_2m * wvp * wvp * wvp * wvp
            + 1.33374846e-3 * t_2m * t_2m * wvp * wvp * wvp * wvp
            + 3.55375387e-3 * sfcwind * wvp * wvp * wvp * wvp
            + -5.13027851e-4 * t_2m * sfcwind * wvp * wvp * wvp * wvp
            + 1.02449757e-4 * sfcwind * sfcwind * wvp * wvp * wvp * wvp
            + -1.48526421e-3 * dt * wvp * wvp * wvp * wvp
            + -4.11469183e-5 * t_2m * dt * wvp * wvp * wvp * wvp
            + -6.80434415e-6 * sfcwind * dt * wvp * wvp * wvp * wvp
            + -9.77675906e-6 * dt * dt * wvp * wvp * wvp * wvp
            + 8.82773108e-2 * wvp * wvp * wvp * wvp * wvp
            + -3.01859306e-3 * t_2m * wvp * wvp * wvp * wvp * wvp
            + 1.04452989e-3 * sfcwind * wvp * wvp * wvp * wvp * wvp
            + 2.47090539e-4 * dt * wvp * wvp * wvp * wvp * wvp
            + 1.48348065e-3 * wvp * wvp * wvp * wvp * wvp * wvp)

    return utci

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def utci_constructor():
    # access the variables
    global utci, tas, d, cosmu0, mrt, rlds, rlus, rsds, rsus, sfcwind, hur, rsdt, daylght_frc, _first_call_done
    utci = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("utci", jg), flag=comin.COMIN_FLAG_WRITE)
    tas = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", jg), flag=comin.COMIN_FLAG_READ)
    cosmu0 = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("cosmu0", jg), flag=comin.COMIN_FLAG_READ)
    rlds = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rlds", jg), flag=comin.COMIN_FLAG_READ) 
    rlus = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rlus", jg), flag=comin.COMIN_FLAG_READ)
    rsds = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rsds", jg), flag=comin.COMIN_FLAG_READ)
    rsus = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rsus", jg), flag=comin.COMIN_FLAG_READ)
    sfcwind = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("sfcwind", jg), flag=comin.COMIN_FLAG_READ)
    hur = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("hur", jg), flag=comin.COMIN_FLAG_READ)
    #esat = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("esat", jg), flag=comin.COMIN_FLAG_READ)  # in Pa
    rsdt = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("rsdt", jg), flag=comin.COMIN_FLAG_READ)
    daylght_frc = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("daylght_frc", jg), flag=comin.COMIN_FLAG_READ)
    mrt = comin.var_get([comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("mrt", jg), flag=comin.COMIN_FLAG_WRITE)
    _first_call_done = False

@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def compute_utci():
    global _first_call_done
    if not _first_call_done:
        _first_call_done = True
        if rank==0:
            print(f'FHN: first call detected. Setting _first_call_done to True', file=sys.stderr)
        return
    mask_2d = (decomp_domain_np != 0)
    cosmu0_np = np.ma.masked_array(np.squeeze(cosmu0), mask=mask_2d)
    rlds_np = np.ma.masked_array(np.squeeze(rlds), mask=mask_2d)
    rlus_np = np.ma.masked_array(np.squeeze(rlus), mask=mask_2d)
    rsds_np = np.ma.masked_array(np.squeeze(rsds), mask=mask_2d)
    rsus_np = np.ma.masked_array(np.squeeze(rsus), mask=mask_2d)
    tas_np = np.ma.masked_array(np.squeeze(tas), mask=mask_2d)
    hur_np = np.asarray(hur)  # hur is a 3D array; get the value closest to the surface
    hur_np = hur_np[:, 0, :]
    hur_np = np.ma.masked_array(np.squeeze(hur_np), mask=mask_2d)
    #esat_np = np.ma.masked_array(np.squeeze(esat), mask=mask_2d)
    sfcwind_np = np.ma.masked_array(np.squeeze(sfcwind), mask=mask_2d)
    rsdt_np = np.ma.masked_array(np.squeeze(rsdt), mask=mask_2d)
    daylght_frc_np = np.ma.masked_array(np.squeeze(daylght_frc), mask=mask_2d)
    utci_np = np.squeeze(np.asarray(utci))
    mrt_np = np.squeeze(np.asarray(mrt))

    esat = calc_sat_pres_mixed(tas_np)
    wvp = esat * (hur_np/100)  # water vapor pressure in Pa; convert hur from 0-100 to 0-1
    wvp = wvp / 1000  # in kPa

    gamma = np.arcsin(cosmu0_np)
    fp = 0.308 * np.cos(0.988*gamma - gamma**2/50000)

    # in ICON, psctm(jg) = tsi/dist_sun**2 * fsolrad, but psctm cannot be
    # accessed via ComIn. Solution: work with rsdt0 (= psctm)
    # rsdt(jc) = rsdt0 * cosmu0(jc) * daylght_frc(jc)
    # factor = rsdt0 * cosmu0 = rsdt/daylght_frc
    factor = rsdt_np / daylght_frc_np
    s_star = rsds_np*(factor**-1)
    s_star = np.where(s_star > 0.85, 0.85, s_star)
    fdir_ratio = np.exp(3 - 1.34*s_star - 1.65 * (s_star**-1))
    fdir_ratio = np.where(fdir_ratio > 0.9, 0.9, fdir_ratio)

    rsds_dir = fdir_ratio * rsds_np
    rsds_diff = rsds_np - rsds_dir

    i_star = np.where(cosmu0_np > 0.001, rsds_dir/cosmu0_np, 0)

    mrt_np[:] = compute_mrt(rlds_np, rlus_np, rsds_np, rsus_np, rsds_diff, fp, i_star)
    mrt_np[:] = mrt_np[:] - 273.15  # in Celsius

    tas_np = tas_np - 273.15  # convert to Celsius
    tas_data = np.ma.getdata(tas_np)
    sfcwind_data = np.ma.getdata(sfcwind_np)
    mrt_data = np.ma.getdata(mrt_np[:])
    wvp_data = np.ma.getdata(wvp)

    utci_np[:] = _compute_utci(tas_data, sfcwind_data, mrt_data, wvp_data)
