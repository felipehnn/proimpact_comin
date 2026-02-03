# Notes on the `precipitation_accumulation_aes.py` plugin

This plugin is intended to be exclusively used for the AES module. It won't
work with NWP.

Call as e.g.
```bash
/path/to/precipitation_accumulation_aes.py --interval 300 --floor 1.0
```

Possible arguments are:
1. `--interval <int>`: Set the accumulation interval in seconds (default is 300).
2. `--floor <float>`: Set a floor in kg/m2 at or below which the accumulated precipitation is set to NaN.
3. `--floor_to_zero`: Parsed adding zeros instead of NaNs (not recommended to use because NaNs save more disk space).
4. `--no_land_mask`: Parsed if the accumulated precipitation on ocean cells is stored; otherwised they are masked out and replaced with NaNs.
5. `--lon_min <float> --lon_max <float> --lat_min <float> --lat_max <float>`: Define a coordinate box (in degrees; [-180, 180] for lon and [-90, 90] for lat). Values that fall inside this box are stored and the rest is replaced with NaN.

It is highly recommended to set a value for `--floor` (1e-5 by default). This
value is in units of kg/m2 and specifies a lower value below which the
precipitation is set to NaN (by default) or to zero if `--floor_to_zero` is
parsed. This is done to save disk space, specially if Zarr is used.

Add `--no-land-mask` precipitation over ocean cells is desired.

---

Early tests on the R2B4 grid show no impact on the runtime. For very high
temporal resolutions (i.e. low values of `--interval`), the disk space
requirements can be relatively moderate. For example, in tests run for 2 days
and 6 hours and for a Healpix level of `z=6`:

- `--interval 300 --floor 0.0 --no_land_mask --floor_to_zero` uses 53M of disk space,
- `--interval 300 --floor 0.01 --no_land_mask --floor_to_zero` uses 8M of disk space,
- `--interval 300 --floor 0.01 --no_land_mask` uses 7.3M of disk space,
- `--interval 300 --floor 0.1 --no_land_mask` uses 3.4M of disk space.

Naturally, the amount of disk space increases if the value of `floor` is
increased. However, significant disk space can be saved if land masking is
applied and/or if a lan/lot box is also applied.

- `--interval 300 --floor 0.0001 --no_land_mask` uses 17M of disk space,
- `--interval 300 --floor 0.0001 --no_land_mask --lon_min -150 --lon_max 50 --lat_min 30 --lat_max 70` uses 4.5M of disk space,
- `--interval 300 --floor 0.0001 --lon_min -150 --lon_max 50 --lat_min 30 --lat_max 70` uses 3.6M of disk space.

By comparison, hourly `pr` datasets requires 5.2M of disk space and `tas`
datasets need 3.8M of space for the same setups. Therefore, 300s (5 minutes)
accumulated precipitation can use less disk space than hourly data when the
focus is on moderate to high precipitation.

Even when the floor is 1e-4 kg/m2, which is virtually zero, the reduction in
disk space is considerable when one focuses on a subset of the entire domain
and leaves out ocean cells.
