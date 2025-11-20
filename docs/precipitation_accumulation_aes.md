# Notes on the `precipitation_accumulation_aes.py` plugin

This plugin is intended to be exclusively used for the AES module. It won't
work with NWP.

Call as e.g.
```bash
$ /path/to/precipitation_accumulation_aes.py --interval 300 --floor 1.0
```

Set the required value of `--interval` in seconds to record and accumulate
the precipitation.

It is highly encouraged to set a value for `--floor`. This value is in units
of kg/m2 and specifies a lower value below which the precipitation is set to
NaN (by default) or to zero if `--floor_to_zero` is parsed. This is done to
save disk space, specially if `hiopy` in combination with `zarr` is used.

Early tests on the R2B4 grid show no impact on the runtime. For very high
temporal resolutions (i.e. low values of `--interval`), the disk space
requirements can be relatively moderate. For example, in tests run for 2 days
and 6 hours and for a Healpix level of `z=6`:

- `--interval 300 --floor 0.0 --floor_to_zero` uses 53M of disk space,
- `--interval 300 --floor 0.01 --floor_to_zero` uses 8M of disk space,
- `--interval 300 --floor 0.01` uses 7.3M of disk space.
- `--interval 300 --floor 0.1` uses 3.4M of disk space.

By comparison, hourly `pr` datasets requires 5.2M of disk space and `tas`
datasets need 3.8M of space for the same setups. Therefore, 300s (5 minutes)
accumulated precipitation can use less disk space than hourly data when the
focus is on moderate to high precipitation.
