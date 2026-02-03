# Notes on the `utci.py` plugin

This plugin is intended to be used when the AES module of ICON is in use.
It won't work with NWP.

Call as e.g.
```bash
/path/to/utci.py
```

which by default applies a land masking, leaving out ocean cells in the dataset.

Possible argument:

- `--no_land_mask`: Parsed if the one wants to keep the UTCI on land cells. 

The usage of `--no_land_mask` is discouraged unless strictly neccesary.
Keeping it saves more than half of disk space: the dataset goes up to 3.7M from
1.5M if the argument is parsed in a test done for on the R2B4 grid.

---

The Universal Thermal Climate Index (UTCI) measures thermal comfort experienced
by humans. It depends on the temperature, humidity, wind speed, and mean
radiant temperature. It requires an expensive computation as the index itself
consists of a 4th-order polynomial of those quantities with more than 200 terms.

The implementation of the plugin **requires** to include the ICON variable `hur`
(a 3-dimensional array containing the relative humidity) in the namelists. This
can be done as a 2-dimensional array at the lowest level as

```fortran
&output_nml
 output_filename  = "${EXPNAME}_atm_2d_surf"
 ml_varlist       = 'hur'
 m_levels         = '1'
 .
 .
 .
/
```

Or in hiopy by passing the attributes

```python
"var_list": ['hur'],
"name_of_level": "surface_level",
"positive_direction": "up",
"is_pressure_level": False,
"levels": [0],
"collection_selection": [0]
```

A masking puts `NaN` in locations where the UTCI is outside of its validity
range.

## Tests

The plugin implements numba's just-in-time (JIT) compiler for the function that
calculates the UTCI. Here are some NodeHours numbers of the `mpmd.conf` for a
simulation using the R2B4 grid ran for 5d6h on 4 nodes in Levante:


| numba | time 1 | time 2 | time 3 | time 4 | time 5 | average |
| ----- | ------ | ------ | ------ | ------ | ------ | ------- |
| No    | 0.35   | 0.34   | 0.36   | 0.36   | 0.36   | 0.353   |
| Yes   | 0.34   | 0.36   | 0.35   | 0.34   | 0.34   | 0.346   |

