# Notes on the `wind_3d.py` plugin

This plugin is intended to be exclusively used for the AES module. It won't
work with NWP.

Call as e.g.
```bash
/path/to/wind_3d.py
```

The parameters of this plugin are found inside of it. As of now, many things
are hardcoded and will be moved. Still, two constants can be changes directly
inside the plugin:

- `cut_in <float>` which is the cut-in value for the wind speed in m/s. I.e., values below this number will be set to NaN.
- `avg_interval` is the interval in seconds over which the wind speed is accumulated and then averaged.
