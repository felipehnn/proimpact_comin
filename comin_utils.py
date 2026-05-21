"""
ComIn Plugin Utilities - Common boilerplate for ComIn plugins.

This module provides utilities to reduce repetitive code in ComIn plugins:
- PluginContext: MPI, domain, and decomposition setup
- PluginLogger: Rank-aware logging
- register_variable: Simplified variable registration
- LandMask: Land masking helper
- Array utilities: to_numpy, to_masked

Author: Felipe Navarrete (GERICS-Hereon; felipe.navarrete@hereon.de)
"""

import sys
from types import SimpleNamespace
import numpy as np
import comin
from mpi4py import MPI


class PluginContext:
    """
    Holds common plugin context: MPI communicator, domain info, and masks.

    Attributes
    ----------
    jg : int
        Domain grid ID (default: 1)
    comm : MPI.Comm
        MPI communicator
    rank : int
        MPI rank of this process
    domain : comin domain object
        Domain descriptor from comin.descrdata_get_domain
    decomp_domain : np.ndarray
        Domain decomposition array
    mask_2d : np.ndarray
        Boolean mask where True = halo cells (non-prognostic)
    nlev : int
        Number of vertical levels
    dt : float
        Physics timestep in seconds

    Example
    -------
    >>> ctx = PluginContext(jg=1)
    >>> print(f"Running on rank {ctx.rank} with {ctx.nlev} levels")
    """

    def __init__(self, jg=1):
        self.jg = jg

        # MPI setup
        self.comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
        self.rank = self.comm.Get_rank()

        # Domain setup
        self.domain = comin.descrdata_get_domain(jg)
        self.decomp_domain = np.asarray(self.domain.cells.decomp_domain)
        self.mask_2d = (self.decomp_domain != 0)
        self.nlev = self.domain.nlev

        # Physics timestep (may not be available at import time, so we lazy-load)
        self._dt = None

    @property
    def dt(self):
        """Physics timestep in seconds (lazy-loaded)."""
        if self._dt is None:
            self._dt = comin.descrdata_get_timesteplength(self.jg)
        return self._dt

    @property
    def clon_deg(self):
        """Cell longitudes in degrees."""
        return np.rad2deg(np.asarray(self.domain.cells.clon))

    @property
    def clat_deg(self):
        """Cell latitudes in degrees."""
        return np.rad2deg(np.asarray(self.domain.cells.clat))


class PluginLogger:
    """
    Rank-aware logger for ComIn plugins.

    Only rank 0 prints messages, avoiding duplicate output in parallel runs.

    Parameters
    ----------
    plugin_name : str
        Name of the plugin (e.g., "utci.py")
    ctx : PluginContext, optional
        If provided, uses ctx.rank. Otherwise creates its own MPI query.

    Example
    -------
    >>> logger = PluginLogger("utci.py", ctx)
    >>> logger.info("Land mask enabled.")
    ComIn - utci.py: Land mask enabled.
    """

    def __init__(self, plugin_name, ctx=None):
        self.plugin_name = plugin_name
        if ctx is not None:
            self.rank = ctx.rank
        else:
            comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
            self.rank = comm.Get_rank()

    def _log(self, message):
        if self.rank == 0:
            print(f"ComIn - {self.plugin_name}: {message}", file=sys.stderr)

    def info(self, message):
        """Log an informational message."""
        self._log(message)

    def warn(self, message):
        """Log a warning message."""
        self._log(f"WARNING - {message}")

    def error(self, message):
        """Log an error message."""
        self._log(f"ERROR - {message}")

    def finished(self):
        """Log plugin finished message (for destructor callbacks)."""
        self._log("Plugin finished.")


def register_variable(name, jg, standard_name=None, long_name=None, units=None,
                      zaxis_id=None, hgrid_id=1):
    """
    Register a ComIn variable with metadata.

    Parameters
    ----------
    name : str
        Variable name
    jg : int
        Domain grid ID
    standard_name : str, optional
        CF standard name
    long_name : str, optional
        Descriptive name
    units : str, optional
        Units string
    zaxis_id : int, optional
        Vertical axis ID (default: COMIN_ZAXIS_2D)
    hgrid_id : int, optional
        Horizontal grid ID (default: 1)

    Returns
    -------
    tuple
        Variable descriptor (name, jg)

    Example
    -------
    >>> vd = register_variable("utci", 1,
    ...                        standard_name="UTCI",
    ...                        long_name="Universal Thermal Climate Index",
    ...                        units="C")
    """
    vd = (name, jg)
    comin.var_request_add(vd, lmodexclusive=False)

    if zaxis_id is None:
        zaxis_id = comin.COMIN_ZAXIS_2D

    metadata_kwargs = {'hgrid_id': hgrid_id, 'zaxis_id': zaxis_id}
    if standard_name is not None:
        metadata_kwargs['standard_name'] = standard_name
    if long_name is not None:
        metadata_kwargs['long_name'] = long_name
    if units is not None:
        metadata_kwargs['units'] = units

    comin.metadata_set(vd, **metadata_kwargs)
    return vd


class LandMask:
    """
    Helper for applying land masks to output arrays.

    Parameters
    ----------
    ctx : PluginContext
        Plugin context
    enabled : bool
        Whether land masking is enabled
    logger : PluginLogger, optional
        Logger for status messages

    Example
    -------
    >>> land_mask = LandMask(ctx, enabled=not args.no_land_mask, logger=logger)
    >>> # In constructor callback:
    >>> land_mask.init_sftlf_var(entry_point)
    >>> # In output callback:
    >>> land_mask.apply(output_array)
    """

    def __init__(self, ctx, enabled=True, logger=None):
        self.ctx = ctx
        self.enabled = enabled
        self.logger = logger
        self._sftlf_var = None
        self._mask = None

        if logger:
            if enabled:
                logger.info("Land mask enabled. Ocean cells will be masked out.")
            else:
                logger.info("Land mask disabled. All cells will be included.")

    def init_sftlf_var(self, entry_points):
        """
        Initialize the sftlf variable reference.

        Call this in the constructor callback.

        Parameters
        ----------
        entry_points : list
            List of entry points where sftlf is needed
        """
        if self.enabled:
            self._sftlf_var = comin.var_get(entry_points, ("sftlf", self.ctx.jg),
                                            flag=comin.COMIN_FLAG_READ)

    @property
    def mask(self):
        """Boolean mask where True = land cells."""
        if self._mask is None and self._sftlf_var is not None:
            sftlf_np = np.squeeze(np.asarray(self._sftlf_var))
            self._mask = sftlf_np > 0.0
        return self._mask

    def apply(self, array, fill_value=np.nan):
        """
        Apply land mask to array (in-place).

        Sets non-land cells to fill_value.

        Parameters
        ----------
        array : np.ndarray
            Array to mask
        fill_value : float
            Value for masked cells (default: NaN)
        """
        if self.enabled and self.mask is not None:
            array[~self.mask] = fill_value


class PluginConfig:
    """
    Config loader for ComIn plugins using YAML configuration files.

    The config file is resolved automatically as ``config/<plugin_name>.yaml``
    relative to the directory containing ``comin_utils.py``. No plugin arguments
    are required. The YAML file must follow this structure::

        comin_plugin:
          name: <plugin_name>
          version: 1
          physics: AES
          parameters:
            <key>: <value>

    Parameters in the file override the defaults supplied to load().

    Parameters
    ----------
    plugin_name : str
        Expected plugin name — used to locate the file and validated against
        the file's ``name`` field.
    logger : PluginLogger, optional
        Logger for status messages.

    Example
    -------
    >>> config = PluginConfig("utci", logger=logger)
    >>> args = config.load(defaults={"no_land_mask": False, "interval": 1})
    >>> print(args.interval)
    """

    def __init__(self, plugin_name, logger=None):
        self.plugin_name = plugin_name
        self.logger = logger
        self._parameters = self._read()

    def _read(self):
        import os
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                f"[{self.plugin_name}] pyyaml is required for config file loading. "
                "Install with: pip install pyyaml"
            )

        config_path = os.path.join(
            os.path.dirname(__file__), "config", f"{self.plugin_name}.yaml"
        )
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"[{self.plugin_name}] Config file not found: {config_path}"
            )
        except yaml.YAMLError as exc:
            raise RuntimeError(
                f"[{self.plugin_name}] Failed to parse config file {config_path}: {exc}"
            )

        plugin_section = data.get("comin_plugin", {})

        file_name = plugin_section.get("name")
        if file_name and file_name != self.plugin_name:
            raise RuntimeError(
                f"Config file declares plugin '{file_name}' "
                f"but '{self.plugin_name}' was expected."
            )

        if self.logger:
            self.logger.info(f"Loaded config from {config_path}")

        return plugin_section.get("parameters", {})

    def load(self, defaults=None):
        """
        Return plugin parameters merged with defaults as a SimpleNamespace.

        Values in the config file override the defaults. Keys present only
        in defaults are kept as-is.

        Parameters
        ----------
        defaults : dict, optional
            Default values for parameters not specified in the config file.

        Returns
        -------
        SimpleNamespace
            One attribute per parameter.
        """
        result = dict(defaults or {})
        result.update(self._parameters)
        return SimpleNamespace(**result)


# =============================================================================
# Array Utility Functions
# =============================================================================

def to_numpy(var):
    """
    Convert ComIn variable to squeezed numpy array.

    Parameters
    ----------
    var : comin variable
        Variable from comin.var_get

    Returns
    -------
    np.ndarray
        Squeezed numpy array
    """
    return np.squeeze(np.asarray(var))


def to_masked(var, mask):
    """
    Convert ComIn variable to masked numpy array.

    Parameters
    ----------
    var : comin variable
        Variable from comin.var_get
    mask : np.ndarray
        Boolean mask (True = masked/invalid)

    Returns
    -------
    np.ma.MaskedArray
        Masked array
    """
    return np.ma.masked_array(np.squeeze(var), mask=mask)


