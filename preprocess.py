"""preprocessing script

Example to run on command line: 
```bash
python preprocess.py \
    --wrf-run /afs/crc.nd.edu/user/r/rdimitro/Public/FATIMA/ \
    --case-name MYNN_NSSL17_78-107lev \
    --file-prefix wrfout_d02 \
    --proc-dir ./data \
    --levs "np.arange(100, 5000, 100)" \
    --interp_var geopotential_height
```
"""

import argparse
import ast
from typing import Union
from pathlib import Path
import logging
import pyproj

import numpy as np
import pandas as pd
import xarray as xr
import xwrf, pint_xarray

import metpy.calc as mcalc
from pint import Quantity
import xgcm
import pdb
from dask.diagnostics.progress import ProgressBar
from dask import config as dask_config

dask_config.set(scheduler="threads")


def parse_arange_expr(expr: str) -> np.ndarray:
    """Safely parse np.arange(start, stop, step) expression"""
    try:
        tree = ast.parse(expr, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Not a function call")
        if not isinstance(tree.body.func, ast.Attribute):
            raise ValueError("Not np.arange call")
        if tree.body.func.attr != "arange":
            raise ValueError("Not arange function")

        args = []
        for arg in tree.body.args:
            if isinstance(arg, ast.Num):
                args.append(arg.n)
            else:
                raise ValueError("Non-numeric argument")

        if len(args) != 3:
            raise ValueError("arange needs 3 arguments")

        return np.arange(*args)
    except Exception as e:
        raise ValueError(f"Invalid arange expression: {e}")


def read_wrfout(paths: list[Path], t_chunk=2) -> xr.Dataset:
    """Read raw wrfout (output directly from wrf) and return as one file"""
    logging.info(f"Reading wrfout files: {paths}")
    ds_raw = xr.open_mfdataset(
        paths, combine="nested", concat_dim="Time", chunks={"Time": t_chunk}
    )
    ds_raw = ds_raw.xwrf.postprocess().xwrf.destagger()
    ds_raw = ds_raw.sortby("Time")
    return ds_raw


def _to_levs(
    ds: xr.Dataset, levs: Union[list, np.ndarray], lev_type="geopotential_height"
) -> xr.Dataset:
    """Project an input xr.Dataset from original wrf vertical coordinate to predefined altitude levels"""

    logging.info(f"Project xr.Dataset to altitude levels {levs}")
    dims_2d = ds["T2"].dims
    dims_3d = ds["wind_east"].dims
    vars_2d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_2d)]
    vars_3d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_3d)]

    grid = xgcm.Grid(ds, periodic=False)

    v3d_dict = {}
    for v3d in vars_3d:
        v3d_dict[v3d] = grid.transform(
            ds[v3d].metpy.dequantify(),
            "Z",
            levs,
            target_data=ds[lev_type],
            method="linear",
        )

    v2d_dict = {}
    for v2d in vars_2d:
        v2d_dict[v2d] = ds[v2d].metpy.dequantify()
    v3d_dict.update(v2d_dict)

    ds_p = xr.Dataset(v3d_dict)
    ds_p["wrf_projection"] = ds.wrf_projection
    ds_p.attrs = ds.attrs

    return ds_p


def to_hlevs(ds: xr.Dataset, levs: Union[list, np.ndarray]) -> xr.Dataset:
    return _to_levs(ds, levs, lev_type="geopotential_height")


def to_plevs(ds: xr.Dataset, levs: Union[list, np.ndarray]) -> xr.Dataset:
    return _to_levs(ds, levs, lev_type="air_pressure")


def compute_additional_variables_inplace(ds: xr.Dataset):
    """Compute some useful variables"""

    logging.info("Computing additional variables inplace")
    ds["wind_speed_10"] = mcalc.wind_speed(
        ds["U10"].metpy.quantify(), ds["V10"].metpy.quantify()
    ).metpy.dequantify()
    ds["wind_speed"] = mcalc.wind_speed(
        ds["U"].metpy.quantify(), ds["V"].metpy.quantify()
    ).metpy.dequantify()
    u10 = ds["U10"].compute()
    v10 = ds["V10"].compute()
    u10 = ds["U10"].compute()
    v10 = ds["V10"].compute()
    wdir_10 = mcalc.wind_direction(
        u10.metpy.quantify(), v10.metpy.quantify()
    ).metpy.dequantify()
    ds["wind_direction_10"] = wdir_10.where(
        wdir_10 >= 0, wdir_10 + Quantity(360, "deg")
    )
    u = ds["U"].compute()
    v = ds["V"].compute()
    u = ds["U"].compute()
    v = ds["V"].compute()
    wdir = mcalc.wind_direction(
        u.metpy.quantify(), v.metpy.quantify()
    ).metpy.dequantify()
    ds["wind_direction"] = wdir.where(wdir >= 0, wdir + Quantity(360, "deg"))

    ds["Ta"] = mcalc.temperature_from_potential_temperature(
        ds["air_pressure"].metpy.quantify(),
        ds["air_potential_temperature"].metpy.quantify(),
    ).metpy.dequantify()
    ds["RH"] = mcalc.relative_humidity_from_specific_humidity(
        ds["air_pressure"].metpy.quantify(),
        ds["Ta"].metpy.quantify(),
        ds["QVAPOR"].metpy.quantify(),
    ).metpy.dequantify()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessor to work directly on wrfout files"
    )
    parser.add_argument(
        "--wrf-run",
        type=str,
        default="/scratch365/swang18/Workspace/apps/wrf/run/",
        help="Path to WRF run directory",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="fatima-20220720_20220723",
        help="Name of the WRF case",
    )
    parser.add_argument(
        "--file-prefix", type=str, default="wrfout_d01", help="Prefix for wrfout files"
    )
    parser.add_argument(
        "--proc-dir", type=str, default="./data", help="Processing output directory"
    )
    parser.add_argument(
        "--levs",
        type=str,
        default="np.arange(100, 2000, 100)",
        help="Height levels expression (e.g., 'np.arange(10,100,5)')",
    )
    parser.add_argument(
        "--interp_var",
        choices=["geopotential_height", "air_pressure"],
        default="geopotential_height",
        help="variable used for vertical level interpolation. Used in conjection with --levs",
    )
    parser.add_argument(
        "--logfile", help="name or full path of the logfile", default="./app.log"
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.logfile,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    )
    logging.info(f"program args: {args}")

    ## Input
    run_dir = Path(args.wrf_run)
    case_name = args.case_name
    case_dir = run_dir / case_name
    file_prefix = args.file_prefix
    levs = parse_arange_expr(args.levs)

    proc_dir = Path(args.proc_dir) / case_name

    proc_dir.mkdir(exist_ok=True, parents=True)

    fpaths: list[Path] = list(case_dir.glob(f"{file_prefix}*"))
    logging.info(f"wrfout files detected : {fpaths}")

    ds = read_wrfout(fpaths, t_chunk=1)

    ## Unify chunks to avoid inconsistencies
    ds = ds.unify_chunks()

    ## Compute relevant variables

    compute_additional_variables_inplace(ds)

    ## Project to height levels
    ds_p = (
        to_hlevs(ds, levs)
        if args.interp_var == "geopotential_height"
        else to_plevs(ds, levs)
    )

    ## Replace wrf_projection
    crs: pyproj.CRS = ds_p.wrf_projection.item()
    ds_p["wrf_projection"] = crs.to_proj4()

    ## Include the preprocessing config parameters into the output
    ds_p.attrs.update(
        dict(
            PREPROCESS_WRFRUN=args.wrf_run,
            PREPROCESS_CASE_NAME=args.case_name,
            PREPROCESS_FILE_PREFIX=args.file_prefix,
            PREPROCESS_LEVS=args.levs,
            PREPROCESS_INTERP_VAR=args.interp_var,
            PREPROCESS_TIMESTAMP=pd.Timestamp.now().isoformat(),
        )
    )

    ### Export to Zarr
    zarr_dir = proc_dir
    zarr_dir.mkdir(exist_ok=True)

    with ProgressBar():
        ds_p.to_zarr(zarr_dir / f"{file_prefix}_hlevs.zarr", mode="w")


if __name__ == "__main__":
    main()
