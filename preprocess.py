"""preprocessing script"""

from typing import Union
from pathlib import Path
from cartopy import crs
import yaml

import numpy as np
import xarray as xr
import xwrf, pint_xarray

import metpy.calc as mcalc
import xgcm

import argparse
import logging


def read_wrfout(paths: list[Path], chunks=None) -> xr.Dataset:
    """Read raw wrfout (output directly from wrf) and return as one file"""
    logging.info(f"Reading wrfout files: {paths}")
    ds_raw: xr.Dataset = xr.concat(
        [xr.open_mfdataset(path).xwrf.postprocess() for path in paths], dim="Time"
    )
    if chunks:
        ds_raw = ds_raw.xwrf.destagger().chunk(chunks=chunks)
    ds_raw = ds_raw.sortby("Time")
    return ds_raw


def get_crs(ds: xr.Dataset) -> crs.CRS:
    """Get CRS from wrf output and parse to cartopy.CRS"""
    logging.info("Getting CRS from wrfout")
    wrf_crs = crs.CRS(ds.wrf_projection[0].compute().item())
    wrf_crs_dict = wrf_crs.to_dict()
    new_crs = crs.LambertConformal(
        central_longitude=wrf_crs_dict["lon_0"],
        central_latitude=wrf_crs_dict["lat_0"],
        standard_parallels=(wrf_crs_dict["lat_1"], wrf_crs_dict["lat_2"]),
    )
    return new_crs


def to_hlevs(ds: xr.Dataset, levs: Union[list, np.ndarray]) -> xr.Dataset:
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
            target_data=ds.geopotential_height,
            method="linear",
        )

    v2d_dict = {}
    for v2d in vars_2d:
        v2d_dict = ds[v2d].metpy.dequantify()
    v3d_dict.update(v2d_dict)

    ds_p = xr.Dataset(v3d_dict)
    ds_p["wrf_projection"] = ds.wrf_projection
    ds_p.attrs = ds.attrs
    ds_p = ds_p.persist()

    return ds_p


def compute_additional_variables_inplace(ds: xr.Dataset):
    """Compute some useful variables"""

    logging.info("Computing additional variables inplace")
    ds["wind_speed_10"] = mcalc.wind_speed(
        ds["U10"].metpy.quantify(), ds["V10"].metpy.quantify()
    ).metpy.dequantify()
    ds["wind_speed"] = mcalc.wind_speed(
        ds["U"].metpy.quantify(), ds["V"].metpy.quantify()
    ).metpy.dequantify()
    ds["wind_direction_10"] = mcalc.wind_direction(
        ds["U10"].metpy.quantify(), ds["V10"].metpy.quantify()
    ).metpy.quantify()
    ds["wind_direction"] = mcalc.wind_direction(
        ds["U"].metpy.quantify(), ds["V"].metpy.quantify()
    ).metpy.quantify()

    ds["Ta"] = mcalc.temperature_from_potential_temperature(
        ds["air_pressure"].metpy.quantify(),
        ds["air_potential_temperature"].metpy.quantify(),
    ).metpy.dequantify()
    ds["RH"] = mcalc.relative_humidity_from_specific_humidity(
        ds["air_pressure"].metpy.quantify(),
        ds["Ta"].metpy.quantify(),
        ds["QVAPOR"].metpy.quantify(),
    ).metpy.dequantify()


if __name__ == "__main__":
    logging.basicConfig(
        filename="preprocess.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Preprocessor to work directly on wrfout files"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="yaml file that provides configurations",
        default="./config.yaml",
    )
    parser.add_argument(
        "--parallel",
        help="use this flag if parallel processing with dask is needed",
        action="store_true",
    )
    args = parser.parse_args()
    logging.info(f"program args: {args}")

    ## Input
    with open("config.yaml", "r") as config_buf:
        config = yaml.load(config_buf, Loader=yaml.Loader)

        run_dir = Path(config["wrf_run"])
        case_name = config["case_name"]
        case_dir = run_dir / case_name
        file_prefix = config["file_prefix"]

        proc_dir = Path(config["proc_dir"]) / case_name

        proc_dir.mkdir(exist_ok=True, parents=True)

        fpaths: list[Path] = list(case_dir.glob(f"{file_prefix}*"))
    logging.info(f"wrfout files detected : {fpaths}")

    ## Parallel Configuration
    if args.parallel:
        from distributed import LocalCluster

        cluster = LocalCluster()
        client = cluster.get_client()

        logging.info(
            f"Dask local cluster with client daskboard link @ {client.dashboard_link}"
        )

        client.close()

    ds = read_wrfout(fpaths, chunks="auto" if args.parallel else None)

    ## Compute relevant variables

    compute_additional_variables_inplace(ds)

    ## Project to height levels
    ds_p = to_hlevs(ds, np.arange(100, 2000, 100))

    ### Export to Zarr
    zarr_dir = proc_dir / "data_zarr"
    zarr_dir.mkdir(exist_ok=True)

    ds_p = ds_p.chunk(chunks="auto")
    ds_p.to_zarr(zarr_dir / f"{file_prefix}.zarr", mode="w")
