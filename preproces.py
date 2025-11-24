"""preprocessing script"""

import pathlib
from cartopy import crs
import yaml

import numpy as np
import xarray as xr
import xwrf
import pint_xarray

import metpy.calc as mcalc
import xgcm

## Parallel Configuration
PARALLEL = False
if PARALLEL:
    from distributed import LocalCluster

    cluster = LocalCluster()
    client = cluster.get_client()

    print(f"Dask local cluster with client daskboard link @ {client.dashboard_link}")

    client.close()


def read_wrfout(paths, chunks=None):
    ds_raw: xr.Dataset = xr.concat(
        [xr.open_mfdataset(path).xwrf.postprocess() for path in paths], dim="Time"
    )
    if chunks:
        ds_raw = ds_raw.xwrf.destagger().chunk(chunks=chunks)
    ds_raw = ds_raw.sortby("Time")
    return ds_raw


def get_crs(ds: xr.Dataset):
    wrf_crs = crs.CRS(ds.wrf_projection[0].compute().item())
    wrf_crs_dict = wrf_crs.to_dict()
    new_crs = crs.LambertConformal(
        central_longitude=wrf_crs_dict["lon_0"],
        central_latitude=wrf_crs_dict["lat_0"],
        standard_parallels=(wrf_crs_dict["lat_1"], wrf_crs_dict["lat_2"]),
    )
    return new_crs


if __name__ == "__main__":
    ## Input
    with open("config.yaml", "r") as config_buf:
        config = yaml.load(config_buf, Loader=yaml.Loader)

        run_dir = pathlib.Path(config["wrf_run"])
        case_name = config["case_name"]
        case_dir = run_dir / case_name
        file_prefix = config["file_prefix"]

        proc_dir = pathlib.Path(config["proc_dir"]) / case_name

        proc_dir.mkdir(exist_ok=True)

        fpaths = list(case_dir.glob(f"{file_prefix}*"))

    ## Preprocessing
    if PARALLEL:
        chunks = {"x": 44, "y": 43, "Time": 122}
    else:
        chunks = None

    ds = read_wrfout(fpaths)

    ## Compute relevant variables
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

    # Project to pressure levels
    p_levs = np.arange(200, 1000, 10)
    target_levels = p_levs

    dims_2d = ds["T2"].dims
    dims_3d = ds["wind_east"].dims
    vars_2d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_2d)]
    vars_3d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_3d)]
    t_list = ds.Time.to_numpy()

    air_pressure = ds.air_pressure.metpy.quantify().pint.to("hPa").metpy.dequantify()
    grid = xgcm.Grid(ds, periodic=False)

    h_target = np.arange(0, 1000, 10)

    v3d_dict = {}
    for v3d in vars_3d:
        v3d_dict[v3d] = grid.transform(
            ds[v3d].metpy.dequantify(),
            "Z",
            h_target,
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

    ## Export to Zarr
    zarr_dir = proc_dir / "data_zarr"
    zarr_dir.mkdir(exist_ok=True)

    ds_p = ds_p.chunk(chunks)
    ds_p.to_zarr(zarr_dir / f"{file_prefix}.zarr", mode="w")
