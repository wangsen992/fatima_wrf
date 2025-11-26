import logging
from pathlib import Path
import xarray as xr
import xwrf
from distributed import LocalCluster
from ..preprocess import read_wrfout

import os

if __name__ == "__main__":
    logging.basicConfig(
        filename="./app.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    )

    logging.debug("init cluster")
    # cluster = LocalCluster()
    # client = cluster.get_client()

    logging.debug("loading file now")
    fname = Path(
        "/scratch365/swang18/Workspace/apps/wrf/run/fatima-20220720_20220723/wrfout_d01_2022-07-20_00:00:00"
    )
    fname2 = Path(
        "/scratch365/swang18/Workspace/apps/wrf/run/fatima-20220720_20220723/wrfout_d01_2022-07-22_00:00:00"
    )
    # ds = xr.load_dataset(fname, chunks="auto")
    ds = read_wrfout([fname, fname2])
