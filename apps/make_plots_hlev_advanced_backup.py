#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Advanced interactive plotting script for WRF data at height levels.

This script provides enhanced visualization capabilities including:
- Vertical cross-sections
- Time series analysis
- Multiple variable comparisons
- Custom colormaps and styling

Usage:
    python make_plots_hlev_advanced.py <zarr_file_path>

Example:
    python make_plots_hlev_advanced.py ./data/fatima-20220720_20220723/wrfout_d01_hlevs.zarr
"""

import argparse
import pathlib
import sys
import numpy as np
import xarray as xr
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
import panel as pn
from cartopy import crs
from dask import config as dask_config

# Configure dask for synchronous execution
dask_config.set(scheduler="synchronous")

# Initialize plotting extensions
gv.extension("bokeh")
pn.extension()

# Import additional libraries
import xwrf
from distributed import LocalCluster, Client


def get_crs(ds):
    """Extract coordinate reference system from WRF dataset."""
    # wrf_projection is a scalar, so no indexing needed
    wrf_crs = crs.CRS(ds.wrf_projection.compute().item())
    wrf_crs_dict = wrf_crs.to_dict()
    new_crs = crs.LambertConformal(
        central_longitude=wrf_crs_dict["lon_0"],
        central_latitude=wrf_crs_dict["lat_0"],
        standard_parallels=(wrf_crs_dict["lat_1"], wrf_crs_dict["lat_2"])
    )
    return new_crs


def setup_dask_cluster(n_workers=4):
    """Set up a local dask cluster for parallel processing."""
    print(f"Using synchronous scheduler (n_workers={n_workers} ignored)")
    return None


def load_wrf_data(zarr_path):
    """Load WRF data from Zarr archive."""
    print(f"Loading WRF data from: {zarr_path}")
    ds = xr.open_zarr(zarr_path).sel(geopotential_height=slice(None, 300))
    print(f"Data loaded: {ds.dims}")
    return ds
    return new_crs

# Cell 7
run_dir = pathlib.Path("/scratch365/swang18/Workspace/Projects/FATIMA_RENETA_CASES/MYNN_Milbr_all106lev_last/")
case_dir = run_dir
zarr_dir = case_dir/"data_zarr"

# Cell 8
with open(case_dir/"namelist.wps", 'r') as f:
#with open("/scratch365/swang18/Workspace/apps/wrf/run/fatima-20220720_20220723/namelist.wps", 'r') as f:
    lines = f.readlines()

# Cell 9
input_dict = {}
for l in lines:
    if "=" in l:
        k, v = l.split("=")
        input_dict[k.strip()] = v.strip()

# Cell 10
truelat1 = float(input_dict["truelat1"][:-1])
truelat2 = float(input_dict["truelat2"][:-1])
ref_lat, ref_lon = float(input_dict["ref_lat"][:-1]), float(input_dict["ref_lon"][:-1])
dx0, dy0 = float(input_dict["dx"][:-1]), float(input_dict["dy"][:-1])


# Cell 11
parent_id = [int(i) for i in input_dict["parent_id"][:-1].split(",")]
parent_grid_ratio = [int(i) for i in input_dict["parent_grid_ratio"][:-1].split(",")]
i_parent_start = [int(i) for i in input_dict["i_parent_start"][:-1].split(",")]
j_parent_start = [int(i) for i in input_dict["j_parent_start"][:-1].split(",")]
e_we = [int(i) for i in input_dict["e_we"][:-1].split(",")]
e_sn = [int(i) for i in input_dict["e_sn"][:-1].split(",")]

# Cell 12
dom_crs =  crs.LambertConformal(
            central_longitude=ref_lon,
            central_latitude =ref_lat,
            standard_parallels=(truelat1, truelat2)
    )

# Cell 13
num_doms = len(parent_id)

# Cell 14
center_point = gv.Points((-60,44.50), crs=crs.PlateCarree())

# Cell 15
dom_arr = np.zeros((num_doms, 5,2))
dx, dy = np.zeros(num_doms), np.zeros(num_doms)
dx[0] = dx0
dy[0] = dy0
for i in range(1, num_doms):
    dx[i] = dx[i-1] / parent_grid_ratio[i]
    dy[i] = dy[i-1] / parent_grid_ratio[i]


# Cell 16
i = 0
dom1_x0 = 0 - dx[i] * e_we[i]/2
dom1_x1 = 0 + dx[i] * e_we[i]/2
dom1_y0 = 0 - dy[i] * e_sn[i]/2
dom1_y1 = 0 + dy[i] * e_sn[i]/2
dom_arr[0,:,:] = [(dom1_x0, dom1_y0), (dom1_x1, dom1_y0), (dom1_x1, dom1_y1),
                    (dom1_x0, dom1_y1), (dom1_x0, dom1_y0)
                   ]

for i in range(1,num_doms):
    domi_x0 = dom_arr[i-1,0,0] + i_parent_start[i]* dx[i-1]
    domi_x1 = domi_x0 + dx[i] * e_we[i]
    domi_y0 = dom_arr[i-1,0,1] + j_parent_start[i]* dy[i-1]
    domi_y1 = domi_y0 + dx[i] * e_sn[i]
    dom_arr[i, :, :] = [(domi_x0, domi_y0), (domi_x1, domi_y0), (domi_x1, domi_y1),
                        (domi_x0, domi_y1), (domi_x0, domi_y0)
                   ]


# Cell 17
doms = gv.Overlay([gv.Path(dom_arr[i,:,:], crs=dom_crs) for i in range(num_doms)])


# Cell 18
(
    center_point.opts(size=20, marker='o',color='red') \
    * doms.opts(gv.opts.Path(line_width=2))
    * gv.tile_sources.OSM
).opts(width=500, height=500)

# Cell 19
import metpy.calc as mcalc
from metpy.units import units

# Cell 20
ds = xr.open_zarr(zarr_dir/"wrfout_d05.zarr") 

# Cell 21
lon_0 = ds.attrs["STAND_LON"]
lat_0 = ds.attrs["MOAD_CEN_LAT"]
lat_1 = ds.attrs["TRUELAT1"]
lat_2 = ds.attrs["TRUELAT2"]

wrf_crs =  crs.LambertConformal(
            central_longitude=lon_0,
            central_latitude =lat_0,
            standard_parallels=(lat_1, lat_2)
    )

# Cell 22
dims_2d = ds["T2"].dims
dims_3d = ds["wind_east"].dims
vars_2d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_2d) ]
vars_3d = [ds[v].name for v in ds.variables.keys() if (ds[v].dims == dims_3d) ]
t_list = ds.Time.to_numpy()

# Cell 23
var2d_select = pn.widgets.Select(description="var2d", options=vars_2d, value=vars_2d[0])
var3d_select = pn.widgets.Select(description="var3d", options=vars_3d, value=vars_3d[0])
time_widget = pn.widgets.DiscretePlayer(name="time", options=list(t_list),value=t_list[0])
# pres_widget = pn.widgets.Select(name="pressure", options=list(p_levs),value=p_levs[0])
h_widget = pn.widgets.Select(name="height", options=list(ds.geopotential_height.values),value=100)

# Cell 24
gv.opts.defaults(gv.opts.Image(colorbar=True, width=500, height=400, cmap='jet'),)

# Cell 25
x0, x1, y0, y1 = ds.XLONG.min().compute().item(), ds.XLONG.max().compute().item(),\
                 ds.XLAT.min().compute().item(), ds.XLAT.max().compute().item(),  

# Cell 26
cl = gf.coastline().geoms('50m', bounds=(x0,y0,x1,y1))\
    .opts(projection=crs.PlateCarree(),
          xlim=(x0, x1), ylim=(y0, y1), 
          line_color='black',
          line_width=2)

# Cell 27
%%time
def get_image_2d(var_name, time, **opts):
    return gv.Image(ds[var_name].sel(Time=time), ['x','y'], crs=wrf_crs).opts(title=f"{var_name} - {time}", **opts) \
            * cl
def get_image_3d(var_name, time, hlev, **opts):
    return gv.Image(ds[var_name].sel(Time=time, geopotential_height=hlev), ['x','y'], crs=wrf_crs)\
             .opts(title=f"{var_name} - {time}", **opts) \
            * cl
image_2d = pn.bind(get_image_2d, var2d_select, time_widget, tools=['hover'])
image_3d = pn.bind(get_image_3d, var3d_select, time_widget, h_widget, tools=['hover'])

# Cell 28
pane_sfc_2d = pn.Column(pn.panel(image_2d), var2d_select, time_widget)
pane_sfc_2d

# Cell 29
pn.Column(pn.panel(image_3d), var3d_select, time_widget, h_widget)

# Cell 30
p1 = np.array([-60.03666, 43.9146])
p2 = np.array([-60.0222, 43.9321])
p3 = p1 + 2* (p2 - p1)
line = gv.Path([tuple(p1),tuple(p3)], ["Longitude", "Latitude"], crs=crs.PlateCarree())\
         .opts(line_color='r', line_width=4)

# Cell 31
line * gv.tile_sources.OSM

# Cell 32
print(image_3d())

# Cell 33
line_wrf = gv.operation.project(line, 
                                projection=image_3d().Image.I.crs)

# Cell 34
image_2d() * line

# Cell 35
N = 500
arr = line_wrf.array()
px, py = np.linspace(*arr[:,0], N), np.linspace(*arr[:,1], N)
pxy = ((px-px.min())**2 +(py-py.min())**2)**(0.5) 
pz = ds.isel(geopotential_height=slice(0,250)).geopotential_height.values
p_slice = np.array([[px, py, np.ones(px.size) * pz[i]] for i in range(pz.size)]).transpose(0,2,1)

# Cell 36
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

# Cell 37
grid = [ds.y.values, ds.x.values, ds.geopotential_height.values]
points = np.array([p_slice[:,:,1].flatten(), p_slice[:,:,0].flatten(), p_slice[:,:,2].flatten()]).T

# Cell 38
def image_slice(time, var_3d, **opts):
    data = ds[var_3d].sel(Time=time).compute()
    values = data.values
    values_interped = interpn(grid, values, points).reshape(p_slice.shape[:-1])
    return hv.Image((pxy, pz, values_interped), ['loc', 'height'], var_3d)\
                .opts(cmap='jet', colorbar=True, 
                      width=700, height=300, 
                      invert_yaxis =False, 
                      title=str(time) + " " + var_3d, **opts)

# Cell 39
var3d_select1 = pn.widgets.Select(description="var3d", options=vars_3d, value=vars_3d[0])
time_widget1 = pn.widgets.DiscretePlayer(name="time", options=list(t_list),value=t_list[0])
plot_slice = pn.bind(image_slice, time_widget1, var3d_select1)

# Cell 40
pn.Column(pn.panel(plot_slice), time_widget1, var3d_select1)

