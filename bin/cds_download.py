import configparser
import logging
from pathlib import Path
from datetime import datetime

import cdsapi
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PRES_LEVS = [
    "1000",
    "975",
    "950",
    "925",
    "900",
    "875",
    "850",
    "825",
    "800",
    "775",
    "750",
    "700",
    "650",
    "600",
    "550",
    "500",
    "450",
    "400",
    "350",
    "300",
    "250",
    "200",
    "175",
    "150",
    "125",
    "100",
    "70",
    "50",
    "30",
    "20",
    "10",
    "7",
    "5",
    "3",
    "2",
    "1",
]
V_P = ["geopotential", "U", "V", "Temperature", "Relative humidity"]
V_SFC = [
    "surface_pressure",
    "Mean sea level pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "skin_temperature",
    "sea_surface_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "land_sea_mask",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

DATASET_PL = "reanalysis-era5-pressure-levels"
DATASET_SFC = "reanalysis-era5-single-levels"


def check_cds_auth():
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        raise FileNotFoundError(f"~/.cdsapirc not found at {cdsapirc}")

    config = configparser.ConfigParser()
    config.read(str(cdsapirc))

    if not config.has_section("url"):
        raise ValueError("~/.cdsapirc missing [url] section")
    if not config.has_section("key"):
        raise ValueError("~/.cdsapirc missing [key] section")

    url = config.get("url", "url")
    key = config.get("key", "key")
    if not url or not key:
        raise ValueError("~/.cdsapirc has empty url or key")

    logger.info("CDS API credentials validated")


def download_era5(client: cdsapi.Client, dt: datetime, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    request_pl = {
        "product_type": ["reanalysis"],
        "variable": V_P,
        "date": [dt.strftime("%Y-%m-%d")],
        "time": [dt.strftime("%H:%M")],
        "pressure_level": PRES_LEVS,
        "data_format": "grib",
    }
    target_pl = output_dir / f"era5_pl_{dt.strftime('%Y%m%d-%H%M')}.grb"
    logger.info(f"Downloading pressure levels: {target_pl}")
    client.retrieve(DATASET_PL, request_pl, str(target_pl))

    request_sfc = {
        "product_type": ["reanalysis"],
        "variable": V_SFC,
        "date": [dt.strftime("%Y-%m-%d")],
        "time": [dt.strftime("%H:%M")],
        "data_format": "grib",
    }
    target_sfc = output_dir / f"era5_sfc_{dt.strftime('%Y%m%d-%H%M')}.grb"
    logger.info(f"Downloading surface fields: {target_sfc}")
    client.retrieve(DATASET_SFC, request_sfc, str(target_sfc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ERA5 data from CDS API")
    parser.add_argument("--start-dt", required=True, help="Start datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--end-dt", required=True, help="End datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--freq", required=True, help="Frequency (pandas offset, e.g. 6h)")
    parser.add_argument(
        "--output-dir",
        default="data_extern",
        help="Output directory (default: data_extern)",
    )
    args = parser.parse_args()

    check_cds_auth()
    client = cdsapi.Client()

    dt_list = pd.date_range(start=args.start_dt, end=args.end_dt, freq=args.freq)
    output_dir = Path(args.output_dir)

    logger.info(f"Downloading ERA5 data from {args.start_dt} to {args.end_dt} at {args.freq} interval")
    logger.info(f"Output directory: {output_dir}")

    for dt in dt_list:
        try:
            download_era5(client, dt, output_dir)
        except Exception as e:
            logger.error(f"Failed to download data for {dt}: {e}")
            raise
