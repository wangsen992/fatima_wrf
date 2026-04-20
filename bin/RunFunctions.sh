#!/bin/bash

function wrf_init_case() {
  # starting from a empty case
  #WPS
  ln -s $WPS_DIR/geogrid ./geogrid
  ln -s $WPS_DIR/metgrid ./metgrid
  ln -s $WPS_DIR/ungrib ./ungrib
  ln -s $WPS_DIR/util ./util
  ln -s $WPS_DIR/../WPS_GEOG ./WPS_GEOG

  cp $WPS_DIR/namelist.wps ./namelist.wps
  cp $WRF_DIR/run/namelist.input ./namelist.input

  MODEL=ECMWF
  ln -s $WPS_DIR/ungrib/Variable_Tables/Vtable.$MODEL Vtable

  # copy run case support files
  find $WRF_DIR/run -not -name *.exe \
    -exec cp {} ./ \;
}

function wrf_clean_case() {
  find . ! -name "cds_download.py" \
    ! -name "case_*.sh" \
    -not -path "." -exec rm -rf {} \;
}
