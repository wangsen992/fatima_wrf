#!/bin/bash

if [[ -z $WRF_DIR || -z $WPS_DIR ]]; then
  echo "Either WRF_DIR or WPS_DIR is not defined, source WRF installation properly before continue"
  exit 1
else
  echo "WRF_DIR=${WRF_DIR}"
  echo "WPS_DIR=${WPS_DIR}"
  export PATH=$PATH:${WRF_DIR}/main:${WPS_DIR}
fi

# source useful shell functions to running wrf cases
CWD=$(pwd)/$(dirname $0)
source ${CWD}/bin/RunFunctions.sh
