import numpy as np
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)
from netCDF4 import Dataset
import xarray as xr
import gc, glob, tqdm

import sys
sys.path.insert(0,'../')
from util.wrf_process import (calc_derive, object_tracking, read_and_write)
import read_config

varname = str(sys.argv[1])
memb=str(sys.argv[2])
#############################################################################################
# Read configuration file
#############################################################################################
config_set = read_config.read_config('../config.ini')
# Experiment Type
if config_set['track_exp']=='CTRL':
    i_senstest = False
else:
    i_senstest = True

filelist = glob.glob(f'../storage/memb{memb}/wrfout_d02_2013-11-0*')
for file in filelist:
    print(file)
    #############################################################################################
    # Open files
    #############################################################################################
    # Open the NetCDF file
    ncfile = Dataset(file)
    # Get basic domain settings
    lons,lats,pres = read_and_write.get_basic_domain(ncfile)
    # Dates and Times
    dates = file.split('/')[-1].split('_')[-2]
    times = file.split('/')[-1].split('_')[-1].replace('%3A','_')[:5]
    #############################################################################################
    # Derive Relative Vorticity
    #############################################################################################
    # Prepare variable for tracking
    if varname=='U':
        uv = getvar(ncfile, "uvmet")
        var = uv[0]
    elif varname=='V':
        uv = getvar(ncfile, "uvmet")
        var = uv[1]
    else:
        var = getvar(ncfile, varname)
    #############################################################################################
    # Save to pickle
    #############################################################################################    
    filepath = f"../storage/memb{memb}/proc/{varname}/{dates}_{times}_{varname}.pkl"
    read_and_write.save_to_pickle(var,filepath)
    del ncfile,var
    gc.collect()
