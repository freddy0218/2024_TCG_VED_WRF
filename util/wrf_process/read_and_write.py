from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, interplevel)
import pickle

def get_basic_domain(ncfile=None):
    # Get the pressure levels
    pres = getvar(ncfile, "pres")
    # Get the sea level pressure
    slp = getvar(ncfile, "slp")
    # Get the latitude and longitude points
    lats, lons = latlon_coords(slp)
    return lons,lats,pres

import pickle

def save_to_pickle(data,savepath):
    with open(savepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def depickle(savepath):
    with open(savepath, 'rb') as handle:
        b = pickle.load(handle)
    return b

