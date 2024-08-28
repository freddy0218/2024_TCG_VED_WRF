import numpy as np
import sys

def relvort(u, v, lat, lon):
    
    a = 6371e3 # Earth radius, m
    deg2rad = np.pi/180
    deg2meters = a * deg2rad
    try:
        cosf = np.cos(np.radians(lat)).data
    except:
        cosf = np.cos(np.radians(lat))

    dudy = np.gradient( u , lat*deg2meters , axis=1)
    dvdx = np.gradient( v , lon*deg2meters , axis=2) / cosf[np.newaxis,:,np.newaxis]

    # print("Shape of gradient variable:",np.shape(dvdx))
    vor = (dvdx - dudy)

    return vor

def standardization(data):
    return (data-np.nanmean(data))/np.nanstd(data)
