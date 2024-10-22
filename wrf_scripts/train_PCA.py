import numpy as np
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, interplevel, ll_to_xy)
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import xarray as xr
import gc, glob
import pickle
from tqdm import tqdm
import sys
sys.path.insert(1, '../')
from util.ml import (preproc)
from util.wrf_process import (read_and_write,calc_derive)
import read_config

comb_num=int(sys.argv[1])
exptype=int(sys.argv[2])
nx_sm = int(sys.argv[3])
nx_repeat = int(sys.argv[4])

rthratlw,rthratsw=[],[]
for memb in [1,2,3,4,5,6,7,8,9,10]:
    if memb<10:
        rthratlw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb0{memb}/proc/cutcart_rthratlw.pkl'))[0])['pol'])
        rthratsw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb0{memb}/proc/cutcart_rthratsw.pkl'))[0])['pol'])
    else:
        rthratlw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb{memb}/proc/cutcart_rthratlw.pkl'))[0])['pol'])
        rthratsw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb{memb}/proc/cutcart_rthratsw.pkl'))[0])['pol'])

def do_smoothing(f,nx_sm,nx_repeat,nt_smooth):
    rthratlw_smooth = []
    for i in (range(len(f))):
        rthratlw_smooth.append(calc_derive.smooth_var(f[i],nx_sm,nx_repeat,nt_smooth))
    return rthratlw_smooth

rthratlw_smooth,rthratsw_smooth=[],[]
for i in tqdm(range(len(rthratlw))):
    rthratlw_smooth.append(do_smoothing(rthratlw[i],nx_sm,nx_repeat,0))
    rthratsw_smooth.append(do_smoothing(rthratsw[i],nx_sm,nx_repeat,0))

rthratlw_flat,rthratsw_flat = [],[]
for memb in range(len(rthratlw)):
    rthratlw_flat.append([obj.reshape(-1) for obj in rthratlw_smooth[memb]][:-1])
    rthratsw_flat.append([obj.reshape(-1) for obj in rthratsw_smooth[memb]][:-1])
del rthratlw,rthratsw
gc.collect()

import random
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

n_batches=3
n_comps = 140
Xfiles = sorted(glob.glob(f'../storage/proc/PCA/0000/PCA_*.pkl'))
# Choosing 2 random numbers
#validindices = random.sample([0,1,2,3,4,5,6,7,9], 2)
valid1 = Xfiles[comb_num].split('/')[-1].split('.')[0].split('_')[1]
valid2 = Xfiles[comb_num].split('/')[-1].split('.')[0].split('_')[2]
validindices = [int(valid1),int(valid2)]
# Separate data
Xtrain,Xvalid,Xtest = {},{},{}
Xtrain['LW'],Xvalid['LW'],Xtest['LW'] = preproc.train_valid_test(expvarlist=rthratlw_flat,validindex=validindices,testindex=[8],concat='Yes')
Xtrain['SW'],Xvalid['SW'],Xtest['SW'] = preproc.train_valid_test(expvarlist=rthratsw_flat,validindex=validindices,testindex=[8],concat='Yes')
del rthratlw_flat,rthratsw_flat
gc.collect()
# Train PCA
PCAdict = {}
for ivar in tqdm(['LW','SW']):
    inc_pca = IncrementalPCA(copy=False,n_components=int(Xtrain[ivar].shape[0]/n_batches)-1)#n_comps)
    for X_batch in tqdm(np.array_split(Xtrain[ivar],n_batches)):
        inc_pca.partial_fit(X_batch)
    PCAdict[ivar]=inc_pca
# Save stuff    
read_and_write.save_to_pickle({'PCA':PCAdict},f'../storage/proc/PCA/{exptype}/PCAsmooth{exptype}_{validindices[0]}_{validindices[1]}.pkl')
read_and_write.save_to_pickle({'train':Xtrain,'valid':Xvalid,'test':Xtest},f'../storage/proc/Xsmooth/{exptype}/Xsmooth{exptype}_{validindices[0]}_{validindices[1]}.pkl')
del PCAdict,Xtrain,Xvalid,Xtest
gc.collect()
