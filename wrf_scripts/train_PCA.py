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
from util.wrf_process import (read_and_write)
import read_config

comb_num=int(sys.argv[1])

rthratlw,rthratsw=[],[]
for memb in [1,2,3,4,5,6,7,8,9,10]:
    if memb<10:
        rthratlw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb0{memb}/proc/cutcart_rthratlw.pkl'))[0])['pol'])
        rthratsw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb0{memb}/proc/cutcart_rthratsw.pkl'))[0])['pol'])
    else:
        rthratlw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb{memb}/proc/cutcart_rthratlw.pkl'))[0])['pol'])
        rthratsw.append(read_and_write.depickle(sorted(glob.glob(f'../storage/memb{memb}/proc/cutcart_rthratsw.pkl'))[0])['pol'])

rthratlw_flat,rthratsw_flat = [],[]
for memb in range(len(rthratlw)):
    rthratlw_flat.append([obj.reshape(-1) for obj in rthratlw[memb]][:-1])
    rthratsw_flat.append([obj.reshape(-1) for obj in rthratsw[memb]][:-1])
del rthratlw,rthratsw
gc.collect()

import random
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

n_batches=3
n_comps = 140
Xfiles = sorted(glob.glob(f'../storage/proc/PCA_*.pkl'))
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
    inc_pca = IncrementalPCA(n_components=int(Xtrain[ivar].shape[0]/n_batches)-1)#n_comps)
    for X_batch in (np.array_split(Xtrain[ivar],n_batches)):
        inc_pca.partial_fit(X_batch)
    PCAdict[ivar]=inc_pca
# Save stuff    
read_and_write.save_to_pickle({'PCA':PCAdict},f'../storage/proc/PCA_{validindices[0]}_{validindices[1]}.pkl')
#read_and_write.save_to_pickle({'train':Xtrain,'valid':Xvalid,'test':Xtest},f'../../storage/proc/X_{validindices[0]}_{validindices[1]}.pkl')
del PCAdict,Xtrain,Xvalid,Xtest
gc.collect()
