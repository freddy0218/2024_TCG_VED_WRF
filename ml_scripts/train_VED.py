"""──────────────────────────────────────────────────────────────────────────┐
│ Loading necessary libraries to build and train model                       │
└──────────────────────────────────────────────────────────────────────────"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os,sys,gc
import numpy as np
import pickle
import torch
import proplot as plot
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import glob
import properscoring as ps
from copy import deepcopy
plot.rc.update({'figure.facecolor':'w','axes.labelweight':'ultralight',
                'tick.labelweight':'ultralight','gridminor.linestyle':'--','title.weight':'normal','linewidth':0.5})
import random
import sys
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/')
from util.ml import (preproc,vae)
from util.wrf_process import (read_and_write)
import read_config

# Read configuration file
config_set = read_config.read_config('../config.ini')
i=int(str(sys.argv[1]))
exptype=str(sys.argv[2])

startname=int(12)

suffix = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/storage'
X = read_and_write.depickle(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[i])
validindices = sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][int(startname):].split('.')[0]

#PCA = read_and_write.depickle(sorted(glob.glob(f'../storage/proc/PCA/{exptype}/PCAsmooth{exptype}*'+str(validindices)+'*'))[0])
y = read_and_write.depickle(sorted(glob.glob('../storage/proc/y*'+str(validindices)+'*'))[0])
#X['test'] = X.pop('Xtest')
nummem = X['sizes']

train_data,val_data,test_data = preproc.prepare_tensors(X,y,'No')
batch_size = 10
num_workers = 2
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=False)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False)
del X,y
gc.collect()
    
read_and_write.save_to_pickle(nummem,suffix+f'/proc/VEDsmooth_{exptype}/'+
                              str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][int(startname):].split('.')[0])+'/losscoeff_0/'+'nummem.pkl')
losscoeff=float(config_set['ML_losscoeff'])

study = read_and_write.depickle(suffix+f'/proc/VEDsmooth_{exptype}/'+
                                str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][int(startname):].split('.')[0])+'/losscoeff_0/'+'bestparams.pkt')

times = ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']
for itime in tqdm(range(len(times))):
    models,losses = [],[]
    model = vae.VAE(nummem[-2],nummem[-1],1,1,1,nummem)
    optimizers = [torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])]
    loss = torch.nn.L1Loss()
    for optimizer in optimizers:
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.5e-8, max_lr=7e-5,cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)  #1e-18
        num_epochs = 1000*40#26
        early_stopper = vae.EarlyStopping(patience=2000, verbose=False, delta=1.5e-5, path='checkpoint.pt', trace_func=print)
        model,loss,_ = vae.train_model(model=model,optimizer=optimizer,scheduler=[scheduler,scheduler2],numepochs=num_epochs,early_stopper=early_stopper,variance_store=None,\
                                lossfunc=loss,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,l2_lambda=study.best_params['l2_lambda'],count=10,vaeloss_coeff=losscoeff)
        models.append(model)
        losses.append(loss)
    #torch.save(models, '../tmp/torch_try/ts/'+str(expname)+'/0/'+'models'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pt')
    #read_and_proc.save_to_pickle('../tmp/torch_try/ts/'+str(expname)+'/0/'+'losses'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pkt',losses,'PICKLE')
    if losscoeff==1.0:
        losscoeff2 = int(losscoeff)
        torch.save(models,suffix+f'/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][int(startname):].split('.')[0])+'/losscoeff_0/'+'modelstest_vae_'+str(times[itime])+'.pk')
        read_and_write.save_to_pickle(losses,suffix+f'/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][int(startname):].split('.')[0])+'/losscoeff_0/'+'lossestest_vae_'+str(times[itime])+'.pkt')
    else:
        torch.save(models,filepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_vae_'+str(times[itime])+'.pk')
        
#read_and_write.save_to_pickle(nummem,suffix+'/proc/VED/'+str(sorted(glob.glob('../storage/proc/X*pkl'))[i].split('/')[-1][12:].split('.')[0])+'/losscoeff_0/'+'nummem.pkl')

#read_and_write.save_to_pickle(filepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/lossestest'+str(splitnum)+'_vae_'+str(times[itime])+'.pkt',losses,'PICKLE')  
