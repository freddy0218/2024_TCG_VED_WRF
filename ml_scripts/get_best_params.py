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

i=int(str(sys.argv[1]))
exptype=str(sys.argv[2])

startname=int(12)

def objective(trial):
    models,losses = [],[]
    model = vae.VAE(nummem[-2],nummem[-1],1,1,1,nummem)
    #droprate = trial.suggest_float("droprate",0.05,0.45)
    lr = trial.suggest_float("lr",1e-6,1e-3)#,log=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = vae.vae_loss
    n_epochs = 10000
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4,cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)

    schedulerCY,schedulerLS = scheduler2,scheduler

    l2_lambda = trial.suggest_float("l2_lambda",0.01,0.02)

    train_losses = []
    val_losses = []
    for epoch in range(1,n_epochs+1):
        loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,_,_ = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,losscoeff)
            batch_loss.backward()
            optimizer.step()
            schedulerCY.step()
            loss += batch_loss.item()
        loss = loss/len(train_loader)
        train_losses.append(loss)
        criterion = vae.vae_loss
        val_loss,_,_ = vae.eval_model(model,
                              val_loader,
                              criterion,
                             l2_lambda,
                                  losscoeff)
        schedulerLS.step(val_loss)
        val_losses.append(val_loss)
        if epoch%1000 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs))
            print("Loss: {:.4f}".format(loss))
    return loss

#print(sorted(glob.glob(f'../storage/proc/PCA/{exptype}/PCAsmooth{exptype}*')))
#print(aaaaa)
PCA = read_and_write.depickle(sorted(glob.glob(f'../storage/proc/PCA/{exptype}/PCAsmooth{exptype}*'))[i])
X = read_and_write.depickle(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[i])
y = read_and_write.depickle(sorted(glob.glob(f'../storage/proc/y*'))[i])
#X['test'] = X.pop('Xtest')
nummem = X['sizes']
validindices = sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][startname:].split('.')[0]
#LWstop = np.abs(PCA['PCA']['LW'].explained_variance_ratio_.cumsum()-0.5).argmin()
#SWstop = np.abs(PCA['PCA']['SW'].explained_variance_ratio_.cumsum()-0.8).argmin()

train_data,val_data,test_data = preproc.prepare_tensors(X,y,'No')
batch_size = 10
num_workers = 2

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)
del PCA,X,y
gc.collect()
    
import optuna
#nummem = [0,LWstop,SWstop]
losscoeff=1
study = optuna.create_study(directions=["minimize"])
study.optimize(objective, n_trials=6)#, timeout=300)

suffix = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/storage'
if losscoeff==1.0:
    losscoeff2 = int(losscoeff)
    read_and_write.save_to_pickle(study,
                                  suffix+f'/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][startname:].split('.')[0])+
                                  '/losscoeff_0/'+'bestparams.pkt')
    for losscoeff in [0.9,0.65,0.55,0.45,0.35,0.3,0.25,0.95,0.85,0.8,0.75,0.7,0.6,0.5,0.4]:
        read_and_write.save_to_pickle(study,suffix+f'/proc/VEDsmooth_{exptype}/'+
                                      str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*pkl'))[i].split('/')[-1][startname:].split('.')[0])+
                                      '/losscoeff_'+str(losscoeff)+'/'+'bestparams.pkt')
else:
    read_and_write.save_to_pickle(study,suffix+'/proc/VED/'+str(sorted(glob.glob('../storage/proc/X*pkl'))[i].split('/')[-1][12:].split('.')[0])+'/losscoeff_0/'+'bestparams.pkt')
