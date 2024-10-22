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

config_set = read_config.read_config('../config.ini')
imemb=int(str(sys.argv[1]))
vae_losscoeff=float(sys.argv[2])
exptype=str(sys.argv[3])

startname=int(12)
class resume_training:
    def __init__(self,splitnum=None,droprate=None,nonln_num=None,timelag=None,batch_size=None,num_workers=2):
        self.splitnum=splitnum
        self.droprate=droprate
        self.vaeloss_coeff=nonln_num
        self.timelag = timelag
        self.batch_size = batch_size
        self.num_workers=2
        
    def get_data(self,suffix='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/',config_set=config_set):
        #PCA = read_and_write.depickle(sorted(glob.glob(suffix+'storage/proc/PCA/PCA*'))[self.splitnum])
        X = read_and_write.depickle(sorted(glob.glob(suffix+f'storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum])
        y = read_and_write.depickle(sorted(glob.glob(suffix+'storage/proc/y*'))[self.splitnum])
        #X['test'] = X.pop('Xtest')
        
        validindices = sorted(glob.glob(suffix+f'storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0]
        brchindex = X['sizes']
        #LWstop = np.abs(PCA['PCA']['LW'].explained_variance_ratio_.cumsum()-float(config_set['ML_LWnumcomps'])).argmin()
        #SWstop = np.abs(PCA['PCA']['SW'].explained_variance_ratio_.cumsum()-float(config_set['ML_SWnumcomps'])).argmin()
        train_data,val_data,test_data = preproc.prepare_tensors(X,y,'No')
        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=self.batch_size,shuffle=False)
        return train_loader,val_loader,test_loader,brchindex#[0,LWstop,SWstop]
    
    def continue_training(self,suffix='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/',config_set=None,exp='e',scheduler_lr=[1e-14,5e-10],early_stopper=None):
        i=self.splitnum
        train_loader,val_loader,_,brchindex = self.get_data(config_set=config_set)
        study = read_and_write.depickle(suffix+f'storage/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0])+\
                                        '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'bestparams.pkt')
        original_model = vae.VAE(brchindex[-2],brchindex[-1],1,1,1,brchindex)
        #######################################################################################################################################
        # Transfer state dict
        pretrained_model = torch.load(suffix+f'storage/proc/VEDsmooth_{exptype}/'+\
                                      str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0])+\
                                      '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'modelstest_vae_'+str(exp)+'.pk')[0]
        model_dict = original_model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        original_model.load_state_dict(model_dict)
        #######################################################################################################################################
        #######################################################################################################################################
        optimizer = torch.optim.Adam(original_model.parameters(), lr=study.best_params['lr'])
        #lossfunc = torch.nn.L1Loss()
        #scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-16, max_lr=5e-10,cycle_momentum=False) #1e-9/1e-5
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=scheduler_lr[0], max_lr=scheduler_lr[1],cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-20)
        #######################################################################################################################################
        
        lowest_val_loss = float('inf')
        best_model = None
        schedulerCY,schedulerLS = scheduler2,scheduler
        train_losses,trainrecon_losses,trainkl_losses = [],[],[]
        val_losses,valrecon_losses,valkl_losses = [],[],[]
        
        for epoch in tqdm(range(20000)):
            original_model.train()
            train_loss = 0
            trainrecon_loss = 0
            trainkl_loss = 0
            # Training loop here
            for features, labels in train_loader:
                optimizer.zero_grad()
                reconX,mu1,logvar1,mu2,logvar2 = original_model(features)
                batch_loss,recon_loss,kl_loss = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,self.vaeloss_coeff)
                batch_loss.backward()
                optimizer.step()
                schedulerCY.step()
                
                train_loss += batch_loss.item() 
                trainrecon_loss += recon_loss.item()
                trainkl_loss += kl_loss.item()
                
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            trainrecon_loss = trainrecon_loss / len(train_loader)
            trainrecon_losses.append(trainrecon_loss)
            trainkl_loss = trainkl_loss / len(train_loader)
            trainkl_losses.append(trainkl_loss)

            # Validation loop
            original_model.eval()
            with torch.no_grad():
                val_loss = 0
                val_reconloss = 0
                val_klloss = 0
                val_loss,val_reconloss,val_klloss = 0,0,0
                for features, labels in val_loader:
                    reconX,mu1,logvar1,mu2,logvar2 = original_model(features)
                    batch_loss,recon_loss,kl_loss = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,self.vaeloss_coeff)
                    val_loss+=batch_loss.item()
                    val_reconloss+=recon_loss.item()
                    val_klloss+=kl_loss.item()
            
                val_loss = val_loss / len(val_loader)
                val_reconloss = val_reconloss / len(val_loader)
                val_klloss = val_klloss / len(val_loader)
                val_losses.append(val_loss)
                valrecon_losses.append(val_reconloss)
                valkl_losses.append(val_klloss)

            # Check if the current model has the lowest validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = original_model#.state_dict()

            if early_stopper:
                if early_stopper.__call__(val_loss, original_model):
                    break
                
            #torch.save(best_model, savefilepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_vae_'+str(times[i])+'.pk')
            torch.save(original_model.state_dict(), suffix+f'storage/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0])+
                       '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'modelstest_vae_'+str(exp)+'_best_weights.pk')
            torch.save(best_model, suffix+f'storage/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0])+
                       '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'modelstest_vae_'+str(exp)+'_best.pk')
            read_and_write.save_to_pickle({'trainALL':train_losses,'valALL':val_losses,'trainRECON':trainrecon_losses,'valRECON':valrecon_losses,'trainKL':trainkl_losses,'valKL':valkl_losses},
                                          suffix+f'storage/proc/VEDsmooth_{exptype}/'+str(sorted(glob.glob(f'../storage/proc/Xsmooth/{exptype}/Xtimeseries*'))[self.splitnum].split('/')[-1][startname:].split('.')[0])+
                                          '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'lossestest_vae_'+str(exp)+'_best.pkt',
                                        )
        return None

for exp in ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']:#['a','b','c','d','e','f','g','h','i']:
    print(exp)
    early_stopper = vae.EarlyStopping(patience=200, verbose=False, delta=1.5e-5, path='checkpoint.pt', trace_func=print)
    resume_training(imemb,None,vae_losscoeff,None,5,2).continue_training(config_set=config_set,exp=exp,scheduler_lr=[1e-14,5e-10],early_stopper=early_stopper)
    #except:
    #    continue
