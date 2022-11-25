# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:28:52 2022

@author: Sascha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
import seaborn as sns
from scipy.stats import anderson_ksamp
from scipy.stats import gaussian_kde
'''
#''' #makes kernel crash
import torch
import torch.utils.data as data_utils
from torch.optim import Adam
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from tqdm.auto import tqdm
#'''

df = pd.read_csv('C:/Users/Sascha/Documents/Data_Challenges/Genhack/Flash/data/df_train.csv',index_col = 0)

X_train = df[['s1', 's2', 's3', 's4', 's5', 's6']].to_numpy()
'''
#KDE as benchmark Model==========================
kde = gaussian_kde(X_train.T)
samples = kde.resample(10**4)
for i in range(6):
    plt.figure()
    sns.kdeplot(data = X_train[:,i],fill=True)
    sns.kdeplot(data = samples[i,:])
#================================================
'''
#'''
train = torch.tensor(X_train.astype(np.float32)) 
train = train.reshape((train.shape[0],1,train.shape[1]))
train_tensor = data_utils.TensorDataset(train)
train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=32, shuffle=True)

model = Unet1D(
    dim = 64,
    #dim_mults = (1, 2, 4, 8),
    dim_mults = (1,2),
    channels = 1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 6,
    timesteps = 1000,
    objective = 'pred_v'
)
train_lr = 10**(-3)
adam_betas = adam_betas = (0.9, 0.99)
optimizer = Adam(diffusion.parameters(), lr = train_lr, betas = adam_betas)
num_iter = 10**3
for batch_ndx in range(num_iter):
    
    sample = next(iter(train_loader))
    optimizer.zero_grad()
    loss = diffusion(train)
    loss.backward()
    optimizer.step()
    print(batch_ndx)

@torch.no_grad()
def p_sample_loop(noise):
    #batch, device = shape[0], self.betas.device

    #img = torch.randn(shape, device=device)
    img = noise

    x_start = None

    for t in tqdm(reversed(range(0, diffusion.num_timesteps)), desc = 'sampling loop time step', total = diffusion.num_timesteps):
        self_cond = x_start if diffusion.self_condition else None
        img, x_start = diffusion.p_sample(img, t, self_cond)

    #img = unnormalize_to_zero_to_one(img)
    return img   

noise = torch.randn((1,1,6))
sample = p_sample_loop(noise)     
#training_seq = torch.randn(8, 1, 6) # features are normalized from 0 to 1
#loss = diffusion(training_seq)
#loss = diffusion(train)
#loss.backward()

# after a lot of training

#sampled_seq = diffusion.sample(batch_size = 100)
#sampled_seq.shape # (100, 1, 6)

#sns.kdeplot(data=X_train[:, 0], fill=True)
#sns.kdeplot(data=X_train[:, 0], fill=True)
#'''
