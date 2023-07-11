# -*- coding: utf-8 -*-
"""
Plotting results for the following manuscript:
    "Forward sensitivity analysis and mode dependent control for closure modeling of Galerkin systems",
     Computers & Mathematics with Applications, 2023
     by: Shady Ahmed, Omer San

Last checked: Monday, July 10, 2023
@author: Shady Ahmed
contact: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

import sys

# format plots
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

from jax import numpy as jnp
from jax import vmap

#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

re  = 1e4 #control Reynolds
nu = 1/re   #control dissipation

tm = 1      #maximum time
nt = 100   #number of timesteps per each parameter value 

dt = tm/nt
t = np.linspace(0, tm, nt+1)

ns = 100   #number of snapshot per each parameter value 
freq = int(nt/ns)

nr = 6     #number of modes

#%% FOM snapshot generation for training
print('Loading data/results...')

filename = './results/full_obs.npz'
dataF = np.load(filename)

ufom = dataF['ufom']
apod = dataF['apod']
upod = dataF['upod']
lpod = dataF['lpod']
agp = dataF['agp']
ugp = dataF['ugp']
lgp = dataF['lgp']

afsmF = dataF['afsm']
ufsmF = dataF['ufsm']
lfsmF = dataF['lfsm']
uobsF = dataF['uobs']
zF = dataF['z']


filename = './results/sparse_obs.npz'
dataS = np.load(filename)
afsmS = dataS['afsm']
ufsmS = dataS['ufsm']
lfsmS = dataS['lfsm']
uobsS = dataS['uobs']
locS = dataS['loc']
zS = dataS['z']


#%%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
ax = ax.flat
for i, k in enumerate([0,5]):
    ax[i].plot(t,apod[k,:], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    ax[i].plot(t,agp[k,:], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    ax[i].plot(t,afsmF[k,:], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
    ax[i].set_xlabel(r'$t$', fontsize=18, labelpad=5)
    ax[i].set_ylabel(r'$a_{'+str(k+1) +'}(t)$', fontsize=18, labelpad=0)

ax[0].legend(loc="center", bbox_to_anchor=(1.15,1.1), ncol=5, fontsize=15, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.9, wspace=0.35)

plt.savefig('./plots/coeff_full.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/coeff_full.pdf', dpi = 500, bbox_inches = 'tight')

#%%
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
ax = ax.flat
for i, k in enumerate([0,5]):
    ax[i].plot(t,apod[k,:], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    ax[i].plot(t,agp[k,:], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    ax[i].plot(t,afsmS[k,:], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
    ax[i].set_xlabel(r'$t$', fontsize=18, labelpad=5)
    ax[i].set_ylabel(r'$a_{'+str(k+1) +'}(t)$', fontsize=18, labelpad=0)

ax[0].legend(loc="center", bbox_to_anchor=(1.15,1.1), ncol=5, fontsize=15, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.9, wspace=0.35)

plt.savefig('./plots/coeff_sparse.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/coeff_sparse.pdf', dpi = 500, bbox_inches = 'tight')

#%%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
ax = ax.flat
for k in range(2):

    line1,  = ax[k].plot(x,ufom[:,(k+1)*50], label=r'\bf{FOM}', color = 'k', linewidth=2)
    line2,  = ax[k].plot(x,upod[:,(k+1)*50], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    line3,  = ax[k].plot(x,ugp[:,(k+1)*50], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    line4,  = ax[k].plot(x,ufsmF[:,(k+1)*50], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)

    ax[k].set_xlabel(r'$x$',fontsize=18, labelpad=0)
    ax[k].set_ylabel(r'$u(x,t='+str(t[(k+1)*50]) +')$',fontsize=18)
ax[0].legend(handles=[line1, line2, line3, line4], loc="center", bbox_to_anchor=(1.2,1.1), ncol =5, fontsize = 15, handletextpad=0.3 ,columnspacing=2)
fig.subplots_adjust(hspace=0.9, wspace=0.4)

plt.savefig('./plots/field_full.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/field_full.pdf', dpi = 500, bbox_inches = 'tight')

#%%

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
ax = ax.flat
for k in range(2):
    line1,  = ax[k].plot(x,ufom[:,(k+1)*50], label=r'\bf{FOM}', color = 'k', linewidth=2)
    line2,  = ax[k].plot(x,upod[:,(k+1)*50], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    line3,  = ax[k].plot(x,ugp[:,(k+1)*50], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    line4,  = ax[k].plot(x,ufsmS[:,(k+1)*50], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)

    ax[k].set_xlabel(r'$x$',fontsize=18, labelpad=0)
    ax[k].set_ylabel(r'$u(x,t='+str(t[(k+1)*50]) +')$',fontsize=18)
ax[0].legend(handles=[line1, line2, line3, line4], loc="center", bbox_to_anchor=(1.2,1.1), ncol =5, fontsize = 15, handletextpad=0.3 ,columnspacing=2)
fig.subplots_adjust(hspace=0.9, wspace=0.4)

plt.savefig('./plots/field_sparse.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/field_sparse.pdf', dpi = 500, bbox_inches = 'tight')

#%%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
mpl.rc('font', **font)



nlvl = 101
low = ufom.min() + 0.1
high = ufom.max()  - 0.1
cval = np.linspace(low,high,nlvl)
mapp = 'coolwarm'


[xx,tt] = np.meshgrid(x,t, indexing='ij')
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
ax = ax.flat

for i in range(4):
    ax[i].set_rasterization_zorder(-1)
cs = ax[0].contourf(xx,tt,ufom,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[0].set_title(r'\bf{FOM}')

cs = ax[1].contourf(xx,tt,upod,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[1].set_title(r'\bf{True Closure}')

cs = ax[2].contourf(xx,tt,ugp,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[2].set_title(r'\bf{No Closure}')

cs = ax[3].contourf(xx,tt,ufsmF,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[3].set_title(r'\bf{FSM Closure}')

for i in range(4):
    ax[i].set_xlabel(r'$x$',fontsize=20, labelpad=0)
    ax[i].set_ylabel(r'$t$',fontsize=20)
    
fig.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig('./plots/field_full_cont.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/field_full_cont.pdf', dpi = 500, bbox_inches = 'tight')

#%%
[xx,tt] = np.meshgrid(x,t, indexing='ij')
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
ax = ax.flat

cs = ax[0].contourf(xx,tt,ufom,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[0].set_title(r'\bf{FOM}')

cs = ax[1].contourf(xx,tt,upod,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[1].set_title(r'\bf{True Closure}')

cs = ax[2].contourf(xx,tt,ugp,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[2].set_title(r'\bf{No Closure}')

cs = ax[3].contourf(xx,tt,ufsmS,cval,cmap=mapp,extend='both')
cs.set_clim([low,high])
ax[3].set_title(r'\bf{FSM Closure}')

for i in range(4):
    ax[i].set_xlabel(r'$x$',fontsize=20, labelpad=0)
    ax[i].set_ylabel(r'$t$',fontsize=20)
    
fig.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig('./plots/field_sparse_cont.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/field_sparse_cont.pdf', dpi = 500, bbox_inches = 'tight')


#%%
# Define the mean(spatially) squared loss for a single pair (ut,up)
def squared_error(ut, up):
    return jnp.inner(ut-up, ut-up)/jnp.shape(ut)[0]

# Batched version via vmap
mse_batched = vmap(squared_error, in_axes=(1,1))


lpodu = np.mean(upod**2,axis=0)
lgpu = mse_batched(ugp,upod)
lfsmFu = mse_batched(ufsmF,upod)
lfsmSu = mse_batched(ufsmS,upod)

lpoda = np.mean(apod**2,axis=0)
lgpa = mse_batched(agp,apod)
lfsmFa = mse_batched(afsmF,apod)
lfsmSa = mse_batched(afsmS,apod)


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

ax[0].plot(t,np.sqrt(lgpa/lpoda), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[0].plot(t,np.sqrt(lfsmFa/lpoda), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[0].set_xlabel(r"$t$", fontsize=22)
ax[0].set_ylabel(r"$\|\widehat{a}-a\|_2  \ / \  \|\widehat{a}\|_2$", fontsize=22, labelpad=8)
    

ax[1].plot(t,np.sqrt(lgpu/lpodu), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[1].plot(t,np.sqrt(lfsmFu/lpodu), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[1].set_xlabel(r"$t$", fontsize=22)
ax[1].set_ylabel(r"$\|\widehat{u}-u\|_2  \ / \  \| \widehat{u}\|_2$", fontsize=22, labelpad=8)
      
ax[0].legend(loc="center", bbox_to_anchor=(1.1,-0.25), ncol=5, fontsize=18, handletextpad=0.4, columnspacing=2)
fig.subplots_adjust(wspace=0.35)

plt.savefig('./plots/brg-error-full.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/brg-error-full.pdf', dpi = 300, bbox_inches = 'tight')


#%%
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

ax[0].plot(t,np.sqrt(lgpa/lpoda), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[0].plot(t,np.sqrt(lfsmSa/lpoda), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[0].set_xlabel(r"$t$", fontsize=22)
ax[0].set_ylabel(r"$\|\widehat{a}-a\|_2  \ / \  \|\widehat{a}\|_2$", fontsize=22, labelpad=8)
    
ax[1].plot(t,np.sqrt(lgpu/lpodu), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[1].plot(t,np.sqrt(lfsmSu/lpodu), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[1].set_xlabel(r"$t$", fontsize=22)
ax[1].set_ylabel(r"$\|\widehat{u}-u\|_2  \ / \  \| \widehat{u}\|_2$", fontsize=22, labelpad=8)
    
   
ax[0].legend(loc="center", bbox_to_anchor=(1.1,-0.25), ncol=5, fontsize=18, handletextpad=0.4, columnspacing=2)
fig.subplots_adjust(wspace=0.35)

plt.savefig('./plots/brg-error-sparse.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/brg-error-sparse.pdf', dpi = 300, bbox_inches = 'tight')
