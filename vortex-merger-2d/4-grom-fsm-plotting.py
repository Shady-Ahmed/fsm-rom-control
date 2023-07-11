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

from numpy.random import seed
seed(0)

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
nx = 256    #spatial grid number
ny = 256
nr = 6       #number of modes 
re = 5000 #Reynolds number
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dts = 1e-1 #timestep between snapshots
tm = 50
dt = 1.0e-1 #timestep for GROM
nt = int(tm/dt)
ns = tm*10    #number of stored snapshots 
freq = int(nt/ns)

x = np.linspace(0,lx,nx+1)
y = np.linspace(0,ly,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')
t = np.linspace(0,tm,ns+1)

#%% FOM snapshot generation for training
print('Loading data/results...')

filename = './results/vm-fsm-full.npz'
dataF = np.load(filename)
wfom = dataF['wfom']
apod = dataF['apod']
wpod = dataF['wpod']
lpod = dataF['lpod']
agp = dataF['agp']
wgp = dataF['wgp']
lgp = dataF['lgp']
afsmF = dataF['afsm']
wfsmF = dataF['wfsm']
lfsmF = dataF['lfsm']
wobsF = dataF['wobs']
zF = dataF['z']

filename = './results/vm-fsm-sparse.npz'
dataS = np.load(filename)
afsmS = dataS['afsm']
wfsmS = dataS['wfsm']
lfsmS = dataS['lfsm']


#%%

colormap = 'RdGy_r'

nlvl = 60
lvl0 = 0.05
lvl1 = 1
lvl = np.linspace(lvl0, lvl1, nlvl, endpoint=True)
ctick = np.linspace(0, lvl1, 5, endpoint=True)

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
ax = ax.flat
for i, time in enumerate([0,10,20,30,40,50]):
    n = np.where(t==time)[0][0]
    
    ct=ax[i].contour(x,y,wfom[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    ax[i].set_xlabel(r'$x$', fontsize = 26)
    ax[i].set_ylabel(r'$y$', fontsize = 26)
    ax[i].set_title(r'$t='+str(time)+'$', fontsize = 26)

    ax[i].set_xticks([0,2,4,6])
    ax[i].set_yticks([0,2,4,6])

fig.subplots_adjust(bottom=0.18,hspace=0.45,wspace=0.45)
cbar_ax = fig.add_axes([0.31, 0.05, 0.4, 0.04])
 
norm= mpl.colors.Normalize(vmin=0, vmax=ct.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = ct.cmap)
sm.set_array([])
CB = fig.colorbar(sm, cax = cbar_ax, ticks=ctick, orientation='horizontal',extend='both')
plt.savefig('./plots/vmfom.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/vmfom.pdf', dpi = 300, bbox_inches = 'tight')

    

#%%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(15,8))
ax = ax.flat
for i, k in enumerate([0,1,2,3,4,5]):
    ax[i].plot(t,apod[k,:], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    ax[i].plot(t,agp[k,:], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    ax[i].plot(t,afsmF[k,:], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
    ax[i].set_xlabel(r'$t$', fontsize=20, labelpad=5)
    ax[i].set_ylabel(r'$a_{'+str(k+1) +'}(t)$', fontsize=20, labelpad=0)

ax[0].legend(loc="center", bbox_to_anchor=(1.9,1.15), ncol=5, fontsize=18, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.savefig('./plots/vm-coeff-full.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/vm-coeff-full.pdf', dpi = 500, bbox_inches = 'tight')

#%%
fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(15,8))
ax = ax.flat
for i, k in enumerate([0,1,2,3,4,5]):
    ax[i].plot(t,apod[k,:], label=r'\bf{True Closure}', color = 'C0', linewidth=2)
    ax[i].plot(t,agp[k,:], label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
    ax[i].plot(t,afsmS[k,:], label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
    ax[i].set_xlabel(r'$t$', fontsize=20, labelpad=5)
    ax[i].set_ylabel(r'$a_{'+str(k+1) +'}(t)$', fontsize=20, labelpad=0)

ax[0].legend(loc="center", bbox_to_anchor=(1.9,1.15), ncol=5, fontsize=18, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.savefig('./plots/vm-coeff-sparse.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./plots/vm-coeff-sparse.pdf', dpi = 500, bbox_inches = 'tight')


#%%
colormap = 'RdGy_r'
nlvl = 60
    
lvl0 = 0.05
lvl1 = 1
lvl = np.linspace(lvl0, lvl1, nlvl, endpoint=True)
ctick = np.linspace(0, lvl1, 5, endpoint=True)


fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
#ax = ax.flat
for i, time in enumerate([40,50]):
    n = np.where(t==time)[0][0]
    
    ct=ax[i][0].contour(x,y,wpod[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ct=ax[i][1].contour(x,y,wgp[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ct=ax[i][2].contour(x,y,wfsmF[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ax[i][0].text(-0.7, 0.45, r"$t = " + str(time) +"$", va='center',fontsize=25, transform=ax[i][0].transAxes)

    for j in range(3):
        ax[i][j].set_xlabel(r'$x$', fontsize = 26)
        ax[i][j].set_ylabel(r'$y$', fontsize = 26)    
        ax[i][j].set_xticks([0,2,4,6])
        ax[i][j].set_yticks([0,2,4,6])


ax[0][0].set_title(r'\bf{True Closure}', fontsize = 26, pad=10)
ax[0][1].set_title(r'\bf{No Closure}', fontsize = 26, pad=10)
ax[0][2].set_title(r'\bf{FSM Closure}', fontsize = 26, pad=10)


rect = plt.Rectangle(
    (-0.02, 0.51), 0.94, 0.39, fill=False, color="g", lw=2.5, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='-',alpha=1)

rect = plt.Rectangle(
    (-0.02, 0.09), 0.94, 0.39, fill=False, color="g", lw=2.5, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='-')

fig.subplots_adjust(bottom=0.2,hspace=0.35,wspace=0.45)
cbar_ax = fig.add_axes([0.31, 0.05, 0.4, 0.04])
 
norm= mpl.colors.Normalize(vmin=0, vmax=ct.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = ct.cmap)
sm.set_array([])
CB = fig.colorbar(sm, cax = cbar_ax, ticks=ctick, orientation='horizontal',extend='both')

plt.savefig('./plots/vm-field-full.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/vm-field-full.pdf', dpi = 300, bbox_inches = 'tight')

#%%
# Define the mean(spatially) squared loss for a single pair (ut,up)
def squared_error(ut, up):
    return jnp.inner(ut-up, ut-up)/jnp.shape(ut)[0]

# Batched version via vmap
mse_batched = vmap(squared_error, in_axes=(1,1))

lpodw = np.mean(wpod**2,axis=0)
lgpw = mse_batched(wgp,wpod)
lfsmFw = mse_batched(wfsmF,wpod)
lfsmSw = mse_batched(wfsmS,wpod)

lpoda = np.mean(apod**2,axis=0)
lgpa = mse_batched(agp,apod)
lfsmFa = mse_batched(afsmF,apod)
lfsmSa = mse_batched(afsmS,apod)

#%%
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

ax[0].plot(t,np.sqrt(lgpa/lpoda), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[0].plot(t,np.sqrt(lfsmFa/lpoda), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[0].set_xlabel(r"$t$", fontsize=22)
ax[0].set_ylabel(r"$\|\widehat{a}-a\|_2  \ / \  \|\widehat{a}\|_2$", fontsize=22, labelpad=8)
    
ax[1].plot(t,np.sqrt(lgpw/lpodw), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[1].plot(t,np.sqrt(lfsmFw/lpodw), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[1].set_xlabel(r"$t$", fontsize=22)
ax[1].set_ylabel(r"$\|\widehat{\omega}-\omega\|_2  \ / \  \| \widehat{\omega}\|_2$", fontsize=22, labelpad=8)
    
ax[0].legend(loc="center", bbox_to_anchor=(1.1,-0.25), ncol=5, fontsize=18, handletextpad=0.4, columnspacing=2)
fig.subplots_adjust(wspace=0.35)

plt.savefig('./plots/vm-error-full.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/vm-error-full.pdf', dpi = 300, bbox_inches = 'tight')

#%%

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

ax[0].plot(t,np.sqrt(lgpa/lpoda), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[0].plot(t,np.sqrt(lfsmSa/lpoda), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[0].set_xlabel(r"$t$", fontsize=22)
ax[0].set_ylabel(r"$\|\widehat{a}-a\|_2  \ / \  \|\widehat{a}\|_2$", fontsize=22, labelpad=8)
    

ax[1].plot(t,np.sqrt(lgpw/lpodw), label=r'\bf{No Closure}',linestyle='-', color = 'C3', linewidth=2)
ax[1].plot(t,np.sqrt(lfsmSw/lpodw), label=r'\bf{FSM Closure}',linestyle='--', color = 'C2', linewidth=2)
ax[1].set_xlabel(r"$t$", fontsize=22)
ax[1].set_ylabel(r"$\|\widehat{\omega}-\omega\|_2  \ / \  \| \widehat{\omega}\|_2$", fontsize=22, labelpad=8)
    
ax[0].legend(loc="center", bbox_to_anchor=(1.1,-0.25), ncol=5, fontsize=18, handletextpad=0.4, columnspacing=2)
fig.subplots_adjust(wspace=0.35)

plt.savefig('./plots/vm-error-sparse.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/vm-error-sparse.pdf', dpi = 300, bbox_inches = 'tight')


#%%
colormap = 'RdGy_r'
nlvl = 60
    
lvl0 = 0.05
lvl1 = 1
lvl = np.linspace(lvl0, lvl1, nlvl, endpoint=True)
ctick = np.linspace(0, lvl1, 5, endpoint=True)

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
for i, time in enumerate([40,50]):
    n = np.where(t==time)[0][0]
    
    ct=ax[i][0].contour(x,y,wpod[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ct=ax[i][1].contour(x,y,wgp[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ct=ax[i][2].contour(x,y,wfsmS[:,n].reshape([nx+1,ny+1]),lvl,cmap=colormap,extend='both',linewidths=1)
    ct.set_clim([lvl0, lvl1])
    
    ax[i][0].text(-0.7, 0.45, r"$t = " + str(time) +"$", va='center',fontsize=25, transform=ax[i][0].transAxes)

    for j in range(3):
        ax[i][j].set_xlabel(r'$x$', fontsize = 26)
        ax[i][j].set_ylabel(r'$y$', fontsize = 26)    
        ax[i][j].set_xticks([0,2,4,6])
        ax[i][j].set_yticks([0,2,4,6])


ax[0][0].set_title(r'\bf{True Closure}', fontsize = 26, pad=10)
ax[0][1].set_title(r'\bf{No Closure}', fontsize = 26, pad=10)
ax[0][2].set_title(r'\bf{FSM Closure}', fontsize = 26, pad=10)


rect = plt.Rectangle(
    (-0.02, 0.51), 0.94, 0.39, fill=False, color="g", lw=2.5, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='-',alpha=1)

rect = plt.Rectangle(
    (-0.02, 0.09), 0.94, 0.39, fill=False, color="g", lw=2.5, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='-')

fig.subplots_adjust(bottom=0.2,hspace=0.35,wspace=0.45)
cbar_ax = fig.add_axes([0.31, 0.05, 0.4, 0.04])
  
norm= mpl.colors.Normalize(vmin=0, vmax=ct.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = ct.cmap)
sm.set_array([])
CB = fig.colorbar(sm, cax = cbar_ax, ticks=ctick, orientation='horizontal',extend='both')

plt.savefig('./plots/vm-field-sparse.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./plots/vm-field-sparse.pdf', dpi = 300, bbox_inches = 'tight')
