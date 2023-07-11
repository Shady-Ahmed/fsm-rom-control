#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full order model (FOM) data generation for the 2D vortex merger problem 

Ref: "Forward sensitivity analysis and mode dependent control for closure modeling of Galerkin systems",
      Computers & Mathematics with Applications, 2023
     by: Shady Ahmed, Omer San

Last checked: Monday, July 10, 2023
@author: Shady Ahmed
contact: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from utilities import (fps,arak_jac_per,laplacian_per)

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#%% Define Functions
# compute rhs using Arakawa scheme
# computed at all physical domain points (all boundary points included)
def rhs(nx,ny,dx,dy,re,w,s,x,y,ts):
    jac = arak_jac_per(nx,ny,dx,dy,w,s)
    lap = laplacian_per(nx,ny,dx,dy,w)
    r = -jac + lap/re
    return r

# set initial condition for vortex merger problem
def vm_ic(nx,ny,x,y):
    w = np.empty((nx+1,ny+1))
    sigma = np.pi
    xc1 = np.pi-np.pi/4.0
    yc1 = np.pi
    xc2 = np.pi+np.pi/4.0
    yc2 = np.pi
    w = np.exp(-sigma*((x-xc1)**2 + (y-yc1)**2)) + np.exp(-sigma*((x-xc2)**2 + (y-yc2)**2))
    return w
  
#%% inputs
    
re = 5000 #Reynolds number
  
pi = np.pi
lx = 2.0*pi
ly = 2.0*pi
nx = 256 #nx=ny
ny = 256

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

x = np.linspace(0.0,2.0*pi,nx+1)
y = np.linspace(0.0,2.0*pi,ny+1)

x, y = np.meshgrid(x, y, indexing='ij')

tm = 50 #maximum time
dt = 1.0e-3 #timestep
nt = int(tm/dt)

ns = tm*10 #number of saved snapshots (for each parameter value)
freq = int(nt/ns)

#create data folder
if os.path.isdir("./data/re"+str(re)):
    print('Data folder already exists')
else: 
    print('Creating data folder')
    os.makedirs("./data/re"+str(re))
    
#%% 
# allocate the vorticity and streamfunction arrays
w = np.empty((nx+1,ny+3)) 
s = np.empty((nx+1,ny+3))
ww = np.empty((nx+1,ny+1))
r = np.empty((nx+1,ny+1))

#%%
# set the initial condition
w0 = vm_ic(nx,ny,x,y)
w = np.copy(w0)
s = fps(nx, ny, dx, dy, -w)
time = 0.0
filename = "./data/re"+str(re)+"/w_0.npy"
np.save(filename,w)

#%%
# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
for k in range(1,nt+1):
    time = time + dt
    
    #stage-1
    r = rhs(nx,ny,dx,dy,re,w,s,x,y,time)
    ww = w + dt*r
    s = fps(nx, ny, dx, dy, -ww)
    
    #stage-2
    r = rhs(nx,ny,dx,dy,re,ww,s,x,y,time)
    ww = 0.75*w + 0.25*ww + 0.25*dt*r
    s = fps(nx, ny, dx, dy, -ww)
    
    #stage-3
    r = rhs(nx,ny,dx,dy,re,ww,s,x,y,time)
    w = aa*w + bb*ww + bb*dt*r
    s = fps(nx, ny, dx, dy, -w)
    
    if (k%freq == 0):
        filename = "./data/re"+str(re)+"/w_"+str(int(k/freq))+".npy"
        np.save(filename,w)

    if (k%(10*freq) == 0): #write on screen every 10 snapshots
        print(k, " ", time)

#%%

# create plot folder
if os.path.isdir("./plots"):
    print('Plots folder already exists')
else: 
    print('Creating plots folder')
    os.makedirs("./plots")


# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

cs = axs[0].contour(w0[1:nx+2,1:ny+2].T, 120, cmap = 'jet')
axs[0].text(0.4, -0.1, '$t = 0.0$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
cs = axs[1].contour(w[1:nx+2,1:ny+2].T, 120, cmap = 'jet')
axs[1].text(0.4, -0.1, '$t = '+str(dt*nt)+'$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("./data/field_fdm.png", bbox_inches = 'tight')
