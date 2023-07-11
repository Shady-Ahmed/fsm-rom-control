#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSM estimation of closure parameters from full field measurements 
for the 2D vortex merger problem

Ref: "Forward sensitivity analysis and mode dependent control for closure modeling of Galerkin systems",
      Computers & Mathematics with Applications, 2023
     by: Shady Ahmed, Omer San

Last checked: Monday, July 10, 2023
@author: Shady Ahmed
contact: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
import matplotlib as mpl

import os, sys
from utilities import (fps,arak_jac_per,laplacian_per)

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# format plots
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)


#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,nr): #Basis Construction
    n,ns = u.shape
    U,s,Vh = np.linalg.svd(u, full_matrices=False)
    phi = U[:,:nr]
    eigv = s**2
    #compute RIC (relative inportance index)
    ric = np.cumsum(eigv)/np.sum(eigv)*100   
    return phi,eigv,ric

def PODproj(u,phi): #Projection
    a = np.dot(phi.T, u)  # u = phi * a
    return a

def PODrec(a,phi): #Reconstruction    
    u = np.dot(phi,a)    
    return u


###############################################################################
# Galerkin ROM Routines
###############################################################################
def GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a): 
    
    l1 = lam[0]
    l2 = lam[1]

    b_cc1 = b_cc[0]
    b_cc2 = b_cc[1]

    b_lc1 = b_lc[0]
    b_lc2 = b_lc[1]
    
    r1, r2, r3 = [np.zeros(nr) for _ in range(3)]
    
    #constant term
    r1 = b_c + l1*b_cc1 + l2*b_cc2
    
    #linear term
    r2 = np.dot(b_l,a) + l1*np.dot(b_lc1, a) + l2*np.dot(b_lc2, a)
    # for k in range(nr):
    #     for i in range(nr):
    #         r2[k] += b_l[k,i]*a[i]
                
    #nonlinear term
    r3 = np.sum(np.dot(b_nl,a)*a, axis=1)
    # for k in range(nr):
    #     for i in range(nr):
    #         for j in range(nr):
    #             r3[k] += b_nl[k,i,j]*a[i]*a[j]

    r = r1 + r2 + r3
    return r

# time integration using third-order Runge Kutta method
def GROM_RK3(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a, dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    
    #stage-1
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a)
    a0 = a + dt*r

    #stage-2
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a0)
    a0 = 0.75*a + 0.25*a0 + 0.25*dt*r

    #stage-3
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a0)
    a = c1*a + c2*a0 + c2*dt*r

    return a


# GROM RHS term (projection of FOM operators onto ROM basis functions)
def const_term(nx,ny,dx,dy,wm2,sm2,re,phiw):
    '''
    this function returns 
    
    '''
    f = -arak_jac_per(nx,ny,dx,dy,wm2,sm2) \
        +(1/re)*laplacian_per(nx,ny,dx,dy,wm2)
    tmp = f.reshape(-1,)
    b_c = np.dot(tmp.T,phiw)
          
    return b_c

def lin_term(nx,ny,dx,dy,re,wm2,sm2,phiw,phis,nr):

    # linear term 
    b_l = np.zeros([nr,nr])
    for i in range(nr):
        phiw2 = phiw[:,i].reshape([nx+1,ny+1])
        phis2 = phis[:,i].reshape([nx+1,ny+1])
        f = -arak_jac_per(nx,ny,dx,dy,wm2,phis2) \
            -arak_jac_per(nx,ny,dx,dy,phiw2,sm2) \
            +(1/re)*laplacian_per(nx,ny,dx,dy,phiw2)
            
        tmp = f.reshape(-1,)
        b_l[:,i] = np.dot(tmp.T,phiw) 
            
    return b_l

def nonlin_term(nx,ny,dx,dy,phiw,phis,nr):
    '''
    this function returns -(Jacobian)
    
    '''
    # linear term 
    b_nl = np.zeros([nr,nr,nr])
    for i in range(nr):
        phiw2 = phiw[:,i].reshape([nx+1,ny+1])
        for j in range(nr):
            phis2 = phis[:,j].reshape([nx+1,ny+1])
            f = -arak_jac_per(nx,ny,dx,dy,phiw2,phis2)
             
            tmp = f.reshape(-1,)
            b_nl[:,i,j] = np.dot(tmp.T,phiw) 
                  
    return b_nl

def closure_const_term(nx,ny,dx,dy,wm2,re,phiw):
           
    # constant term
    tmp = wm2.reshape(-1,)
    b_cc0 = np.dot(tmp.T, phiw)  #damping/friction
    
    f = (1/re)*laplacian_per(nx,ny,dx,dy,wm2)
    tmp = f.reshape(-1,)
    b_cc1 = np.dot(tmp.T,phiw)
    
    b_cc = [b_cc0,b_cc1]
    return b_cc

def closure_lin_term(nx,ny,dx,dy,re,phiw,nr):

    b_lc0 = np.eye(nr) #friction/damping -- bases are orthonormal

    # linear term 
    b_lc1 = np.zeros([nr,nr])  #dissipation
    for i in range(nr):
        phiw2 = phiw[:,i].reshape([nx+1,ny+1])
        f = (1/re)*laplacian_per(nx,ny,dx,dy,phiw2)
        tmp = f.reshape(-1,)
        b_lc1[:,i] = np.dot(tmp.T,phiw)
        
    b_lc = [b_lc0,b_lc1]
    return b_lc


###############################################################################
# FSM ROM Routines
###############################################################################

# Jacobian of RHS
def GROMrhsJ(nr,lam,b_c,b_cc,b_l,b_lc,b_nl,a): #f(u)

    l1 = lam[0]
    l2 = lam[1]

    b_cc1 = b_cc[0]
    b_cc2 = b_cc[1]

    b_lc1 = b_lc[0]
    b_lc2 = b_lc[1]

    df = np.zeros([nr,nr+2*nr])
    df[:,:nr] = b_l + l1.reshape([-1,1])*b_lc1 + l2.reshape([-1,1])*b_lc2
    for k in range(nr):
        df[k,:nr] += np.dot(b_nl[k,:,:], a) + np.dot((b_nl[k,:,:]).T, a) - np.diag(b_nl[k,:,:])*a
                        
    df[:,nr:2*nr] = np.diag(b_cc1 + np.dot(b_lc1, a))
    df[:,2*nr:] = np.diag(b_cc2 + np.dot(b_lc2, a))
    
    return df  

# Forward sensitivites dynamics
def UVrhs(nr,  lam, b_c, b_cc, b_l, b_lc, b_nl, a, U, V):
    
    df = GROMrhsJ(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a)
    dfa = df[:,:nr]
    dfnue = df[:,nr:]
    ru = dfa @ U 
    rv = dfa @ V + dfnue

    return ru, rv


def Vrhs(nr,  lam, b_c, b_cc, b_l, b_lc, b_nl, a, V):
    
    df = GROMrhsJ(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a)
    dfa = df[:,:nr]
    dfnue = df[:,nr:]
    rv = dfa @ V + dfnue

    return rv


# time integration using third-order Runge Kutta method
def FSM_RK3(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a, V, dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    
    #stage-1
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a)
    rv = Vrhs(nr,  lam, b_c, b_cc, b_l, b_lc, b_nl, a, V)
    a0 = a + dt*r
    V0 = V + dt*rv

    #stage-2
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a0)
    rv = Vrhs(nr,  lam, b_c, b_cc, b_l, b_lc, b_nl, a0, V0)
    a0 = 0.75*a + 0.25*a0 + 0.25*dt*r
    V0 = 0.75*V + 0.25*V0 + 0.25*dt*rv

    #stage-3
    r = GROMrhs(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a0)
    rv = Vrhs(nr,  lam, b_c, b_cc, b_l, b_lc, b_nl, a0, V0)
    a = c1*a + c2*a0 + c2*dt*r
    V = c1*V + c2*V0 + c2*dt*rv

    return a, V  


# Observational map
def h(a):
    z = a
    return z

# Jacobian of observational map
def Dh(a):
    D = np.eye(len(a))
    return D


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


#%% Reading data
x = np.linspace(0,lx,nx+1)
y = np.linspace(0,ly,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')
t = np.linspace(0,tm,ns+1)

wfom = np.zeros([(nx+1)*(ny+1),ns+1])
wm = np.zeros((nx+1)*(ny+1))
sm = np.zeros((nx+1)*(ny+1))

for n in range(ns+1):
    file_input = "./data/re"+str(re)+"/w_"+str(int(n))+ ".npy"
    ww = np.load(file_input)
    wfom[:,n] = ww.reshape(-1,)

#%%
# mean subtraction
wm = np.mean(wfom,1)
wm2 = wm.reshape([nx+1,ny+1])
sm2 = fps(nx,ny,dx,dy,-wm2)
w = wfom - wm.reshape(-1,1)

#%% POD basis computation for training data
print('Computing POD basis...')
phiw = np.zeros([(nx+1)*(ny+1),nr]) # POD modes    
phis = np.zeros([(nx+1)*(ny+1),nr]) # POD modes     
eigv = np.zeros(ns+1) #Eigenvalues      
phiw,eigv,ric = POD(w,nr) 
for i in range(nr):
    phiw2 = phiw[:,i].reshape([nx+1,ny+1])
    phis2 = fps(nx,ny,dx,dy,-phiw2)
    phis[:,i] = phis2.reshape(-1,)

#%% Calculating true POD modal coefficients
atrue = np.zeros((ns+1,nr))
print('Computing true POD coefficients...')
atrue = PODproj(w,phiw)
#Unifying signs [not necessary]
phiw = phiw/np.sign(atrue[:,0].reshape([1,-1]))
phis = phis/np.sign(atrue[:,0].reshape([1,-1]))
atrue = atrue/np.sign(atrue[:,0].reshape([-1,1]))

#%% Galerkin projection precomputed coefficients
print('Computing GP coefficients...')
b_c = const_term(nx,ny,dx,dy,wm2,sm2,re,phiw)
b_l = lin_term(nx,ny,dx,dy,re,wm2,sm2,phiw,phis,nr)
b_nl = nonlin_term(nx,ny,dx,dy,phiw,phis,nr)
b_cc = closure_const_term(nx,ny,dx,dy,wm2,re,phiw)
b_lc = closure_lin_term(nx,ny,dx,dy,re,phiw,nr)

#%% Generate Observations from a twin experiment
print('Collecting twin experiment measurements...')

sig2 = 0.01
sig = np.sqrt(sig2)
R = sig**2*np.eye(nr)
Ri = np.linalg.inv(R)

Nz = 10 #number of observations per assimilation window

t_wind = tm #assimilation window
t_z = np.linspace(0,t_wind,Nz+1)

Nt = int(np.round(t_wind/dt,decimals=0)) #number of timesteps per assimilation window
Ns = int(Nt/freq)                        #number of snapshots per assimilation window
Ntz = int(Nt/Nz)       #number of timesteps between observations
Nsz = int(Ntz/freq)    #number of snapshots between observations

ind = np.arange(Nsz,Ns+1,Nsz)
wobs = wfom[:,ind] + np.random.normal(0,sig,wfom[:,ind].shape)
z = PODproj(wobs- wm.reshape(-1,1),phiw)
    
#%% FSM eddy viscosity estimation itertations
print('Estimating parameters via FSM...')

a0 = np.copy(atrue[:,0])
lam_b = [np.zeros(nr),np.zeros(nr)]
lam = np.copy(lam_b)
max_iter= 100
npp = 2*nr #number of parameters
for jj in range(max_iter):
    #U = np.eye(nr)
    V = np.zeros((nr,npp))

    H = np.zeros((1,npp))
    e = np.zeros((1,1))
    W = np.zeros((1,1)) #weighting matrix
    k = 0
    a = a0
    for i in range(1,Nt+1):#, Ntz+1):
         
        #propagate forward in time
        a, V = FSM_RK3(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a, V, dt)
        
        #check for observations
        if int(i/freq) == ind[k]:
            Hk = Dh(a) @ V
            H = np.vstack((H,Hk))
            ek = h(a) - z[:,k]
            e = np.vstack((e,ek.reshape([-1,1])))
            W = block_diag(W,Ri)
            k = k+1
            
    H = np.delete(H, (0), axis=0)
    e = np.delete(e, (0), axis=0)
    W = np.delete(W, (0), axis=0)
    W = np.delete(W, (0), axis=1)
    
    # solve weighted least-squares
    W1 = np.sqrt(W) 
    dc = np.linalg.lstsq(W1@H, -W1@e, rcond=None)[0]
    lam[0] = np.copy(lam[0]) + np.copy(dc[:nr].ravel())
    lam[1] = np.copy(lam[1]) + np.copy(dc[nr:].ravel())
    if np.linalg.norm(dc) <= 1e-6:
        break
      
#%% Solving ROM    
print('Solving ROM...')
    
agp = np.zeros([nr,ns+1])
afsm = np.zeros([nr,ns+1])

ag = np.copy(atrue[:,0])
agp[:,0] = np.copy(ag)

af = np.copy(atrue[:,0])
afsm[:,0] = np.copy(af)

time = 0
for k in range(1,nt+1):
    time = np.round(time+dt,decimals=5)
            
    ag = GROM_RK3(nr, 0*lam, b_c, b_cc, b_l, b_lc, b_nl, ag, dt) #solving GROM without closure 
    af = GROM_RK3(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, af, dt)   #solving GROM with closure 

    if (k%freq == 0):
        if (k/freq)%20 == 0:
            print(k, " ", time)
        agp[:,int(k/freq)] = np.copy(ag)
        afsm[:,int(k/freq)] = np.copy(af)

#%% Reconstruction
wpod = PODrec(atrue,phiw) + wm.reshape(-1,1)#Reconstruction   
wgp  = PODrec(agp,phiw)   + wm.reshape(-1,1)#Reconstruction   
wfsm = PODrec(afsm,phiw)  + wm.reshape(-1,1) #Reconstruction   

def RMSE(ua,ub):
    er = (ua-ub)**2
    er = np.mean(er)
    er = np.sqrt(er)
    return er

# Computing RMSE
lpod, lgp, lfsm = [np.zeros(ns+1) for _ in range(3)]
for i in range(ns+1):  
    lpod[i] = RMSE(wfom[:,i], wpod[:,i])
    lgp[i] = RMSE(wfom[:,i], wgp[:,i])
    lfsm[i] = RMSE(wfom[:,i], wfsm[:,i])  
 
#%% Saving results

#create data folder
if os.path.isdir("./results"):
    print('Reults folder already exists')
else: 
    print('Creating results folder')
    os.makedirs("./results")
 
print('Saving results')      
np.savez('./results/vm-fsm-full.npz', apod=atrue, agp=agp, afsm=afsm,\
                                      wpod=wpod, wgp=wgp, wfsm=wfsm,\
                                      lpod=lpod, lgp=lgp, lfsm=lfsm,\
                                      wfom=wfom, wobs=wobs, z=z, tind=ind)