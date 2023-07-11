# -*- coding: utf-8 -*-
"""
FSM estimation of closure parameters from full field measurements 
for the 1D Burgers problem with an initial condition of a square wave

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
from numpy import linalg as LA
from scipy.linalg import block_diag

from numpy.random import seed
seed(0)

import os

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
    U,s,Vh = LA.svd(u, full_matrices=False)
    phi = U[:,:nr]
    eigv = s**2
    #compute RIC (relative inportance index)
    ric = np.cumsum(eigv)/np.sum(eigv)*100   
    return phi,eigv,ric

def PODproj(u, phi): #Projection
    a = np.dot(phi.T, u)  # u = Phi * a
    return a

def PODrec(a, phi): #Reconstruction    
    u = np.dot(phi, a)    
    return u

###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]   
    up[e] = r[e]/b[e]   
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd


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
    r2 = np.dot(b_l, a) + l1*np.dot(b_lc1, a) + l2*np.dot(b_lc2, a)
    
    #nonlinear term
    r3 = np.sum(np.dot(b_nl, a)*a, axis=1)

    r = r1 + r2 + r3
    return r



def GROMrhs2(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a): 
    
    l1 = lam[0]
    l2 = lam[1]

    b_cc1 = b_cc[0]
    b_cc2 = b_cc[1]

    b_lc1 = b_lc[0]
    b_lc2 = b_lc[1]
    
    r1, r2, r3 = [np.zeros(nr) for _ in range(3)]
    
    for k in range(nr):
        #constant term
        r1[k] = b_c[k] + l1[k]*b_cc1[k] + l2[k]*b_cc2[k]
        
        for i in range(nr):
            #linear term
            r2[k] += b_l[k,i]*a[i] + l1[k]*b_lc1[k,i]*a[i] + l2[k]*b_lc2[k,i]*a[i]
            for j in range(nr):
                #nonlinear term
                r3[k] += b_nl[k,i,j]*a[i]*a[j]
    r = r1 + r2 + r3
    return r

#%%

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
def const_term(nr, um, phi, re):

    umd = pade4d(um,dx,nx)     #first derivative of um    
    umdd = pade4dd(um,dx,nx)   #second derivative of um    
    
    # constant term
    b_c = np.zeros(nr)
    for k in range(nr):
        tmp = -um*umd + (1/re)*umdd
        b_c[k] = np.dot(tmp.T, phi[:,k])
            
    return b_c

def lin_term(nr, um, phi, re):
    
    umd = pade4d(um,dx,nx)     #first derivative of um    

    phid = np.zeros([nx+1,nr])
    phidd = np.zeros([nx+1,nr])
    for i in range(nr):
        phid[:,i] = pade4d(phi[:,i],dx,nx)   #first derivative of phi    
        phidd[:,i] = pade4dd(phi[:,i],dx,nx) #second derivative of phi    

    # linear term 
    b_l = np.zeros([nr,nr])
    for k in range(nr):
        for i in range(nr):
            tmp = -um*phid[:,i] - phi[:,i]*umd + (1/re)*phidd[:,i]
            b_l[k,i] = np.dot(tmp.T , phi[:,k]) 

    return b_l

def nonlin_term(nr, phi, re):
    
    phid = np.zeros([nx+1,nr])
    for i in range(nr):
        phid[:,i] = pade4d(phi[:,i], dx, nx) #first derivative of phi  
        

    # nonlinear term 
    b_nl = np.zeros([nr,nr,nr])    
    for k in range(nr):
        for i in range(nr):
            for j in range(nr):
                tmp = phi[:,i]*phid[:,j]
                b_nl[k,i,j] = - np.dot( tmp.T, phi[:,k] ) 

    return b_nl

def closure_const_term(nr, um, phi):
       
    umdd = pade4dd(um,dx,nx)   #second derivative of um    
    
    # constant term
    b_cc0 = np.zeros(nr)
    b_cc1 = np.zeros(nr)

    for k in range(nr):
        tmp = um
        b_cc0[k] = np.dot(tmp.T, phi[:,k])  #damping/friction

        tmp = umdd
        b_cc1[k] = np.dot(tmp.T, phi[:,k])  #dissipation
    
    b_cc = [b_cc0,b_cc1]
    return b_cc

def closure_lin_term(nr, um, phi, re):

    phidd = np.zeros([nx+1,nr])
    for i in range(nr):
        phidd[:,i] = pade4dd(phi[:,i],dx,nx) #second derivative of phi   
                
    b_lc0 = np.eye(nr) #friction/damping -- bases are orthonormal

    b_lc1 = np.zeros([nr,nr]) #dissipation
    # linear term   
    for k in range(nr):
        for i in range(nr):
            tmp = phidd[:,i]
            b_lc1[k,i] = np.dot(tmp.T, phi[:,k]) 
   
    b_lc = [b_lc0,b_lc1]
    return b_lc


###############################################################################
# FSM ROM Routines
###############################################################################

# Jacobian of RHS
def GROMrhsJ(nr, lam, b_c, b_cc, b_l, b_lc, b_nl, a): #f(u)

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
print('Reading FOM snapshots...')
ufom = np.load('./data/ufom_re'+str(int(re))+'.npy')

# mean subtraction
um = np.mean(ufom,1)
u = ufom - um.reshape(-1,1)

#%% POD basis computation for training data
print('Computing POD basis...')
phi = np.zeros((nx+1,nr)) # POD modes     
eigv = np.zeros((ns+1)) #Eigenvalues      
phi, eigv, ric  = POD(u, nr) 
        
#%% Calculating true POD modal coefficients
atrue = np.zeros((ns+1,nr))
print('Computing true POD coefficients...')
atrue = PODproj(u,phi)
#Unifying signs [not necessary]
phi = phi/np.sign(atrue[:,0].reshape([1,-1]))
atrue = atrue/np.sign(atrue[:,0].reshape([-1,1]))

#%% Galerkin projection precomputed coefficients
print('Computing GP coefficients...')
b_c = const_term(nr,um,phi,re)
b_l = lin_term(nr,um,phi,re)
b_nl = nonlin_term(nr,phi,re)
b_cc = closure_const_term(nr,um,phi)
b_lc = closure_lin_term(nr,um,phi,re)

#%% Generate Observations from a twin experiment
sig2 = 0.01
sig = np.sqrt(sig2)
R = sig**2 * np.eye(nr)
Ri = np.linalg.inv(R)

Nz = 10 #number of observations per assimilation window

t_wind = 1 #assimilation window
t_z = np.linspace(0,t_wind,Nz+1)

Nt = int(np.round(t_wind/dt,decimals=0)) #number of timesteps per assimilation window
Ns = int(Nt/freq)   #number of snapshots per assimilation window
Ntz = int(Nt/Nz)       #number of timesteps between observations
Nsz = int(Ntz/freq)    #number of snapshots between observations
ind = np.arange(Nsz,Ns+1,Nsz)
uobs = ufom[:,ind] + np.random.normal(0,sig,ufom[:,ind].shape)
z = PODproj(uobs- um.reshape(-1,1),phi)

#%% FSM eddy viscosity estimation itertations
a0 = np.copy(atrue[:,0])
lam_b = [np.zeros(nr),np.zeros(nr)]
lam = np.copy(lam_b)
max_iter= 100
npp = 2*nr #number of parameters
for jj in range(max_iter):
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
upod = PODrec(atrue,phi) + um.reshape(-1,1)#Reconstruction   
ugp  = PODrec(agp,phi)   + um.reshape(-1,1)#Reconstruction   
ufsm = PODrec(afsm,phi)  + um.reshape(-1,1) #Reconstruction   

def RMSE(ua,ub):
    er = (ua-ub)**2
    er = np.mean(er)
    er = np.sqrt(er)
    return er

# Computing RMSE
lpod, lgp, lfsm = [np.zeros(ns+1) for _ in range(3)]
for i in range(ns+1):  
    lpod[i] = RMSE(ufom[:,i] , upod[:,i])
    lgp[i] = RMSE(ufom[:,i] , ugp[:,i])
    lfsm[i] = RMSE(ufom[:,i] , ufsm[:,i])  

   
#%% Saving results

#create data folder
if os.path.isdir("./results"):
    print('Reults folder already exists')
else: 
    print('Creating results folder')
    os.makedirs("./results")
 
print('Saving results')      
np.savez('./results/full_obs.npz', apod=atrue, agp=agp, afsm=afsm,\
                                   upod=upod, ugp=ugp, ufsm=ufsm,\
                                   lpod=lpod, lgp=lgp, lfsm=lfsm,\
                                   ufom=ufom, uobs=uobs, z=z)
