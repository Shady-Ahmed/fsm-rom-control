#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the 2D vortex merger problem 

Ref: "Forward sensitivity analysis and mode dependent control for closure modeling of Galerkin systems",
      Computers & Mathematics with Applications, 2023
     by: Shady Ahmed, Omer San

Last checked: Monday, July 10, 2023
@author: Shady Ahmed
contact: shady.ahmed@okstate.edu
"""
#%% Import libraries
import numpy as np
import pyfftw

#%% fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[:-1,:-1],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
           
    #periodicity
    u = np.empty((nx+1,ny+1)) 
    u[:nx,:ny] = ut
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
       
    return u

#%% Compute jacobian using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def arak_jac_per(nx,ny,dx,dy,wf,sf):
    
    w = np.zeros((nx+3,ny+3))
    w[1:nx+2,1:ny+2] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    s = np.zeros((nx+3,ny+3))
    s[1:nx+2,1:ny+2] = sf
    
    # periodic
    s[:,ny+1] = s[:,1]
    s[nx+1,:] = s[1,:]
    s[nx+1,ny+1] = s[1,1]
    
    # ghost points
    s[:,0] = s[:,ny]
    s[:,ny+2] = s[:,2]
    s[0,:] = s[nx,:]
    s[nx+2,:] = s[2,:]
    
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    jac = (j1+j2+j3)*hh
    
    return jac
    
#%% compute laplacian
def laplacian_per(nx,ny,dx,dy,wf):
    w = np.zeros((nx+3,ny+3))
    w[1:nx+2,1:ny+2] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
        
    return lap    
