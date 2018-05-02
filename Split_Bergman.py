#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:30:16 2018

@author: andrewmccallister

Modifying matlab script from Tom Goldstein to python

mrics.m  by Tom Goldstein (TomGoldstein1@gmail.com)
%     This file contains methods for performing compressed sensing
%  recontructions of images from k-space data using the Split Bregman 
%  method.  
%     To use the method, simply add this "m" file to your current directory, 
%  and then call the following method:
%   
%              u = mrics(R,F, mu, lambda, gamma, nInner, nOuter);
%   
%  The inputs to this function are described below:
%  
%  R - This is a matrix which determines which elements of K-Space
%           are known.  Take R(m,n)=0 if Fourier mode (m,n) is unknown,
%           and R(m,n)=1 if the corresponding mode is known.
%  F - This is the K-Space (Fourier) data.  In other words, F is the 
%           Fourier transform of the image you wish to recover.  If a
%           Fourier mode is known, then it should have a non-zero value.
%           If a Fourier mode is unknown, then simply set the corresponding
%           entry in this matrix to zero.  If you have set the values
%           in this matrix properly, then you should have (R.*F==F).
%  mu- The parameter on the fidelity term in the Split Bregman method.  The
%           best choice will depend on how the data is scaled.
% lambda - The coefficient of the constraint term in the Split Bregman
%      model. For most problems, I suggest using lambda=mu.
% gamma - This is a regularization parameter.  I suggest that you take
%      gamma = mu/100.
% nInner - This determines how many "inner" loops the Split Bregman method
%      performs (i.e. loop to enforce the constraint term).  I suggest
%      using nInner = 30 to be safe.  This will usually guarantee good
%      convergence, but will make things a bit slow.  You may find that you
%      can get away with nInner = 5-10
% nOuter - The number of outer (fidelity term) Bregman Iterations.  This
%      parameter depends on how noisy your data is, but I find that
%      nOuter=5 is usually about right.
"""
import numpy as np
#function u = mrics(R,f, mu, lambda, gamma, nInner, nBreg)
#    [rows,cols] = size(f);
def mrics(R,f, mu, Lambda, gamma, nInner, nBreg):
    rows,cols=f.shape      
#         % Reserve memory for the auxillary variables
#    f0 = f;
#    u = zeros(rows,cols);
#    x = zeros(rows,cols);
#    y = zeros(rows,cols);
#    bx = zeros(rows,cols);
#    by = zeros(rows,cols);     
    f0 = f
    u = np.zeros(f.shape,dtype='complex64')
    x = np.zeros(f.shape,dtype='complex64')
    y = np.zeros(f.shape,dtype='complex64')
    bx = np.zeros(f.shape,dtype='complex64')
    by = np.zeros(f.shape,dtype='complex64')
    test = np.zeros((rows,cols,nInner*nBreg),dtype='complex64')
#     % Build Kernels
#    scale = sqrt(rows*cols);
#    murf = ifft2(mu*(conj(R).*f))*scale;
    scale=np.sqrt(rows*cols)
    murf = np.fft.ifft2(np.multiply(mu,np.multiply(np.conjugate(R),f)))*scale
#    uker = zeros(rows,cols);
#    uker(1,1) = 4;uker(1,2)=-1;uker(2,1)=-1;uker(rows,1)=-1;uker(1,cols)=-1;
#    uker = mu*(conj(R).*R)+lambda*fft2(uker)+gamma;
    uker = np.zeros(f.shape,dtype='complex64')
    uker[0,0] = 4
    uker[0,1] = -1
    uker[1,0] = -1
    uker[rows-1,0] = -1
    uker[0,cols-1] = -1
    uker = np.multiply(mu,np.multiply(np.conjugate(R),R))+Lambda*np.fft.fft2(uker)+gamma;
    counter=0
#
#    %  Do the reconstruction
#    for outer = 1:nBreg;
    for outer in range(nBreg):
#        for inner = 1:nInner;
        for outer in range(nInner):
#             % update u   
#            rhs = murf+lambda*Dxt(x-bx)+lambda*Dyt(y-by)+gamma*u;
            #print(x.shape,bx.shape)
            rhs = murf+Lambda*Dxt(x-bx)+Lambda*Dyt(y-by)+gamma*u
#            u = ifft2(fft2(rhs)./uker);
            u = np.fft.ifft2(np.divide(np.fft.fft2(rhs),uker))
            test[:,:,counter]=u
#            % update x and y
#            dx = Dx(u);
#            dy  =Dy(u);
#            [x,y] = shrink2( dx+bx, dy+by,1/lambda);
            dx = Dx(u)
            dy = Dy(u)
            x,y= shrink2(dx+bx,dy+by,1/Lambda)
#            % update bregman parameters
#            bx = bx+dx-x;
#            by = by+dy-y;
            bx = bx+dx-x
            by = by+dy-y
            counter=counter+1
#        end
#        f = f+f0-R.*fft2(u)/scale;
#        murf = ifft2(mu*R.*f)*scale;
        f = f+f0-np.multiply(R,np.fft.fft2(u))/scale
        murf = np.fft.ifft2(np.multiply(mu*R,f))*scale
#    end
    return u,test
#  
#return;
#
#
#function d = Dx(u)
#[rows,cols] = size(u); 
#d = zeros(rows,cols);
#d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
#d(:,1) = u(:,1)-u(:,cols);
def Dx(u):
    rows,cols = u.shape
    d = np.zeros(u.shape, dtype='complex64')
    d[:,1:cols] = u[:,1:cols]-u[:,0:cols-1]
    d[:,0] = u[:,0]-u[:,cols-2]
    return d
#return
#
#function d = Dxt(u)
#[rows,cols] = size(u); 
#d = zeros(rows,cols);
#d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
#d(:,cols) = u(:,cols)-u(:,1);
#return
def Dxt(u):
    rows,cols = u.shape
    d = np.zeros(u.shape, dtype='complex64')
    d[:,0:cols-1] = u[:,0:cols-1]-u[:,1:cols]
    d[:,cols-1] = u[:,cols-1]-u[:,0]
    return d
#function d = Dy(u)
#[rows,cols] = size(u); 
#d = zeros(rows,cols);
#d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
#d(1,:) = u(1,:)-u(rows,:);
#return
def Dy(u):
    rows,cols = u.shape
    d = np.zeros(u.shape, dtype='complex64')
    d[1:rows,:] = u[1:rows,:]-u[0:rows-1,:]
    d[0,:] = u[0,:]-u[rows-1,:]
    return d
#function d = Dyt(u)
#[rows,cols] = size(u); 
#d = zeros(rows,cols);
#d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
#d(rows,:) = u(rows,:)-u(1,:);
#return
def Dyt(u):
    rows,cols = u.shape
    d = np.zeros(u.shape, dtype='complex64')
    d[0:rows-1,:] = u[0:rows-1,:]-u[1:rows,:]
    d[rows-1,:] = u[rows-1,:]-u[0,:]
    return d

#function [xs,ys] = shrink2(x,y,lambda)
#
#s = sqrt(x.*conj(x)+y.*conj(y));
#ss = s-lambda;
#ss = ss.*(ss>0);
#
#s = s+(s<lambda);
#ss = ss./s;
#
#xs = ss.*x;
#ys = ss.*y;
#
#return;
def shrink2(x,y,Lambda):
    s = np.sqrt(np.multiply(x,np.conjugate(x))+np.multiply(y,np.conjugate(y)))
    ss=s-Lambda
    ss = np.squeeze(np.multiply(ss,[ss>0]))
    s = np.squeeze(s+[s<Lambda])
    ss = np.divide(ss,s)
    xs = np.multiply(ss,x)
    ys = np.multiply(ss,y)
    return xs,ys
    


    
