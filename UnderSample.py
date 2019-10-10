#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:39:18 2018

@author: andrewmccallister
"""

import numpy as np

def NormGaussian2D(x,y,ux,uy,sx,sy):
    return np.exp(-((x-ux)**2/sx**2+(y-uy)**2/sy**2))

def Sample(F,dist,mult=0.5,sparsity=0.5,enc_dir=0):
    if dist =='random':
        R = np.random.random(F.shape)
        R = np.squeeze(np.int16([R<sparsity]))
        return R
    elif dist =='Gaussian_Center':
        R = np.random.random(F.shape)
        X = np.zeros(F.shape)
        il=F.shape[0]
        jl=F.shape[1]
        imid=(F.shape[0]-1)/2
        jmid=(F.shape[1]-1)/2
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                X[i,j] = NormGaussian2D(i,j,imid,jmid,il*mult,jl*mult)
        R=np.squeeze(np.int16([R<X]))
        under=np.sum(R)/R.size
        print('undersampling = {}'.format(under))
        return R
    elif dist =='random_line':
        R = np.random.random(F.shape[enc_dir])
        R = np.int16([R<0.5])
        R = R.T*np.ones(F.shape)
        return R
        
        