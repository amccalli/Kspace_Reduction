#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:37:26 2018

@author: andrewmccallister
"""

#%          test_mrics.m by Tom Goldstein  (tagoldst@math.ucla.edu)
#%   This file is meant to demonstrate how to properly use mrics.m
#%   When this script is run, it will first build a simple test image.  The
#%   method then builds a sampling matrix, R, with entries randomly chosen 
#%   to be 1 or 0.  The compressed sensing data is then computed using the
#%   folrmula F = R.*fft2(image).  Gaussian noisy is added to the CS data.
#%   Finally, the mrics method is used to reconstruct the image form the
#%   sub-sampled K-Space data.
#
#  
#N = 128; % The image will be NxN
#sparsity = .25; % use only 25% on the K-Space data for CS 
#mu = .1;
#lambda = .1;
#gamma = mu/1000;
#
#  % build an image of a square
#image = zeros(N,N);
#image(N/4:3*N/4,N/4:3*N/4)=255;
# 
# % build the sampling matrix, R
#R = rand(N,N);
#R = double(R<sparsity);
#
# % Form the CS data
#
#F = R.*fft2(image)/N;
#
#% Recover the image
#recovered = mrics(R,F, mu, lambda, gamma,10, 4);
#
#% build a figure to display results
#figure;
#subplot(2,2,1);
#imagesc(abs(image)); colormap('gray');
#title('Original');
#subplot(2,2,2);
#imagesc(abs(R)); colormap('gray');
#title('R');
#subplot(2,2,3);
#imagesc(abs(ifft2(F))); colormap('gray');
#title('Set unknown to 0');
#subplot(2,2,4);
#imagesc(abs(recovered)); colormap('gray');
#title('Split Bregman Recovery');
import numpy as np
import Split_Bergman
import matplotlib.pyplot as plt


N=128
sparsity = 0.25
mu = 0.1
Lambda=0.1
gamma = mu/1000
image = np.zeros((N,N))
image[np.int16((N/4)-1):np.int16((3*N/4)),np.int16((N/4)-1):np.int16((3*N/4))]=255
R = np.random.random((N,N))
R = np.squeeze(np.int16([R<sparsity]))
F = np.multiply(R,np.fft.fft2(image)/N)
recovered,test = Split_Bergman.mrics(R,F, mu, Lambda, gamma,10, 4)

plt.figure(1)
plt.subplot(221)
plt.imshow(np.absolute(image),'gray')
#plt.colorbar()

plt.subplot(222)
plt.imshow(np.absolute(R),'gray')
#plt.colorbar()

plt.subplot(223)
plt.imshow(np.absolute(np.fft.ifft2(F)),'gray')
#plt.colorbar()

plt.subplot(224)
plt.imshow(np.absolute(recovered),'gray')
#plt.colorbar()
plt.show()
    
if __name__ == '__main__':
    main()