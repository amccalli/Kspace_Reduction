#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:23:21 2018

@author: andrewmccallister
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import nibabel as nib

def RangeOnes(arr):
    # go through the array from left to right
    for i in range(0, arr.size):
        # if true, then return index
        if (arr[0] == 0 and arr[i] == 1):
                return i,arr.size-1,(arr.size-1)-i
        elif (arr[1] == 1 and arr[i] == 0):
                return 0,i-1,arr.size-i
    # 1's are not present in the array
    return -1

def POCS_recon2D(kspace,threshold):
    [kx1,kx2,kxC]=RangeOnes(np.int16(np.any(kspace,1)))
    [ky1,ky2,kyC]=RangeOnes(np.int16(np.any(kspace,0)))
    #M_p is the part of kspace that was aquired with zeros elsewhere
    M_p=kspace
    mask=np.int16(kspace==0)
    plt.figure(0)
    plt.imshow(np.abs(M_p),vmin=0,vmax=1,cmap="gray")
    #M_l is the low frequency part of kspace with zeros elsewhere
    M_l=np.zeros(kspace.shape,dtype=complex)
    if (kx1 == 0 and ky1 == 0 ):
        M_l[kxC:kx2,kyC:ky2]=kspace[kxC:kx2,kyC:ky2]
        x1_l=kxC
        y1_l=kyC
        x2_l=kx2
        y2_l=ky2
        print(x1_l,x2_l,y1_l,y2_l)
    elif(kx1 != 0 and ky1 != 0 ):
        M_l[kx1:kxC,ky1:kyC]=kspace[kx1:kxC,ky1:kyC]
        x1_l=kx1
        y1_l=ky1
        x2_l=kxC
        y2_l=kyC
        print(x1_l,x2_l,y1_l,y2_l)
        plt.figure(1)
        plt.imshow(np.abs(M_l),vmin=0,vmax=1,cmap="gray")
    elif (kx1 == 0 and ky1 != 0):
        M_l[kxC:kx2,ky1:kyC]=kspace[kxC:kx2,ky1:kyC]
        x1_l=kxC
        y1_l=ky1
        x2_l=kx2
        y2_l=kyC
        print(x1_l,x2_l,y1_l,y2_l)
    else:
        M_l[kx1:kxC,kyC:ky2]=kspace[kx1:kxC,kyC:ky2]
        x1_l=kx1
        y1_l=kyC
        x2_l=kxC
        y2_l=ky2
        print(x1_l,x2_l,y1_l,y2_l)
    #Inverse Fourier transform low frequency term and use to estimate the phase
    m_l=np.fft.ifft2(M_l)
    plt.figure(2)
    plt.imshow(np.abs(m_l),cmap="gray")
    phase_l=np.angle(m_l)
    plt.figure(3)
    plt.imshow(phase_l,cmap="gray")
#    while(NMSE>threshold):
#        m_p=np.fft.ifft2(M_p)
#        #Make image guess
#        m_r=np.abs(m_p)*np.exp(1j*phase_l)
#        #Convert back to k-space to compare to original image
#        M_r=np.fft.ifft2(m_r)
#        #Get NMSE for center of kspace
#        NMSE=np.sqrt((1/(2*(x2_l-x1_l)*(y2_l-y1_l)))*np.sum((np.abs(M_r[x1_l:x2_l,y1_l:y2_l]-Mp[x1_l:x2_l,y1_l:y2_l])/np.abs(Mp[x1_l:x2_l,y1_l:y2_l]))**2))      
#        M_p=M_r
#    return M_p,NMSE
    i=0
    NMSE=np.zeros(15)
    for i in range(0,15):
        m_p=np.fft.ifft2(M_p)
        #plt.figure(3*i+4)
        #plt.imshow(np.abs(m_p))
        #Make image guess
        m_r=np.abs(m_p)*np.exp(1j*phase_l)
        #plt.figure(3*i+5)
        #plt.imshow(np.abs(m_r))
        #Convert back to k-space to compare to original image
        M_r=np.fft.fft2(m_r)
        #Get NMSE for center of kspace
        NMSE[i]=np.sqrt((1/(2*(x2_l-x1_l)*(y2_l-y1_l)))*np.sum((np.abs(M_r[x1_l:x2_l,y1_l:y2_l]-M_p[x1_l:x2_l,y1_l:y2_l])/np.abs(M_p[x1_l:x2_l,y1_l:y2_l]))**2))      
        #plt.figure(3*i+6)
        #plt.imshow(np.abs(M_r-M_p))
        M_p=M_r*mask+kspace
    plt.figure(4)
    plt.imshow(np.abs(M_r),vmin=0,vmax=1,cmap="gray")
    plt.figure(5)
    plt.imshow(np.abs(np.fft.ifft2(kspace)),cmap="gray")
    plt.figure(6)
    plt.imshow(np.abs(np.fft.ifft2(M_r)),cmap="gray")

    return M_r,NMSE
    
img = nib.load('/Volumes/DREW_USB/Statistics/Brain/t1_icbm_normal_1mm_pn1_rf20.mnc')
data = img.get_data()
plt.figure(7)
plt.imshow(np.flip(np.flip(data[91,:,:],0),1),cmap="gray")
kdata=np.fft.fftshift(np.fft.ifft2(data[91,:,:]))
plt.figure(8)
plt.imshow(np.abs(kdata),vmin=0,vmax=1,cmap="gray")
kdata_cut=np.zeros((kdata.shape),dtype=complex)
kdata_cut[80:,65:]=kdata[80:,65:]
#plt.imshow(np.abs(kdata_cut),vmin=0,vmax=0.2)
M_p,NMSE=POCS_recon2D(kdata_cut,10)
