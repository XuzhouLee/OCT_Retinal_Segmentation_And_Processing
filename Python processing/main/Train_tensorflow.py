# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:54 2020

@author: thuli
"""
import numpy as np 
from OneHotUtil import *
datasets=np.load('OCT_retinal_datasets.npz')
image_sets=np.asarray(datasets['input'])
mask_sets=np.asarray(datasets['mask'])
mask_onehot_sets=[];
[m,n,l]=np.shape(mask_sets)
for i in range(l):
    temp=onehot_initialization(mask_sets[:,:,i])
    mask_onehot_sets.append(temp)
mask_onehot_sets=np.array(mask_onehot_sets);
print(mask_onehot_sets[0])
#%%
##############################
#Preprocessing for the images#
import skimage.restoration as sr
import numpy as np
import dicom
import glob
import h5py
import os 
import scipy.io as scio
from skimage import exposure
from skimage.io import imsave,imread
from PIL import Image
from medpy.filter.noise import immerkaer
from scipy.io import savemat
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import re
#####################################

denoisedimages=[];
hval=10
alpha=15
beta=1
def denoiseImage(image):
    denoisedimages=[];
    maxvalue=np.max(image)
    newimage=image*(255.0/maxvalue).astype(np.uint8)
    denoised =sr.denoise_nl_means(newimage,multichannel=False,h=hval)
    denoised=denoised -(alpha*beta)
    denoised[denoised<0]=0
    denoised=denoised.astype(np.uint8)
    return denoised;
for i in range(l):
    temp=denoiseImage(image_sets[:,:,i])
    denoisedimages.append(temp)
print(len(denoisedimages))
plt.imshow(denoisedimages[0],cmap="gray")
#%%
#creat a sample weights methods
weighted_images = []
for i in range(l):
    image = mask_sets[:,:,i]
    weighted_image = np.zeros((m,n))
    for j in range(m):
        for k in range(n):
             if(round(image[j][k]/20)==1):
                w2 = 11.459
             elif(image[j][k] == 2):
                w2 = 5.63
             elif(image[j][k]== 3):
                w2 = 11.007 
             elif(image[j][k] == 4):
                w2 = 14.368 
             elif(image[j][k]== 5):
                w2 = 3.336 
             elif(image[j][k]== 6):
                w2 = 13.647 
             elif(image[j][k]== 7):
                w2 = 16.978 
             else:
                w2 = 0
             if(j!=0 and j!=215):
                if(image[j+1][k]-image[j-1][k]>0 and w2<>0):
                    w1 = 15 
                   # count = count +1
                    count[int(image[j-1][k])] = count[int(image[j-1][k])] + 1 
                else:
                    w1 = 0
             else:
                w1 = 0
             w = 1 + w1 + w2
             weighted_image[j][k] = w
    weighted_images.append(weighted_image)



    

