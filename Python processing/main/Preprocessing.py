# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:45:06 2020

@author: thuli
"""
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from scipy.io import loadmat
import skimage.restoration as sr
###############################################
#Save all the images as npz file#
path = r"C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\processed_mat";
image_sets=loadmat(path+r"\\resized_image.mat")['resized_images'];
mask_sets=loadmat(path+r"\\resized_mask1.mat")['masks1'];
weight_sets=loadmat(path+r"\\weighted_mask.mat")['weighted_samples'];
image_sets[image_sets>250]=0;
#%%
###############################################
#Denoise the input images
[m,n,l]=np.shape(image_sets);
#Define a function for single image denoising
def denoise_image(image,val=10,alpha=15,beta=1):
    max_val=np.max(image);
    new_image=image*((255.0)/max_val).astype(np.uint8);
    #Denoise image with non-local means
    denoised=sr.denoise_nl_means(new_image,multichannel=False,h=val)
    denoised=denoised-(alpha*beta);
    denoised[denoised<0]=0
    denoised=denoised.astype(np.uint8)
    return denoised;

denoised_sets=[];

for i in range(l):
    image=image_sets[:,:,i]
    denoised_image=denoise_image(image);
    denoised_sets.append(denoised_image);

denoised_sets=np.array(denoised_sets)
##############################################
#Saved them as npz#
np.savez(path+r'\\Datasets.npz',image_sets=denoised_sets,mask_sets=mask_sets,weight_sets=weight_sets)
