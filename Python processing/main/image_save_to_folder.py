# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:14:02 2020

@author: thuli
"""
"""
Created on Wed Jul 29 16:05:33 2020

@author: thuli
"""
import sys
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\util')
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\model')
#%%
#Loading pre-processed data#
import matplotlib.pyplot as plt
import numpy as np 
from OneHotUtil import *
import cv2
import os
from skimage.io import imsave,imread

datasets=np.load(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\processed_mat\Datasets.npz')
image_sets=np.asarray(datasets['image_sets'])
mask_sets=np.asarray(datasets['mask_sets'])
weight_sets=np.asarray(datasets['weight_sets'])
mask_onehot_sets=[];
[m,n,l]=np.shape(mask_sets)
resize_image=np.zeros((256,256,l))
resize_mask=np.zeros((256,256,l))
resize_weights=np.zeros((256,256,l))
image_path=r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\image'
mask_path=r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\mask'
weight_path=r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\mask'
prefix_image='image'
prefix_mask='image'
prefix_weight='weight'
#%%
for i in range(l):
    temp=image_sets[i,:,:].astype(np.uint8)
    temp2=mask_sets[:,:,i].astype(np.uint8)
    new=cv2.resize(temp,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
    imsave(os.path.join(image_path,prefix_image+str(i)+'.png'),new)
    resize_image[:,:,i]=new
    new2=cv2.resize(temp2,dsize=(256,256),interpolation=cv2.INTER_NEAREST)
    imsave(os.path.join(mask_path,prefix_mask+str(i)+'.png'),new2*10)
    resize_mask[:,:,i]=new2
    temp3=weight_sets[:,:,i]
    new3=cv2.resize(temp3,dsize=(256,256),interpolation=cv2.INTER_NEAREST)
    resize_weights[:,:,i]=new3;
    #imsave(os.path.join(weight_path,prefix_weight+str(i)+'.png'),new3)
image_sets=resize_image;
mask_sets=resize_mask.astype(int);
weight_sets=resize_weights;
print(image_sets.shape)
print(mask_sets.shape)
[m,n,l]=np.shape(mask_sets)
