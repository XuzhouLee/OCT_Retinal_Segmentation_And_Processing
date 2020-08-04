# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:38:59 2020

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
datasets=np.load(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\processed_mat\Datasets.npz')
image_sets=np.asarray(datasets['image_sets'])
mask_sets=np.asarray(datasets['mask_sets'])
weight_sets=np.asarray(datasets['weight_sets'])
mask_onehot_sets=[];
[m,n,l]=np.shape(mask_sets)
resize_image=np.zeros((256,256,l))
resize_mask=np.zeros((256,256,l))
resize_weights=np.zeros((256,256,l))
#%%
for i in range(l):
    temp=image_sets[i,:,:]
    temp2=mask_sets[:,:,i]
    new=cv2.resize(temp,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
    resize_image[:,:,i]=new
    new2=cv2.resize(temp2,dsize=(256,256),interpolation=cv2.INTER_NEAREST)
    resize_mask[:,:,i]=new2
    temp3=weight_sets[:,:,i]
    new3=cv2.resize(temp3,dsize=(256,256),interpolation=cv2.INTER_NEAREST)
    resize_weights[:,:,i]=new3;
image_sets=resize_image;
mask_sets=resize_mask.astype(int);
weight_sets=resize_weights;
print(image_sets.shape)
print(mask_sets.shape)
[m,n,l]=np.shape(mask_sets)
#%%
################################
#One hot encoding the mask sets#
################################
for i in range(l):
    temp=onehot_initialization(mask_sets[:,:,i])
    mask_onehot_sets.append(temp)
mask_onehot_sets=np.array(mask_onehot_sets);
#print(mask_onehot_sets[0])

#%%
#generate training sets
import cv2
train_num=80
train_indices=np.random.choice(l,train_num,replace=False)
train_image_random=[];
train_mask_random=[];
train_weight_random=[];
for i in train_indices:
    train_image_random.append(image_sets[:,:,i])
    train_mask_random.append(mask_onehot_sets[i,:,:,:])
    train_weight_random.append(weight_sets[:,:,i])
    
test_indices=[x for x in range(l) if x not in train_indices]
test_images=[];
test_masks=[];
test_weights=[];
for i in test_indices:
    test_images.append(image_sets[:,:,i])
    test_masks.append(mask_onehot_sets[i,:,:,:])
    test_weights.append(weight_sets[:,:,i])
train_image_random=np.array(train_image_random)
train_mask_random=np.array(train_mask_random)
train_weight_random=np.array(train_weight_random)
test_weights=np.array(test_weights)
test_images=np.array(test_images)
test_masks=np.array(test_masks)
train_image_random=train_image_random[:,:,:, np.newaxis]
test_images=test_images[:,:,:,np.newaxis]
train_weight_random=train_weight_random[:,:,:,np.newaxis]
test_weights=test_weights[:,:,:,np.newaxis]
#convert data type to float to compatible with later calculation
train_image_random=train_image_random.astype('float32')
train_mask_random=train_mask_random.astype('float32')
train_weight_random=train_weight_random.astype('float32')
test_weights=test_weights.astype('float32')
test_images=test_images.astype('float32')
test_masks=test_masks.astype('float32')

