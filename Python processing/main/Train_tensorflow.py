# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:54 2020

@author: thuli
"""
##################
#First append the system path to include models and util libraries
import sys
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\util')
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\model')
#%%
#Loading pre-processed data#
import numpy as np 
from OneHotUtil import *
datasets=np.load(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\processed_mat\Datasets.npz')
image_sets=np.asarray(datasets['image_sets'])
mask_sets=np.asarray(datasets['mask_sets'])
weight_sets=np.asarray(datasets['weight_sets'])
mask_onehot_sets=[];
[m,n,l]=np.shape(mask_sets)
################################
#One hot encoding the mask sets#
################################
for i in range(l):
    temp=onehot_initialization(mask_sets[:,:,i])
    mask_onehot_sets.append(temp)
mask_onehot_sets=np.array(mask_onehot_sets);
print(mask_onehot_sets[0])
#%%
######################
#We should do image augmentation#
#########################
#We will see what happens and then come back to add this part

#%%
#generate training sets
train_indices=np.random.choice(l,80,replace=False)
train_image_random=[];
train_mask_random=[];
for i in train_indices:
    train_image_random.append(image_sets[:,:,i])
    train_mask_random.append(mask_onehot_sets[i,:,:,:])
    
test_indices=[x for x in range(l) if x not in train_indices]
test_images=[]
test_masks=[]
for i in test_indices:
    test_images.append(image_sets[:,:,i])
    test_masks.append(mask_onehot_sets[i,:,:,:])
train_image_random=np.array(train_image_random)
train_mask_random=np.array(train_mask_random)
test_images=np.array(test_images)
test_masks=np.array(test_masks)
#convert data type to float to compatible with later calculation
    


    

