# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:01:34 2020

@author: thuli
"""
import numpy as np
from OneHotUtil import *
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets,models
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import torchvision.utils
from OneHotUtil import *
import torch.nn.functional as F
#Prepare dataset
#Build a dataset which is suitable for ours application
####
#Test One-hot encoding
datasets=np.load('OCT_retinal_datasets.npz')
image_sets=np.asarray(datasets['input'])
mask_sets=np.asarray(datasets['mask'])
temp_mask=mask_sets[:,:,10];
temp_mask_onehot=onehot_initialization(temp_mask);
temp_mask_onehot=np.moveaxis(temp_mask_onehot,-1,0);
print(temp_mask.shape)

#%%
#%%
class OCTDataset(Dataset):
    def __init__(self,images,masks,transform=None):
        self.input_images=images;
        self.input_masks=masks;
        
    def __len__(self):
        [m,n,l]=np.shape(self.input_images)
        return l
    
    def __getitem__(self,idx):
        image=self.input_images[:,:,idx]
        #Now our input channel is "1" if we need to use 3 as the input channel, we will need to un-comment the following in line to stack images together
        image=np.dstack((image,image,image))
        image=np.moveaxis(image,-1,0)
        mask=self.input_masks[:,:,idx]
        mask=onehot_initialization(mask)
        mask=np.moveaxis(mask,-1,0);
        return [image,mask]
#%%
#Since we have done the image augmentation during the preprocessing part,now we can directly import the images into our training and validation sets#
#image standard size (512,512);
datasets=np.load('OCT_retinal_datasets.npz')
image_sets=np.asarray(datasets['input'])
mask_sets=np.asarray(datasets['mask'])
#%%
import random
def restofindex(num,train_index):
    output=[];
    for i in range(num):
        if i in train_index:
            continue
        else:
            output.append(i)
    random.shuffle(output)
    return output
train_index=random.sample(range(1000),800)
val_index=restofindex(1000,train_index)

train_set=OCTDataset(image_sets[:,:,train_index],mask_sets[:,:,train_index])
val_set=OCTDataset(mask_sets[:,:,val_index],mask_sets[:,:,val_index])
image_datasets={
    'train':train_set,'val':val_set}
batch_size=8
dataloaders={
    'train':DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0),
    'val':DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)}
#Test the dataset and dataloader that we have been established.

inputs,masks=next(iter(dataloaders['train']))
print(inputs.shape,masks.shape)
#%%
inputs2,masks2=next(iter(dataloaders['train']))
#%%
masks=masks.float()
temp_dice_loss=dice_loss(masks,F.sigmoid(masks))
print(temp_dice_loss.data.cpu().numpy())
#%%
masks2=masks2.float()
temp_dice_loss=dice_loss(masks,F.sigmoid(masks2))
print(temp_dice_loss.data.cpu().numpy())
