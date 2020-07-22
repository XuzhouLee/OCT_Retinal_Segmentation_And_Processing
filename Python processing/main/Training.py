# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:31:26 2020

@author: thuli
"""
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets,models
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import torchvision.utils
from OneHotUtil import *
#Prepare dataset
#Build a dataset which is suitable for ours application


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
        #Normalize the data
        image=image/255;
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
############################################
#Let's summarize the model and print it out#
############################################   
from ResNet import *  

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(8)
model = model.to(device)

summary(model, input_size=(3, 256,256))


##############
#%%
##########################
#Define the training loop#
##########################
from collections import defaultdict
import torch.nn.functional as F
import time
import copy
from loss_calculation import *

#Define the training functions
def train_model(model, optimizer,scheduler,num_epochs=25) :
    best_model_wts=copy.deepcopy(model.state_dict()) 
    best_loss=1e10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        since=time.time()
        #For each epoch, we have a training phase and a validation phase
        for phase in ['train','val']:
            if phase =='train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("Learning Rate",param_group['lr'])
                #Then set the model to "train" mode
                model.train()
            else:
                #Or set the model to evalutate mode
                model.eval()
            metrics=defaultdict(float)
            epoch_samples=0
            
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                #We need to zero the parameter gradients of the optimzer
                optimizer.zero_grad()
                
                #Forward propagation
                #Only in "train" mode we track the history
                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    loss=calc_loss(outputs,labels,metrics)
                    
                #Then backward propagation
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                
                #Get some statistics
                epoch_samples += inputs.size(0)
            
            print_metrics(metrics,epoch_samples,phase)
            epoch_loss=metrics['loss']/epoch_samples
            
            #Then deep copy the model for the current best model 
            if phase=='val' and epoch_loss<best_loss:
                print("saving best model")
                best_loss=epoch_loss
                best_model_wts=copy.deepcopy(model.state_dict())
        time_elapsed=time.time()-since
        print('{:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model
#%%
import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)    
        self.drop_out1=nn.Dropout2d(0.3)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        x=x.float();
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x=self.drop_out1(x)
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        out=F.softmax(out,dim=0)
        return out
#%%

class SimpleNet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.maxpool=nn.MaxPool2d(2)
        self.maxunpool=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.layer1=convrelu(3,16,3,1)
        self.layer2=convrelu(16,32,3,1)
        self.layer3=convrelu(32,64,3,1)
        self.layer4=convrelu(64,128,3,1)
        
        self.layer5=nn.Dropout2d(0.3)
        self.layer6=convrelu(64+128,64,3,1)
        self.layer7=convrelu(64+32,32,3,1)
        self.layer8=convrelu(32+16,16,3,1)
        
        self.last_conv=nn.Conv2d(16,n_class,1)
    def forward(self,x):
        x=x.float();
        
        conv1=self.layer1(x)
        x=self.maxpool(conv1)
        
        conv2=self.layer2(x)
        x=self.maxpool(conv2)
        
        conv3=self.layer3(x)
        x=self.maxpool(conv3)
        
        conv4=self.layer4(x)
       
        x=self.layer5(conv4)
        
        x=self.maxunpool(x)
        x=torch.cat([x,conv3],dim=1)
        x=self.layer6(x)    
        
        x=self.maxunpool(x)
        x=torch.cat([x,conv2],dim=1)
        x=self.layer7(x)
        
        x=self.maxunpool(x)
        x=torch.cat([x,conv1],dim=1)
        x=self.layer8(x)
        out=self.last_conv(x);
        F.softmax(out,dim=0)
        return out;
       

#%%

#Start to run the trainging 
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
torch.cuda.empty_cache()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class=8;
#model=SimpleNet(8).to(device)
model=UNet(num_class).to(device)
#If we need to freeze the backbone layers
"""
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad=False
"""

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)        
        
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=50)

                
            
        
