# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:05:07 2020

@author: thuli
"""
import numpy as np
import skimage.io as io
import os
import copy
def onehot_initialization(mask):
    ncols=8;
    out = np.zeros( (mask.size,ncols), dtype=np.int)
    out[np.arange(mask.size),mask.ravel()] = 1
    out.shape = mask.shape + (ncols,)
    return out
def onehot2int(image):
    [m,n,l]=image.shape;
    output=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            index = np.argmax(image[i,j])
            output[i,j]=index
    return output

def mask2weight(mask):
    #please make sure the input mask is before onehotinitialization#
    [l,m,n]=np.shape(mask)
    output=copy.deepcopy(mask)
    output=output.astype(np.float64)
    output[output==0]=0
    output[output==1]=11.459
    output[output==2]=5.63
    output[output==3]=11.007
    output[output==4]=14.368 
    output[output==5]=3.336
    output[output==6]=13.647
    output[output==7]=16.978
    for i in range(l):
        for j in range(m):
            for k in range(n):
                if (j!=0 and j!=m-1):
                    if (mask[i,j+1,k]-mask[i,j,k]>0 and output[i,j,k]!=0):
                        output[i,j,k] +=15
    output=output+1;
    output=np.reshape(output,(l,m*n))
    return output
    
    

    
        
    