# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:05:07 2020

@author: thuli
"""
import numpy as np

def one_hot_vector(n,total):
    n=n-1;
    n=np.max((0,n))
    n=np.min((n,total-1))
    output=np.zeros(total)
    output[n]=1
    return output

def mask_to_onehot(mask):
    mask=mask//20;
    mask=mask-1;
    mask=np.maximum(mask,0)
    mask=np.minimum(mask,7)
    [m,n]=np.shape(mask);
    l=8;
    output=np.zeros((m,n,l))
    
    return output 
def onehot_initialization(mask):
    mask=mask//20;
    mask=mask-1;
    mask=np.maximum(mask,0)
    a=np.minimum(mask,7)
    ncols=8;
    out = np.zeros( (a.size,ncols), dtype=np.int)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out