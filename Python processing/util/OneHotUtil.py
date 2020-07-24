# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:05:07 2020

@author: thuli
"""
import numpy as np

def onehot_initialization(mask):
    ncols=8;
    out = np.zeros( (mask.size,ncols), dtype=np.int)
    out[np.arange(mask.size),mask.ravel()] = 1
    out.shape = mask.shape + (ncols,)
    return out