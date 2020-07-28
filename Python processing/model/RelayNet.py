# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:15:57 2020

@author: thuli
"""
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import sys
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\util')
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\model')
from loss_calculation import *
from keras import regularizers, optimizers
def RelayNet(pretrained_weight=None,input_size=(256,256,1),weight_decay=0.0001):
    m=input_size[0];
    n=input_size[1];
    data_shape=m*n;
    inputs=Input(shape=input_size)
    L1=Conv2D(64,kernel_size=(3,3),padding="same",kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    L2=BatchNormalization()(L1)
    L2=Activation('relu')(L2)
    L3 = MaxPooling2D(pool_size=(2,2))(L2)
    L4 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L3)
    L5 = BatchNormalization()(L4)
    L5 = Activation('relu')(L5)
    L6 = MaxPooling2D(pool_size=(2,2))(L5)
    L7 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L6)
    L8 = BatchNormalization()(L7)
    L8 = Activation('relu')(L8)
    L9 = MaxPooling2D(pool_size=(2,2))(L8)
    L10 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L9)
    L11 = BatchNormalization()(L10)
    L11 = Activation('relu')(L11)
    L12 = UpSampling2D(size = (2,2))(L11)
    L13 = Concatenate(axis = 3)([L8,L12])
    L14 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L13)
    L15 = BatchNormalization()(L14)
    L15 = Activation('relu')(L15)
    L16 = UpSampling2D(size= (2,2))(L15)
    L17 = Concatenate(axis = 3)([L16,L5])
    L18 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L17)
    L19 = BatchNormalization()(L18)
    L19 = Activation('relu')(L19)
    L20 = UpSampling2D(size=(2,2),name = "Layer19")(L19)
    L21 = Concatenate(axis=3)([L20,L2])
    L22 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L21)
    L23 = BatchNormalization()(L22)
    L23 = Activation('relu')(L23)
    L24 = Conv2D(8,kernel_size=(1,1),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L23)
    L = Reshape((data_shape,8),input_shape = (m,n,8))(L24)
    L = Activation('softmax')(L)
    model = Model(inputs = inputs, outputs = L)
    model.summary()
    optimiser=optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimiser,loss=customized_loss,metrics=['accuracy',dice_coef],sample_weight_mode='temporal')
    return model;