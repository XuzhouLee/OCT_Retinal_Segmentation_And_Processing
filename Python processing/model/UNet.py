# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:52:25 2020

@author: thuli
An implementation of UNET model with keras
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
from keras import regularizers, optimizers
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\util')
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\model')
from loss_calculation import *
##############################################
"""
def UNet(pretrained_weights=None,input_size=(256,256,1),weight_decay=0.00001):
    m=input_size[0]
    n=input_size[1]
    data_shape=m*n;
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv4)
    ###################################################
    #We have a dropout layer here to avoid overfitting#
    ###################################################
    drop4 = Dropout(0.01)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv5)
    ###################################################
    #We have a dropout layer here to avoid overfitting#
    ###################################################
    drop5 = Dropout(0.01)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv9)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv9)
    conv10 = Conv2D(8, 1, activation = 'sigmoid')(conv9)
    L = Reshape((data_shape,8),input_shape = (m,n,8))(conv10)
    L2=Activation('softmax')(L);
    model = Model(input = inputs, output = L2)
    model.summary();
    #################
    #We use Adam optimizer here with a initial lr=0.01 and our customized loss fuction (combination of binary categorical entropy loss and dice coeff)
    #And the 
    model.compile(optimizer = Adam(lr = 1e-2), loss = customized_loss,metrics=['accuracy',dice_coef],sample_weight_mode='temporal')

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
"""
def unet(pretrained_weights=None,input_size=(256,256,1),weight_decay=0.0001):
    size1=input_size
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv9)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(weight_decay))(conv9)
    
    conv10 = Reshape((size1[0]*size1[1],8),input_shape = (size1[0],size1[1],8))(conv9)
    L2=Activation('softmax')(conv10);
    model = Model(input = inputs, output = L2)
    model.compile(optimizer = Adam(lr = 1e-2), loss = customized_loss ,sample_weight_mode="temporal", metrics=['accuracy',dice_coef])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model


