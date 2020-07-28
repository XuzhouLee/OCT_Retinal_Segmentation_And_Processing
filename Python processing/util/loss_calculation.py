# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:11:38 2020

@author: thuli
"""


from keras import backend as K
import time
import copy
import tensorflow as tf

#We need define the dice loss function for this specific application
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true,y_pred,smooth=1):
    return 1-dice_coef(y_true,y_pred,smooth=smooth)

#Define the loss calculation of our training 
    """
def calc_loss(y_true,y_pred,metrics,sample_weights,bce_weight=0.5):
    
    bce = tf.keras.losses.BinaryCrossentropy()
    bec_loss=bce(y_true,y_pred,sample_weight=sample_weights).numpy()
    dice=dice_loss(y,target)
    loss=bce*bce_weight+dice * (1-bce_weight)
    return loss
    """
###############################################################
def customized_loss(y_true,y_pred):
    return (1*K.categorical_crossentropy(y_true, y_pred))+(0.5*dice_loss(y_true, y_pred))