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
######################
#We should do image augmentation#
#########################
#We will see what happens and then come back to add this part
"""
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode="grayscale",mask_color_mode="grayscale",
                   image_save_prefix="image",mask_save_prefix="mask",flag_multi_class=False,num_class=8,save_to_dir=None,target_size=(256,256),seed=1):
    image_datagen=ImageData
"""



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
#%%
####################
#Start the training#
####################
import keras
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda 
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
from keras.layers import Reshape

from keras import backend as K
from keras import regularizers, optimizers

import scipy.io as scio
import numpy as np    
import os
import matplotlib.pyplot as plt
import math
import re
#from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub
from RelayNet import *
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
data_shape=m*n;
weight_decay_rate=0.0001
model=RelayNet(weight_decay=weight_decay_rate);

"""
weight_decay=0.0001
inputs=Input(shape=(m,n,1))
#Build the model
###########################################################################
#NEED TO BE DONE: SUMMARIZE THE MODEL AND THEN SAVE IT IN THE MODEL FOLDER#
###########################################################################
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
"""
#%%
sample_weights=np.reshape(train_weight_random,(train_num,data_shape))

#Calcualte the weights of labels
train_lables=np.reshape(train_mask_random,(train_num,data_shape,8))
test_lables=np.reshape(test_masks,(l-train_num,data_shape,8))
count=np.zeros(8)
[m2,n2,l2]=np.shape(train_lables)
for i in range(m2):
    for j in range(n2):
        for k in range(l2):
            if (train_lables[i,j,k]==1):
                count[k]+=1
median=np.median(count)
scale=np.zeros(8)
for i in range(8):
    scale[i]=(median)/count[i];
weights=scale/scale[0]

#%%
#Test with RelayNet
from loss_calculation import *
#optimiser=optimizers.Adam(lr=0.01)
#model.compile(optimizer=optimiser,loss=customized_loss,metrics=['accuracy',dice_coef],sample_weight_mode='temporal')
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
csv_logger=CSVLogger('Relaynet_sample_weights_no_augmentation.csv')
model_checkpoint=ModelCheckpoint("Relaynet_sample_weights_denoised_lr_e2_testing_bs_20.hdf5",monitor='val_loss',verbose=1,save_best_only=True)
model.fit(train_image_random,train_lables,batch_size=10,epochs=200,validation_data=(test_images,test_lables),sample_weight=sample_weights,callbacks=[lr_reducer,csv_logger,model_checkpoint])
#%%
from UNet import *
model=UNet();
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
csv_logger=CSVLogger('UNET_no_kernel_regulizer_no_augmentation.csv')
model_checkpoint=ModelCheckpoint("UNET_no_kernel_no_augmentation.hdf5",monitor='val_loss',verbose=1,save_best_only=True)
model.fit(train_image_random,train_lables,batch_size=10,epochs=200,validation_data=(test_images,test_lables),sample_weight=sample_weights,callbacks=[lr_reducer,csv_logger,model_checkpoint])
#%%
model.load_weights(r"Relaynet_sample_weights_denoised_lr_e2_testing_bs_20.hdf5")
test_image=np.squeeze(train_image_random[28])
plt.imshow(test_image)
#%%
from OneHotUtil import *
test_image=test_image.reshape((1,256,256,1))
prediction=model.predict(test_image)
prediction=np.squeeze(prediction,axis=0)
prediction=np.reshape(prediction,(256,256,8))
prediction=np.round(prediction)
predict_image=onehot2int(prediction)
plt.imshow(predict_image)

