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
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\main')
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
sample_weights=np.reshape(train_weight_random,(train_num,data_shape))
train_lables=np.reshape(train_mask_random,(train_num,data_shape,8))
test_lables=np.reshape(test_masks,(l-train_num,data_shape,8))


#%%
#Training with RelayNet and original datasets
from loss_calculation import *
model=RelayNet(weight_decay=weight_decay_rate);
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
csv_logger=CSVLogger('Relaynet_sample_weights_no_augmentation.csv')
model_checkpoint=ModelCheckpoint("Relaynet_sample_weights_denoised_lr_e2_testing_bs_20.hdf5",monitor='val_loss',verbose=1,save_best_only=True)
model.fit(train_image_random,train_lables,batch_size=10,epochs=200,validation_data=(test_images,test_lables),sample_weight=sample_weights,callbacks=[lr_reducer,csv_logger,model_checkpoint])
#Training with UNET and original datasets
#%%
from UNet import *
model=unet();
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
csv_logger=CSVLogger('UNET_original_datasets.csv')
model_checkpoint=ModelCheckpoint("UNET_original_datasets.hdf5",monitor='val_loss',verbose=1,save_best_only=True)
model.fit(train_image_random,train_lables,batch_size=4,epochs=30,validation_data=(test_images,test_lables),callbacks=[lr_reducer,csv_logger,model_checkpoint])
#%%
from Image_augmentation import *
#Let's see the predict result of the two models
model1=RelayNet(weight_decay=weight_decay_rate);
model1.load_weights(r"Relaynet_sample_weights_denoised_lr_e2_testing_bs_20.hdf5")
test_image=np.squeeze(train_image_random[0])
plt.imshow(test_image)
#%%
from OneHotUtil import *
test_image=test_image.reshape((1,256,256,1))
prediction=model1.predict(test_image)
prediction=np.squeeze(prediction,axis=0)
prediction=np.reshape(prediction,(256,256,8))
prediction=np.round(prediction)
predict_image=onehot2int(prediction)
predict_labels=labelVisualize(predict_image,'relay_net._prediction.png')
plt.imshow(predict_labels)
#%%
model2=unet();
model2.load_weights(r"UNET_original_datasets.hdf5");
prediction2=model2.predict(test_image)
prediction2=np.squeeze(prediction2,axis=0)
prediction2=np.reshape(prediction2,(256,256,8))
prediction2=np.round(prediction2)
predict_image2=onehot2int(prediction2)
predict_lable2=labelVisualize(predict_image2,'unet_prediction.png')
plt.imshow(predict_lable2)
#%%
####################################
#Start with UNET sturcutrure with a more specific defined training sets generator
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\main')
from Image_augmentation import *
from UNet import *
data_gen_args=dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
                   shear_range=0.05,
                   zoom_range=0.05,
                   horizontal_flip=True,
                   fill_mode='nearest')
myGene=trainGenerator(4,r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\original_datasets','image','mask',data_gen_args,num_class=8,save_to_dir=None)
model=unet();
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
model_checkpoint=ModelCheckpoint('unet_with_augmentation.hdf5',monitor='loss',verbose=1,save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=20,epochs=50,callbacks=[lr_reducer,model_checkpoint])
#%%
#Let's try the data generator with RelayNet
weight_decay_rate=0.0001
model=RelayNet(weight_decay=weight_decay_rate);
lr_reducer=ReduceLROnPlateau(factor=0.5,cooldown=0,patience=6,min_lr=0.5e-6)
csv_logger=CSVLogger('Relaynet_sample_weights_with_augmentation.csv')
model_checkpoint=ModelCheckpoint("Relaynet_with_augmentation.hdf5",monitor='val_loss',verbose=1,save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=20,epochs=50,callbacks=[lr_reducer,csv_logger,model_checkpoint])
#%%
model2=unet();
model2.load_weights(r"unet_with_augmentation.hdf5");
prediction2=model2.predict(test_image)
prediction2=np.squeeze(prediction2,axis=0)
prediction2=np.reshape(prediction2,(256,256,8))
prediction2=np.round(prediction2)
predict_image2=onehot2int(prediction2)
predict_lable2=labelVisualize(predict_image2,'unet_prediction.png')
plt.imshow(predict_lable2)
