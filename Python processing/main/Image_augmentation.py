# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:14:02 2020

@author: thuli
"""
import numpy as np
import skimage.io as io
import os
from keras.preprocessing.image import ImageDataGenerator
import skimage.transform as trans
import sys
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\util')
sys.path.append(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Python processing\model')
from OneHotUtil import *
import Augmentor
##########################################
#Image visualization with different color#
##########################################
layer1 = [128,128,128]
layer2 = [128,0,0]
layer3 = [192,192,128]
layer4 = [128,64,128]
layer5 = [60,40,222]
layer5 = [128,128,0]
layer6 = [192,128,128]
layer7 = [64,64,128]
all_the_rest = [0,0,0]
color_dict=np.array([all_the_rest,layer1,layer2,layer3,layer4,layer5,layer6,layer7])
def labelVisualize(mask,save_path,num_class=8,color_dict=color_dict):
    img_out=np.zeros(mask.shape+(3,))
    for i in range(num_class):
        img_out[mask==i,:]=color_dict[i]
    io.imsave(os.path.join(save_path),img_out)
    return img_out /255

def NormalizeData(img,mask,flag_multi_class=True,num_class=8):
    if(flag_multi_class):
        img = img / 255
        new_mask=onehot_initialization(mask)
        mask=new_mask;
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
########################################
def adjustData(img,mask,flag_multi_class,num_class):
    img = img / 255
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    mask=np.round(mask/10)
    weights=mask2weight(mask)
    mask=mask.astype(np.int32)
    new_mask=onehot_initialization(mask)
    mask=new_mask
    size=np.shape(mask)
    mask_flat=np.reshape(mask,(size[0],size[1]*size[2],size[3]))
    mask_flat=mask_flat.astype(np.float64)
    return (img,mask_flat,weights)
    
################################
#Image Augmentation#

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 8,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask,sample_weight = adjustData(img,mask,flag_multi_class,num_class)
        yield(img,mask)
        #yield (img,mask,sample_weight)
######################
def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)