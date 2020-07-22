# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:04:35 2020

@author: thuli
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
########################################
#STEP 0: Import dataset and organize it#
########################################

#We first start with manual labeled mask_1 as the ground truth to
#train the networks
images=loadmat('images.mat')
masks1=loadmat('mask1.mat')

image_sets=np.array(images['images'])
mask_sets=np.array(masks1['masks1'])

#Free the memory, we just need the two sets for training,validating and testing
images=0
masks1=0
image_sets[image_sets==255]=0
#Demonstrate a random pair of image and corresponding masks
[m,n,l]=np.shape(image_sets);
demo_num=random.randrange(0,l,1)
demo_image=image_sets[:,:,demo_num]
demo_mask=mask_sets[:,:,demo_num]
plt.subplot(121)
plt.imshow(demo_image)
plt.subplot(122)
plt.imshow(demo_mask)
plt.suptitle('OCT images #'+str(demo_num)+' Segmentation mask #'+str(demo_num))
plt.show()
#Saving them to the input folder and mask folder
from skimage.io import imsave,imread
import os
data_dir=os.path.join(".\data_2")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    input_dir=os.path.join(".\data_2\input")
    os.mkdir(input_dir)
    mask_dir=os.path.join(".\data_2\mask")
    os.mkdir(mask_dir)
else:
    print("The data folder has already built!")
    
#%%
#######################################
def array_to_images(array_3d,path):
    [m,n,l]=np.shape(array_3d)
    file_type=".jpg"
    for i in range(0,l):
        temp_image=array_3d[:,:,i]
        file_name=path+str(i)+file_type
        imsave(file_name,temp_image)
    print("Arrays have been saved in folders as JPEG files!")
def array_to_2dcsv(array_3d,path):
    [m,n,l]=np.shape(array_3d)
    file_type=".csv"
    for i in range(0,l):
        temp_image=array_3d[:,:,i]
        file_name=path+str(i)+file_type
        np.savetxt(file_name,temp_image,delimiter=',')
    print("Arrays have been saved in folders as CSV files!")
input_path=".\data_2\input\I"
array_to_images(image_sets,input_path)
mask_path=".\data_2\mask\I"
mask_sets=mask_sets*20;
array_to_images(mask_sets,mask_path)

#%%
# We plan to utilize the "Augmentor Library to resize our images and ground truth
# We plan to do data augmentation to make make 550 images in total to train the network
import Augmentor

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as trans
p=Augmentor.Pipeline("D:\Duke_DME_dataset\data_2\input")
p.ground_truth("D:\Duke_DME_dataset\data_2\mask")
p.rotate(probability=0.8,max_left_rotation=10,max_right_rotation=10)
p.flip_top_bottom(probability=0.1)
p.random_distortion(probability=0.3,grid_width=2,grid_height=2,magnitude = 1)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.9)
p.sample(1000)
#The augmented data was stored in the output folder#
#Then we can generate a dataset#
