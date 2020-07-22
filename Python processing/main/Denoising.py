# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:54:42 2020

@author: thuli
"""
import skimage.restoration as sr
import numpy as np
import dicom
import glob
import h5py
import os 
import scipy.io as scio
from skimage import exposure
from skimage.io import imsave,imread
from PIL import Image
from medpy.filter.noise import immerkaer
from scipy.io import savemat
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import re
#####################################
#Build helper functions
def atoi (text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]
#####################################
#Connect image folders
root_path="";
filenames=[]
for root, dirnames ,filenames in os.walk("D:\Duke_DME_dataset\ReLayNet-master\ReLayNet-master\ResizedTrainImages\\"):
    filenames.sort(key=natural_keys)
    rootpath=root;
print(filenames)

##################################
images=[]
for filename in filenames :
    filepath =os.path.join(root,filename)
    image= imread(filepath,True)
    images.append(image)
    print(filename)
####################################
#Start the denoising of iamges
denoisedimages=[];
hval=10
alpha=15
beta=1
def denoiseImage(image):
    denoisedimages=[];
    maxvalue=np.max(image)
    newimage=image*(255.0/maxvalue).astype(np.uint8)
    denoised =sr.denoise_nl_means(newimage,multichannel=False,h=hval)
    denoised=denoised -(alpha*beta)
    denoised[denoised<0]=0
    denoised=denoised.astype(np.uint8)
    return denoised;
    
for image in images:
    temp=denoiseImage(image)
    denoisedimages.append(temp)
print(len(denoisedimages))
plt.imshow(denoisedimages[0],cmap="gray")
"""
for item in range(770):
    imsave('/home/iplab/Desktop/DenoisedTrain/denoised_'+str(item+1)+'.png',denoisedimages[item])
"""
