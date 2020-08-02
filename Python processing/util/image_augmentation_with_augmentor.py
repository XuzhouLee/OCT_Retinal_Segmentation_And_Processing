# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 20:05:58 2020

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
p=Augmentor.Pipeline(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\image')
p.ground_truth(r'C:\Users\thuli\OneDrive - Umich\Desktop\OCT retinal segmenation and processing\Data\mask')
p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.3,percentage_area=0.85)
p.sample(500)
