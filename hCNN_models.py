'''
"Multiscale Hierarchical Convolutional Networks"
J.-H. Jacobsen, E. Oyallon, S. Mallat, A.W.M. Smeulders
arxiv.org/

Code by J.-H. Jacobsen in collaboration with E. Oyallon
Informatics Institute, University of Amsterdam & Data Team, ENS
'''

import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, merge, Activation, Dropout, Dense, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from hCNN_utils import hconv2d, hconv3d, hconv4d, reduce_dim_mean, reduce_mean

from rme.datasets import cifar10

import os
import numpy as np

import tensorflow as tf
  
## Blocks
      
def first_block(inputs, n_out, ksize_sp, ksize_ch1, ksize_ch2, l2_reg):
    y = hconv2d(inputs,n_out,l2_reg,ksize=(ksize_sp,ksize_sp))
    y = hconv3d(y,n_out,l2_reg,ksize=(ksize_sp,ksize_sp,ksize_ch2),st=(1,1,2))
    y = hconv4d(y,n_out,l2_reg,ksize=(ksize_sp,ksize_sp,ksize_ch1,ksize_ch2),st=(1,1,2,2))
    y = reduce_dim_mean(y,3)
    return y

def block(inputs, n_out, ksize_sp, ksize_ch1, ksize_ch2, l2_reg, subsample=False):
    if subsample:
        y = hconv4d(inputs,n_out,l2_reg,ksize=(ksize_sp,ksize_sp,ksize_ch1,ksize_ch2),st=(2,2,2,2))
    else:
        y = hconv4d(inputs,n_out,l2_reg,ksize=(ksize_sp,ksize_sp,ksize_ch1,ksize_ch2),st=(1,1,2,2))
    y = reduce_dim_mean(y,3)
    y = hconv4d(y,n_out,l2_reg,ksize=(ksize_sp,ksize_sp,ksize_ch1,ksize_ch2),st=(1,1,2,2))
    y = reduce_dim_mean(y,3)
    return y

def hCNN_cifar10(v_j, v_1, v_2, v_J=10, l2_reg=3e-4):
    x = 32
    inputs = Input((x, x, 3))

    y = first_block(inputs, n_out, 3, v_1, v_2, l2_reg)
    y = block(y, v_j, 3, v_1, v_2, l2_reg, subsample=True)
    y = block(y, v_j, 3, v_1, v_2, l2_reg, subsample=False)
    y = block(y, v_j, 3, v_1, v_2, l2_reg, subsample=True)
    y = block(y, v_j, 3, v_1, v_2, l2_reg, subsample=False)

    y = Lambda(reduce_mean, output_shape=(x/4,v_j/2,v_j),arguments={'axis':1})(y)
    y = Lambda(reduce_mean, output_shape=(v_j/2,v_j),arguments={'axis':1})(y)
    y = Lambda(reduce_mean, output_shape=(v_j,),arguments={'axis':1})(y)
    y = Dense(v_J)(y)
    y = Activation('softmax')(y)
    model = Model(input=inputs, output=y)
    return model
