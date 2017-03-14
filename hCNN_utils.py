'''
"Multiscale Hierarchical Convolutional Networks"
J.-H. Jacobsen, E. Oyallon, S. Mallat, A.W.M. Smeulders
https://arxiv.org/abs/1703.04140

Code by J.-H. Jacobsen in collaboration with E. Oyallon
Informatics Institute, University of Amsterdam & Data Team, ENS
'''

import keras
import keras.backend as K
from keras.callbacks import Callback
from keras.layers import Input, merge, Activation, Dropout, Dense, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf

########
## Various utility functions
########

def reduce_sum(x):
    return K.sum(x,axis=3)

def reduce_mean(x,axis):
    return tf.reduce_mean(x,axis=axis)

def expand_dim(x,axis):
    return K.expand_dims(x,dim=axis)

def reshape(x,shape):
    return tf.reshape(x,shape)

def transpose(x,perm):
    return tf.transpose(x,perm=perm)

def reduce_output_shape(input_shape):
    shape = list(input_shape)
    l = len(shape)
    shape = shape[:(l-1)]
    return tuple(shape)

def expand_dim_output_shape(input_shape):
    shape = list(input_shape)
    shape.append(1)
    return tuple(shape)

def reduce_dim_mean(x,dim):
    shape = list(x.get_shape())
    del shape[dim]
    y = Lambda(reduce_mean, output_shape=tuple(shape),arguments={'axis':dim})(x)
    return y

########
## hCNN Convolution Blocks
########

def hconv2d(x,n3,l2_reg,ksize=(3,3),st=(1,1)):
    # In shape Nn1n2ch
    # Out shape Nn1n2n3
    y = Convolution2D(n3,ksize[0],ksize[1],subsample=(st[0],st[1]),
                     border_mode='same', init='normal',
                     W_regularizer=l2(l2_reg), bias=False)(x)
    y = keras.layers.advanced_activations.ELU(alpha=1.0)(BatchNormalization(axis=3)(y))
    return y

def hconv3d(x,n4,l2_reg,ksize=(3,3,3),st=(1,1,1)):
    # In shape Nn1n2n3
    # Out shape Nn1n2n3n4
    y = Lambda(expand_dim, output_shape=expand_dim_output_shape,arguments={'axis':4})(x)
    y = Convolution3D(n4,ksize[0],ksize[1],ksize[2], subsample=(st[0],st[1],st[2]),
                         border_mode='same', init='normal',
                         W_regularizer=l2(l2_reg), bias=False)(y)
    y = keras.layers.advanced_activations.ELU(alpha=1.0)(BatchNormalization(axis=4)(y))
    return y

def hconv4d(x,n5,l2_reg,ksize=(3,3,3,3),st=(1,1,1,1)):
    # In shape Nn1n2n3n4
    # Out shape Nn1n2n3n4n5
    n1 = int(x.get_shape()[1])
    n2 = int(x.get_shape()[2])
    n3 = int(x.get_shape()[3])
    n4 = int(x.get_shape()[4])

    y = Lambda(transpose,output_shape=tuple((n3,n4,n1,n2)),arguments={'perm':[0,3,4,1,2]})(x)
    y = Lambda(reshape,output_shape=tuple((n1,n2,1)),arguments={'shape':(-1,n1,n2,1)})(y)
    y = Convolution2D(n5*2,ksize[0],ksize[1],subsample=(st[0],st[1]),
                         border_mode='same', init='normal',
                         W_regularizer=l2(l2_reg), bias=False)(y)
    y = keras.layers.advanced_activations.ELU(alpha=1.0)(BatchNormalization(axis=3)(y))
    y = Lambda(reshape,output_shape=tuple((n3,n4,n1/st[0],n2/st[1],n5*2)),arguments={'shape':(-1,n3,n4,n1/st[0],n2/st[1],n5*2)})(y)
    y = Lambda(transpose,output_shape=tuple((n1,n2,n3,n4,n5*2)),arguments={'perm':[0,3,4,1,2,5]})(y)
    y = Lambda(reshape,output_shape=tuple((n3,n4,n5*2)),arguments={'shape':(-1,n3,n4,n5*2)})(y)
    y = Convolution2D(n5,ksize[2],ksize[3],subsample=(st[2],st[3]),
                         border_mode='same', init='normal',
                         W_regularizer=l2(l2_reg), bias=False)(y)
    y = keras.layers.advanced_activations.ELU(alpha=1.0)(BatchNormalization(axis=3)(y))
    y = Lambda(reshape,output_shape=tuple((n1/st[0],n2/st[1],n3/st[2],n4/st[3],n5)),arguments={'shape':(-1,n1/st[0],n2/st[1],n3/st[2],n4/st[3],n5)})(y)
    return y

########
## Util functions by Roberto Mest 
## https://github.com/robertomest/
########

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))

########
## Util function to train data parallel across GPUs
## adapted from: https://github.com/kuza55/keras-extras/
########

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1] // parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1] // parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)
