'''
"Multiscale Hierarchical Convolutional Networks"
J.-H. Jacobsen, E. Oyallon, S. Mallat, A.W.M. Smeulders
arxiv.org/

Code by J.-H. Jacobsen in collaboration with E. Oyallon
Informatics Institute, University of Amsterdam & Data Team, ENS, Paris
'''

import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils

from hCNN_models import hCNN_cifar10
from hCNN_utils import make_parallel
import numpy as np
  
def norm_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])
    data_set -= mean
    data_set /= std
    return data_set

if __name__ == '__main__':
    print 'Training Hierarchical CNN'
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    v_j = 16
    v_1 = 7
    v_2 = 11
    batch = 50

    model = hCNN_cifar10(v_j, v_1, v_2)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print 'Model compiled.'

    (X_train,y_train), (X_test,y_test) = cifar10.load_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    train_set['data'] = norm_data(X_train)
    test_set['data'] = norm_data(X_test)
    train_set['labels'] = np_utils.to_categorical(y_train)
    test_set['labels'] = np_utils.to_categorical(y_test)
    print 'Data Loaded and Preprocessed.'

    nb_epoch = 240
    callbacks = []
    steps = [40,80,120,160,200,220]
    schedule = Step(steps, [0.125,0.0625,0.0313,0.0156,0.0078,0.0039,0.00017], verbose=1)
    callbacks.append(schedule)
    schedule = None
    name = './results/hCNN_weights'
    data_gen = ImageDataGenerator(horizontal_flip=True,
                                  width_shift_range=0.125,
                                  height_shift_range=0.125,
                                  fill_mode='constant')
    data_iter = data_gen.flow(train_set['data'], train_set['labels'],
                              batch_size=batch, shuffle=True)
    print 'Starting fit_generator.'
    model.fit_generator(data_iter,
                        samples_per_epoch=train_set['data'].shape[0],
                        nb_epoch=(nb_epoch - offset),
                        verbose = 1,
                        callbacks=callbacks)
    score = model.evaluate(test_set['data'], test_set['labels'], verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights(name+'.h5')
