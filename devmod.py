from keras import Input, layers, initializers, models
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer, MaxPooling2D, Flatten
import keras.backend as K
import numpy as np

class devmod:
    def __init__(self, input, filter):
        super().__init__()
    
        input_shape = (64, 64, 3)

    def model(self, shape, dtype=None):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(kernel_size=(3, 3), kernel_initializer=self.initialize_kernel, activation='relu'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def initialize_kernel(self, shape, dtype=None):
        customFilter = customFilter.reshape((self.filterSize, self.filterSize, 1, 1))
        assert customFilter.shape == shape
        return K.variable(customFilter, dtype='float32')
        
