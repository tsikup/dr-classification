from tensorflow.keras.layers import *

class Inception_A():
    def __init__(self, filters=64):
        self.filters = filters
        
    # Build the Inception-A block
    def __call__(self, visible):
        
        # Tower 1
        tower_1 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu')(tower_1)
        tower_1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu')(tower_1)
        
        # Tower 2
        tower_2 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_2 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu')(tower_2)
        
        # Tower 3
        tower_3 = AveragePooling2D(pool_size=3, strides=(1,1), padding='same')(visible)
        tower_3 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(tower_3)
        
        # Tower 4
        tower_4 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        
        output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
        
        return output


class Inception_B():
    def __init__(self, filters=64):
        self.filters = filters
        
    # Build the Inception-A block
    def __call__(self, visible):
        
        # Tower 1
        tower_1 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_1 = Conv2D(self.filters, kernel_size=(7,1), padding='same', activation='relu')(tower_1)
        tower_1 = Conv2D(self.filters, kernel_size=(1,7), padding='same', activation='relu')(tower_1)
        tower_1 = Conv2D(self.filters, kernel_size=(7,1), padding='same', activation='relu')(tower_1)
        tower_1 = Conv2D(self.filters, kernel_size=(1,7), padding='same', activation='relu')(tower_1)
        
        # Tower 2
        tower_2 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_2 = Conv2D(self.filters, kernel_size=(1,7), padding='same', activation='relu')(tower_2)
        tower_2 = Conv2D(self.filters, kernel_size=(7,1), padding='same', activation='relu')(tower_2)
        
        # Tower 3
        tower_3 = AveragePooling2D(pool_size=3, strides=(1,1), padding='same')(visible)
        tower_3 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(tower_3)
        
        # Tower 4
        tower_4 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        
        output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
        
        return output
        

class Inception_C():
    def __init__(self, filters=64):
        self.filters = filters
        
    # Build the Inception-A block
    def __call__(self, visible):
        
        # Tower 1
        tower_1 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu')(tower_1)
        tower_11 = Conv2D(self.filters, kernel_size=(3,1), padding='same', activation='relu')(tower_1)
        tower_12 = Conv2D(self.filters, kernel_size=(3,1), padding='same', activation='relu')(tower_1)
        
        # Tower 2
        tower_2 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        tower_21 = Conv2D(self.filters, kernel_size=(1,3), padding='same', activation='relu')(tower_2)
        tower_22 = Conv2D(self.filters, kernel_size=(3,1), padding='same', activation='relu')(tower_2)
        
        # Tower 3
        tower_3 = AveragePooling2D(pool_size=3, strides=(1,1), padding='same')(visible)
        tower_3 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(tower_3)
        
        # Tower 4
        tower_4 = Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(visible)
        
        output = concatenate([tower_11, tower_12, tower_21, tower_22, tower_3, tower_4], axis = 3)
        
        return output
    

