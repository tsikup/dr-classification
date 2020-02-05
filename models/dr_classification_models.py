from base.base_model import BaseModel
from tensorflow.keras import Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Conv2D

import time

class DR_ResNet50(BaseModel):
    def __init__(self, config):
        super(DR_ResNet50, self).__init__(config)
        self.input_shape = tuple(self.config.model.input_shape)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # ResNet50 as backbone network
        # Load pre-trained ResNet50 without the classifier
        self.resnet50_base = ResNet50(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze ResNet50 parameters
        for layer in self.resnet50_base.layers:
            layer.trainable = True
        # self.resnet50_base.trainable = True
        
        ### Add custom classifier
        ## W. Zhang, J. Zhong, S. Yang, Z. Gao, J. Hu, Y. Chen and Z. Yi, 
        ## "Automated identification and grading system of diabetic retinopathy using deep neural networks," 
        ## Knowledge-Based Systems, vol. 175, pp. 12-25, 1 7 2019.
        # GAP
        self.average_pooling = GlobalAveragePooling2D()(self.resnet50_base.output)
        # Block 1
        self.hidden_1 = Dense(1024, activation='relu')(self.average_pooling)
        self.dropout_1 = Dropout(0.25)(self.hidden_1)
        # Block 2
        self.hidden_2 = Dense(512, activation='relu')(self.dropout_1)
        self.dropout_2 = Dropout(0.5)(self.hidden_2)
        # Block 3
        self.hidden_3 = Dense(256, activation='relu')(self.dropout_2)
        self.dropout_3 = Dropout(0.5)(self.hidden_3)
        # Block 4
        self.hidden_4 = Dense(128, activation='relu')(self.dropout_3)
        self.dropout_4 = Dropout(0.5)(self.hidden_4)

        self.output = Dense(5, activation='softmax')(self.dropout_4)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.config.model.optimizer,
              metrics = ['accuracy'])

class DR_InceptionV3(BaseModel):
    def __init__(self, config):
        super(DR_InceptionV3, self).__init__(config)
        self.input_shape = tuple(self.config.model.input_shape)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # InceptionV3 as backbone network
        # Load pre-trained InceptionV3 without the classifier
        self.inceptionv3_base = InceptionV3(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze InceptionV3 parameters
        for layer in self.inceptionv3_base.layers:
            layer.trainable = True
        # self.inceptionv3_base.trainable = True
        
        ### Add custom classifier
        ## W. Zhang, J. Zhong, S. Yang, Z. Gao, J. Hu, Y. Chen and Z. Yi, 
        ## "Automated identification and grading system of diabetic retinopathy using deep neural networks," 
        ## Knowledge-Based Systems, vol. 175, pp. 12-25, 1 7 2019.
        # GAP
        self.average_pooling = GlobalAveragePooling2D()(self.inceptionv3_base.output)
        # Block 1
        self.hidden_1 = Dense(1024, activation='relu')(self.average_pooling)
        self.dropout_1 = Dropout(0.25)(self.hidden_1)
        # Block 2
        self.hidden_2 = Dense(512, activation='relu')(self.dropout_1)
        self.dropout_2 = Dropout(0.5)(self.hidden_2)
        # Block 3
        self.hidden_3 = Dense(256, activation='relu')(self.dropout_2)
        self.dropout_3 = Dropout(0.5)(self.hidden_3)
        # Block 4
        self.hidden_4 = Dense(128, activation='relu')(self.dropout_3)
        self.dropout_4 = Dropout(0.5)(self.hidden_4)

        self.output = Dense(5, activation='softmax')(self.dropout_4)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.config.model.optimizer,
              metrics = ['accuracy'])


        # conv_1      =   Conv2D(32, kernel_size=(7,7), strides = 2, padding = 'valid', activation='relu')(visible_0)
        # maxpool_2   =   MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
        # conv_3      =   Conv2D(32, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(maxpool_2)
        # conv_4      =   Conv2D(32, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_3)
        # maxpool_5   =   MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4)
        # conv_6      =   Conv2D(64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(maxpool_5)
        # conv_7      =   Conv2D(64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_6)
        # maxpool_8   =   MaxPooling2D(pool_size=(2, 2), strides=2)(conv_7)
        # conv_9      =   Conv2D(128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(maxpool_8)
        # conv_10     =   Conv2D(128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_9)
        # conv_11     =   Conv2D(128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_10)
        # conv_12     =   Conv2D(128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_11)
        # maxpool_13  =   MaxPooling2D(pool_size=(2, 2), strides=2)(conv_12)
        # conv_14     =   Conv2D(256, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(maxpool_13)
        # conv_15     =   Conv2D(256, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_14)
        # conv_16     =   Conv2D(256, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_15)
        # conv_17     =   Conv2D(256, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')(conv_16)
        # maxpool_18  =   MaxPooling2D(pool_size=(2, 2), strides=2)(conv_17)
        # dropout_19  =   Dropout(rate = 0.5)(maxpool_18)
        # maxout_20   =   MaxoutDense(output_dim=512)(dropout_19)
        # dropout_21  =   Dropout(rate = 0.5)(maxout_20)
        # maxout_22   =   MaxoutDense(output_dim=512)