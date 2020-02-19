from base.base_model import BaseModel
from models.optimizers import Optimizer
from tensorflow.keras import Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Conv2D
from tensorflow.keras.metrics import *
import numpy as np

class DR_ResNet50(BaseModel):
    def __init__(self, config):
        super(DR_ResNet50, self).__init__(config)
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # ResNet50 as backbone network
        # Load pre-trained ResNet50 without the classifier
        self.resnet50_base = ResNet50(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze ResNet50 parameters
        for layer in self.resnet50_base.layers:
            layer.trainable = trainable
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

        self.output = Dense(self.output_shape, activation='softmax')(self.dropout_4)

        self.model.summary()

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)

class DR_InceptionV3(BaseModel):
    def __init__(self, config):
        super(DR_InceptionV3, self).__init__(config)
        self.optimizer = Optimizer(config)
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

        self.output = Dense(self.output_shape, activation='softmax')(self.dropout_4)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)