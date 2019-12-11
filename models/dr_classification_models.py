from base.base_model import BaseModel
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

class DR_ResNet50(BaseModel):
    def __init__(self, config):
        super(DR_ResNet50, self).__init__(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=(512,512,3))
        # Load pre-trained ResNet50 without the classifier
        self.resnet50_base = ResNet50(include_top=False, input_tensor=self.visible, weights='imagenet')
        # Add custom classifier
        self.average_pooling = GlobalAveragePooling2D()(self.resnet50_base.output)
        self.hidden = Dense(1024, activation='relu')(self.average_pooling)
        self.output = Dense(5, activation='softmax')(self.hidden)
        # Define new model
        self.model = Model(inputs=self.visible, outputs=self.output)

        
        # Freeze ResNet50 parameters
        self.resnet50_base.trainable = False

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