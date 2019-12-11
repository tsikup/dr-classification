from base.base_data_loader import BaseDataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DRDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DRDataLoader, self).__init__(config)
        
        self.train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            validation_split = 0.2)
        
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = 'sparse',
            shuffle = True,
            subset = 'training',
            color_mode="rgb")

        self.validation_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = 'sparse',
            shuffle = True,
            subset = 'validation',
            color_mode="rgb")
        
        self.test_generator = self.test_datagen.flow_from_directory(
            self.config.dataset.test,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = 'sparse',
            shuffle = True,
            color_mode="rgb")

    def get_train_data(self):
        return self.train_generator, self.validation_generator

    def get_test_data(self):
        return self.test_generator
