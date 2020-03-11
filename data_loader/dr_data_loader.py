from base.base_data_loader import BaseDataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings

class DRDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DRDataLoader, self).__init__(config)
        
        self.train_datagen = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True,
            rotation_range = 15,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            validation_split = 0.2)
        
        self.test_datagen = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True,)

        self.train_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'training',
            color_mode="rgb")

        self.validation_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'validation',
            color_mode="rgb")
        
        self.test_generator = self.test_datagen.flow_from_directory(
            directory = self.config.dataset.test,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            color_mode="rgb")

    def get_train_data(self, classes=None):
        self.config.dataset.batch.train_size = len(self.train_generator)
        self.config.dataset.batch.val_size = len(self.validation_generator)
        if(classes is not None):
            return self.new_gen(self.train_generator, classes), self.new_gen(self.validation_generator, classes)
        else:
            return self.train_generator, self.validation_generator

    def get_test_data(self, classes=None):
        self.config.dataset.batch.test_size = len(self.test_generator)
        if(classes is not None):
            return self.new_gen(self.test_generator, classes)
        else:
            return self.test_generator
    
    # Define generator with new classes
    # For example, if classes=[0,0,1,1,1], the 0,1 DR classes and the 2,3,4 DR classes will be respectively merged together.
    def new_gen(self, generator, classes):
        if(len(classes) == 5 and all(isinstance(x, int) for x in classes)):
            for data, labels in generator:
                try: # Merge Categorical Labels
                    assert labels.shape[1] == 5 # Ensure that the dimensions of the labels matrix is [None, 5]
                    assert self.config.dataset.class_mode == "categorical" # Ensure that the class mode is categorical instead of sparse
                    labels_new = [[1, 0] if np.argmax(label) < classes.index(next(filter(lambda x: x==1, classes))) else [0, 1] for label in labels]
                    labels = np.asarray(labels_new)
                except: # Merge Sparse Labels
                    classes = np.array(classes)
                    labels = classes[labels.astype(int)]
                yield data, labels
        else:
            warnings.warn("'classes' must be a list of 5 integers, but yours is a list of size {} and type {}. Returning the original generator".format(len(classes), type(classes)))
            return generator
        

