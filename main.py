from data_loader.dr_data_loader import DRDataLoader
from models.dr_classification_models import *
from trainers.dr_trainer import DRModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.gpus import set_gpus

import tensorflow as tf
import os

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Set number of gpu instances to be used
    # set_gpus(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    print('Physical devices: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    print('Logical devices: {}'.format(len(tf.config.experimental.list_logical_devices('GPU'))))
    
    # create the experiments dirs
    print('Creating directories: {}, {}'.format(config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir))
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator. Classes 0,1 and 2,3,4 are respectively merged together.')
    data_loader = DRDataLoader(config)
    train_data, val_data = data_loader.get_train_data(classes=config.dataset.classes)

    print('Create the model.')
    model = DR_InceptionV3(config)

    print('Create the trainer')
    trainer = DRModelTrainer(model.model, (train_data, val_data), config, data_loader.get_train_data())

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
