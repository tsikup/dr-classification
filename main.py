from data_loader.dr_data_loader import DRDataLoader
from models.dr_classification_models import *
from trainers.dr_trainer import DRModelTrainer
from testers.dr_tester import DRModelTester
from utils.config import process_config
from utils.utils import get_args
from utils.gpus import set_gpus

import tensorflow as tf
import os

def train(config):
    print('Create the data generator. Classes 0,1 and 2,3,4 are respectively merged together.')
    data_loader = DRDataLoader(config)
    train_data, val_data = data_loader.get_train_data(classes=config.dataset.classes)
    test_data = data_loader.get_test_data(classes=config.dataset.classes)

    print('Create the model.')
    model = DR_PreTrainedModel(config)

    print('Create the trainer.')
    trainer = DRModelTrainer(model.model, (train_data, val_data), config)

    print('Start training the model.')
    trainer.train()
    
    print('Create the tester.')
    tester = DRModelTester(model.model, test_data, config)

    print('Test the model.')
    tester.test()
    
def evaluate(config):    
    print('Create the data generator. Classes 0,1 and 2,3,4 are respectively merged together.')
    data_loader = DRDataLoader(config)
    test_data = data_loader.get_test_data(classes=config.dataset.classes)

    print('Create the model.')
    model = DR_PreTrainedModel(config)
    
    print('Loading checkpoint\'s weights')
    model.load(config.tester.checkpoint_path)
    
    print('Create the tester.')
    tester = DRModelTester(model.model, test_data, config)
    
    print('Test the model.')
    tester.test()

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, dirs=True, config_copy=True)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)

    # Set number of gpu instances to be used
    # set_gpus(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices.gpu.id
    
    print('Physical GPU devices: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    print('Logical GPU devices: {}'.format(len(tf.config.experimental.list_logical_devices('GPU'))))
    
    if(config.mode == "train"):
        train(config)
    elif(config.mode == "eval"):
        evaluate(config)

if __name__ == '__main__':
    main()
