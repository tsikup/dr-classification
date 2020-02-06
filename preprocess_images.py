from data_preprocessing.kaggle_eyepacs_preprocess import KagglePreprocess
from utils.utils import get_args
from utils.config import process_config
from pathlib import Path
import glob
import os
import cv2 as cv

import multiprocessing as mp

def preprocess_and_write(dest, image_path, class_num):
    # Create KagglePreprocess instance
    kaggle_preprocessor = KagglePreprocess()
    # read image in color mode
    image = cv.imread(image_path, 1)
    # preprocess image
    new_image = kaggle_preprocessor.preprocess(image, numOfBlurred=3)
    # determite new filepath
    new_image_filepath = os.path.join(dest, str(class_num), os.path.basename(image_path))
    # save new image
    cv.imwrite(new_image_filepath, new_image)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)
        
    # Configure parallel pool for faster data preprocessing
    pool = mp.Pool(mp.cpu_count()-2)
    
    # Classes to search (e.g. if there are 5 folders for each class named by the number of the class, then classes = [0,1,2,3,4])
    classes = config.dataset.classes
    # Directory to read training set
    train_dir = config.dataset.train
    # Directory to read testing set
    test_dir = config.dataset.test
    # Directory to write training set
    train_dest = config.dataset.train_dest
    Path(train_dest).mkdir(parents=True, exist_ok=False)
    # Directory to write testing set
    test_dest = config.dataset.test_dest
    Path(test_dest).mkdir(parents=True, exist_ok=False)
    
    # Iterate through classes
    for class_num in classes:
        # Preprocess training set
        # Create destination folder for each class
        Path(os.path.join(train_dest,str(class_num))).mkdir(parents=True, exist_ok=False)
        # Determine training images' paths
        train_images = glob.glob(os.path.join(train_dir, str(class_num), '*.jpeg'))
        # Iterate through the images
        [pool.apply(preprocess_and_write, args=(train_dest, image_path, class_num)) for image_path in train_images]
            
        # Preprocess testing set
        # Create destination folder for each class
        Path(os.path.join(test_dest,str(class_num))).mkdir(parents=True, exist_ok=False)
        # Determine testing images' paths
        test_images = glob.glob(os.path.join(test_dir, str(class_num), '*.jpeg'))
        # Iterate through the images
        [pool.apply(preprocess_and_write, args=(test_dest, image_path, class_num)) for image_path in test_images]          
        

if __name__ == '__main__':
    main()