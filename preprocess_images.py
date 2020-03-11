from data_preprocessing.kaggle_eyepacs_preprocess import KagglePreprocess
from utils.utils import get_args
from utils.config import process_config
from pathlib import Path
import glob
import os
import cv2 as cv
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import math

def preprocess_and_write(dest, image_path, class_num, numOfBlurred=1):
    # Create KagglePreprocess instance
    kaggle_preprocessor = KagglePreprocess()
    # read image in color mode
    image = cv.imread(image_path, 1)
    # preprocess image
    new_image = kaggle_preprocessor.preprocess(image, numOfBlurred=numOfBlurred)
    # determite new filepath
    new_image_filepath = os.path.join(dest, str(class_num), os.path.basename(image_path))
    # save new image
    cv.imwrite(new_image_filepath, new_image)

def chunked_worker(input_tuple):
    dest = input_tuple[0]
    img_file_list = input_tuple[1]
    class_num = input_tuple[2]
    numOfBlurred = input_tuple[3]
    [preprocess_and_write(dest, img, class_num, numOfBlurred) for img in img_file_list]
    return 0

def call_multi_executor(images, write_dir, class_num, max_workers=4, numOfBlurred=1):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunksize = int(math.ceil(len(images) / float(max_workers)))
        for i in range(max_workers):
            images_chuck = images[(chunksize * i) : (chunksize * (i + 1))]
            executor_input_tuple = (write_dir, images_chuck, class_num, numOfBlurred)
            executor.submit(chunked_worker, executor_input_tuple)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, False, False)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)
    
    # Classes to search (e.g. if there are 5 folders for each class named by the number of the class, then classes = [0,1,2,3,4])
    classes = config.dataset.classes
    
    # Directory to read training set
    train_dir = config.dataset.train
    
    # Directory to read testing set
    test_dir = config.dataset.test
    
    # How many blurred images
    numOfBlurred = config.preprocessing.numOfBlurred
    
    # Max multiprocessing workers
    workers = config.multiprocess.workers
    
    # Directory to write training set
    train_dest = config.dataset.train_dest
    Path(train_dest).mkdir(parents=True, exist_ok=False)
    
    # Directory to write testing set
    test_dest = config.dataset.test_dest
    Path(test_dest).mkdir(parents=True, exist_ok=False)
    
    # ####################### #
    # Iterate through classes #
    # ####################### #
    for class_num in classes:
        # ####################### #
        # Preprocess training set #
        # ####################### #
        # Create destination folder for each class
        Path(os.path.join(train_dest,str(class_num))).mkdir(parents=True, exist_ok=False)
        # Determine training images' paths
        train_images = glob.glob(os.path.join(train_dir, str(class_num), '*.jpeg'))
        # Preprocess images on several processors
        call_multi_executor(train_images, train_dest, class_num, max_workers=workers, numOfBlurred=numOfBlurred)
            
        # ###################### #
        # Preprocess testing set #
        # ###################### #
        # Create destination folder for each class
        Path(os.path.join(test_dest,str(class_num))).mkdir(parents=True, exist_ok=False)
        # Determine testing images' paths
        test_images = glob.glob(os.path.join(test_dir, str(class_num), '*.jpeg'))
        # Iterate through the images
        call_multi_executor(test_images, test_dest, class_num, max_workers=workers, numOfBlurred=numOfBlurred)
        

if __name__ == '__main__':
    main()