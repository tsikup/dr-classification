# This script moves images from folders train/ test/
# to train/grades/0,1,2,3,4 and test/grades/0,1,2,3,4 respectively

from PIL import Image
import csv
import os
import pathlib
import pandas as pd
import numpy as np

'''
Create the subfolders with the given labels from the data frame
'''
def create_folders(dataframe, parent_folder):
    categories = np.unique(dataframe.level)
    for i in categories:
        folder = parent_folder + str(i)
        try:
            pathlib.Path(folder).mkdir(parents=True, exist_ok=False)
        except Exception as e:
            print(e)
            continue

'''
function to move the images to their corresponding folder
'''
def move_files(dataframe, parent_folder, destination_folder):
    try:
        for i, row in dataframe.iterrows():
                filename = r'{}{}.jpeg'.format(parent_folder, row.image)
                new_filename = r'{}{}/{}.jpeg'.format(destination_folder, str(row.level), str(row.image))
                if not os.path.exists(new_filename):
                    try:
                        os.rename(filename, new_filename)
                    except IOError as e:
                        print(e)
                        continue
                else:
                    continue
    except Exception as e:
        print(e)
        exit(0)


def main():
    # Define parent folder and folders for training/testing set
    folder = "/media/hdd/tsiknakisn/SeeFar/Kaggle/EyePACS/"
    train_read_folder = folder + 'train_pre/train/'
    train_destination_folder = folder + 'all_data/train/'
    test_read_folder = folder + 'test_pre/test/'
    test_destination_folder = folder + 'all_data/test/'

    # Read CSV file containing the labels for each image
    train_labels_file = folder + 'trainLabels.csv'
    test_labels_file = folder + 'testLabels.csv'

    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(test_labels_file)

    # Create folders
    create_folders(train_df, train_destination_folder)
    create_folders(test_df, test_destination_folder)

    # Move images
    move_files(train_df, train_read_folder, train_destination_folder)
    move_files(test_df, test_read_folder, test_destination_folder)


if __name__ == '__main__':
    main()
