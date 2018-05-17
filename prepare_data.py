import csv
import os
import os.path as osp
import time
import random

import pandas as pd

import pdb


RAW_DATA_DIR = 'data_raw'
COOKED_DATA_DIR = 'cooked_data'


def print_data_info(folders):
    """Print information of data.
        Inputs:
           folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]
    """

    rec_filename = 'airsim_rec.txt'

    num_of_all_data = 0
    for folder in folders:
        current_dataframe = pd.read_csv(osp.join(folder, rec_filename), sep='\t')
        num_of_all_data += current_dataframe.shape[0]

    print("Number of all data: {}".format(num_of_all_data))
    print("Maybe the number of output data is {}.\n" 
          "Because when each txt file is processed, the first and the last row is excluded."
          .format(num_of_all_data - 2 * len(folders)))


def split_data(data, split_ratio):
    """Split data into three groups by using split ratio
        Inputs:
            data (list): list of tuple data
            split_ratio (tuple): ratio of split 
                0 -> train
                1 -> validation
                2 -> test
        Returns:
            train (list): splitted list of tuple data
            val (list)
            test (list)
    """

    num_train = int(len(data) * split_ratio[0])
    num_val = int(len(data) * split_ratio[1])

    train = data[:num_train]
    val = data[num_train:num_train + num_val]
    test = data[num_train + num_val:]
    print("Number of train:", num_train)
    print("Number of validation:", num_val)
    print("Number of test:", len(data) - num_train - num_val)
    
    return train, val, test


def generate_data(folders):
    """Print information of data.
        Inputs:
           folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]
    """

    fieldnames = ['Imagepath', 'Timestamp', 'Speed (kmph)', 'Throttle', 'Brake', 'Gear', 'Label']

    def write_data(data, filename):
        csvfile = open(osp.join(COOKED_DATA_DIR, filename), 'w')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(data)):
            writer.writerow({
                'Imagepath': data[i][0],
                'Timestamp': data[i][1],
                'Speed (kmph)': data[i][2],
                'Throttle': data[i][3],
                'Brake': data[i][4],
                'Gear': data[i][5],
                'Label': data[i][6]
                })
        csvfile.close()


    rec_filename = 'airsim_rec.txt'
    all_data = []

    print('Generating data...')
    for folder in folders:
        current_dataframe = pd.read_csv(osp.join(folder, rec_filename), sep='\t')

        for i in range(1, current_dataframe.shape[0] - 1):
            # Set path of image for loading in torch.utils.data.DataLoader
            imagepath = osp.join(folder, 'images', current_dataframe.iloc[i]['ImageName'])

            # Label is average of {(t-1), t, (t+1) steering angle}
            current_label = (current_dataframe.iloc[i-1]['Steering'] +
                             current_dataframe.iloc[i]['Steering'] +
                             current_dataframe.iloc[i+1]['Steering']) / 3.0
            # Record previous state
            previous_row = current_dataframe.iloc[i-1]

            current_tuple = (
                imagepath,
                previous_row['Timestamp'],
                previous_row['Speed (kmph)'],
                round(previous_row['Throttle'], 6),
                round(previous_row['Brake'], 6),
                previous_row['Gear'],
                round(current_label, 6)
                )
            
            all_data.append(current_tuple)

    random.shuffle(all_data)
    
    train_eval_test_split = (0.7, 0.2, 0.1)
    train, validation, test= split_data(all_data, train_eval_test_split)

    write_data(all_data, 'all_data.csv')
    write_data(train, 'train.csv')
    write_data(validation, 'validation.csv')
    write_data(test, 'test.csv')

    print('Finish !!!') 


def make_data(folders):
    """Print information of data.
        Inputs:
           folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]
    """

    print_data_info(folders)

    s_time = time.time()
    generate_data(folders)
    e_time = time.time()
    print('Elapsed time: {}'.format(e_time - s_time))


if __name__ == '__main__':

    data_folders = sorted(os.listdir(RAW_DATA_DIR))
    full_path_data_folders = [osp.join(RAW_DATA_DIR, f) for f in data_folders]
    make_data(full_path_data_folders)
