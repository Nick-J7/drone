import csv
import logging
import os
import os.path as osp
import time
import random

import pandas as pd

import pdb


def get_data_count(folders):
    """Get information about number of data.
        Args:
           folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]

        Returns:
            int: number of all data
    """
    rec_filename = 'airsim_rec.txt'

    num_of_all_data = 0
    for folder in folders:
        current_dataframe = pd.read_csv(osp.join(folder, rec_filename), sep='\t')
        num_of_all_data += current_dataframe.shape[0]

    return num_of_all_data


def split_data(data, split_ratio):
    """Split data into three groups by using split ratio.
        Args:
            data (list): list of tuple data
            split_ratio (tuple): ratio of split 
                0 -> train
                1 -> validation
                2 -> test
        Returns:
            tuple: splitted data 
                0 -> train(list)
                1 -> validation(list)
                2 -> test(list)
    """
    num_train = int(len(data) * split_ratio[0])
    num_val = int(len(data) * split_ratio[1])

    train = data[:num_train]
    val = data[num_train:num_train + num_val]
    test = data[num_train + num_val:]
    
    return train, val, test


def generate_data(folders):
    """Generate cooked data.
        Args:
            folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]
        Returns:
            list: all data which is shuffled (list of tuple)
    """
    rec_filename = 'airsim_rec.txt'
    all_data = []

    for folder in folders:
        current_dataframe = pd.read_csv(osp.join(folder, rec_filename), sep='\t')

        for i in range(1, current_dataframe.shape[0] - 1):
            # Set path of image for loading in torch.utils.data.DataLoader
            imagename = current_dataframe.iloc[i]['ImageName']
            imagepath = osp.join(folder, 'images', imagename)

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
    return all_data
    

def make_data(folders, output_dir):
    """Print information of data.
        Args:
            folders (list): full path of raw data directory such as 
                ['data_raw/normal_1', 'data_raw/normal_2', ..]
            output_dir (string): directory for saving output file 
    """

    # Print number of data
    num_of_all_data = get_data_count(folders)
    logging.info("Number of all data: {}."
          .format(num_of_all_data))

    # Generate data
    logging.info('Generating data...')
    s_time = time.time()
    all_data = generate_data(folders)
    e_time = time.time()
    logging.info('Elapsed time for generating data: {}'.format(e_time - s_time))

    # Split data
    train_eval_test_split = (0.7, 0.2, 0.1)
    train, validation, test = split_data(all_data, train_eval_test_split)
    logging.info("Number of train: {}".format(len(train)))
    logging.info("Number of validation: {}".format(len(validation)))
    logging.info("Number of test: {}".format(len(test)))

    # Write data
    fieldnames = ['Imagepath', 'Timestamp', 'Speed (kmph)', 'Throttle', 'Brake', 'Gear', 'Label']

    def write_data(data, filename):
        csvfile = open(osp.join(output_dir, filename), 'w')
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


    write_data(all_data, 'all_data.csv')
    write_data(train, 'train.csv')
    write_data(validation, 'validation.csv')
    write_data(test, 'test.csv')

    logging.info('Finish !!!') 


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    RAW_DATA_DIR = 'data_raw'
    COOKED_DATA_DIR = 'cooked_data'

    data_folders = sorted(os.listdir(RAW_DATA_DIR))
    full_path_data_folders = [osp.join(RAW_DATA_DIR, f) for f in data_folders]
    make_data(full_path_data_folders, COOKED_DATA_DIR)
