import os
import sys
import unittest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from prepare_data import *


class Tester(unittest.TestCase):
    """Test prepare_data."""

    RAW_DATA_DIR = 'test/assets/data_raw'
    COOKED_DATA_DIR = 'test/assets/cooked_data'
    
    data_folders = sorted(os.listdir('test/assets/data_raw'))
    full_path_data_folders = [os.path.join('test/assets/data_raw', f) for f in data_folders]

    def test_get_data_count(self):

        rec_filename = 'airsim_rec.txt'

        num_data = 0
        for folder in Tester.full_path_data_folders:
            with open(os.path.join(folder, rec_filename)) as f:
                num_lines = sum(1 for line in f)
            # The first row is header.
            num_lines -= 1
            num_data += num_lines

        num_of_all_data = get_data_count(Tester.full_path_data_folders)

        self.assertEqual(num_data, num_of_all_data)

    def test_split_data(self):

        num_data = 10
        data = [0 for _ in range(num_data)]
        split_ratio = (0.7, 0.2, 0.1)
        train, validation, test = split_data(data, split_ratio)

        self.assertEqual(len(train), int(num_data * split_ratio[0]))
        self.assertEqual(len(validation), int(num_data * split_ratio[1]))
        self.assertEqual(len(test), int(num_data * split_ratio[2]))
        self.assertEqual(len(train) + len(validation) + len(test), num_data) 

    def test_generate_data(self):

        num_of_all_data = get_data_count(Tester.full_path_data_folders)

        # When each txt file is processed, the first and the last row is excluded.
        num_of_cooked_data = num_of_all_data - 2 * len(Tester.full_path_data_folders)

        all_data = generate_data(Tester.full_path_data_folders)

        self.assertEqual(len(all_data), num_of_cooked_data)

    def test_make_data(self):

        make_data(Tester.full_path_data_folders, Tester.COOKED_DATA_DIR)
        file_list = os.listdir(Tester.COOKED_DATA_DIR)
        self.assertIn('all_data.csv', file_list)
        self.assertIn('train.csv', file_list)
        self.assertIn('validation.csv', file_list)
        self.assertIn('test.csv', file_list)


if __name__ == '__main__':
    unittest.main()
