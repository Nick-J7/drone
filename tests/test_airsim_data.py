import os
import sys
import unittest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from airsim_data import *

import torch
from PIL import Image
import matplotlib.pyplot as plt


class Tester(unittest.TestCase):
    """Test airsim_data."""

    # Sample for test
    image_path = 'test/assets/data_raw/normal_1/images/img_0.png'
    image = pil_loader(image_path)
    label = 0.001
    sample = {'image':image, 'label':label}

    def test_ROICrop(self):
        box = (0, 74, 256, 144)
        crop = ROICrop(box)
        result = crop(Tester.sample)
        self.assertEqual(result['image'].size, (256, 70))

    # TODO: test resize
    def test_Resize(self):
        pass

    def test_BrightJitter(self):
        jitter = BrightJitter(0.3)
        result = jitter(Tester.sample)
        self.assertFalse((np.array(result['image']) == np.array(Tester.sample['image'])).all())

    def test_horizontalFlip(self):
        flip = HorizontalFlip(1)
        result = flip(Tester.sample)

        self.assertEqual(Tester.sample['label'], -result['label'])

        ax = plt.subplot(1, 2, 1)
        plt.tight_layout()
        ax.set_title('original')
        ax.axis('off')
        plt.imshow(Tester.sample['image'])

        ax = plt.subplot(1, 2, 2)
        plt.tight_layout()
        ax.set_title('flipped')
        ax.axis('off')
        plt.imshow(result['image'])
        plt.show()

    def test_ToTensor(self):
        result = ToTensor()(Tester.sample)
        self.assertIsInstance(result['image'], torch.FloatTensor)
        self.assertIsInstance(result['label'], torch.FloatTensor)
        self.assertEqual(result['image'].shape, torch.Size([3, 144, 256]))

    def test_airsimdataset(self):
        airsim_dataset = AirSimDataSet(csv_filepath='test/assets/cooked_data/train.csv')
        for i in range(len(airsim_dataset)):
            image, label = airsim_dataset[i]
            print(i, 'image:', image.size, 'label:', label)

            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(image)
            
            if i == 3:
                plt.show()
                break

    def test_get_data_count(self):
        zero_count, nonzero_count = get_data_count(
            csv_file_path='test/assets/cooked_data/all_data.csv')
        self.assertIs(int(zero_count), 0)
        self.assertIs(int(nonzero_count), 16)


if __name__ == '__main__':
    unittest.main()
