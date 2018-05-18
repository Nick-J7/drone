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

    def test_BrightJitter(self):
        jitter = BrightJitter(0.3)
        result = jitter(Tester.sample)
        self.assertFalse((np.array(result['image']) == np.array(Tester.sample['image'])).all())

    def test_ToTensor(self):
        result = ToTensor()(Tester.sample)
        self.assertIsInstance(result['image'], torch.FloatTensor)
        self.assertIsInstance(result['label'], torch.FloatTensor)

    def test_airsimdataset(self):
        airsim_dataset = AirSimDataSet(csv_file_path='test/assets/cooked_data/train.csv')
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


if __name__ == '__main__':
    unittest.main()
