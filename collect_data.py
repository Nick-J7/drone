import csv
import os
import sys
sys.path.insert(0, os.path.expanduser('~/projects/microsoft_airsim/AirSim/PythonClient/'))

from PIL import Image
from AirSimClient import *

import pdb

image_save_dir = "data/images"

# Set writer
csvfile = open("data/airsim_record.csv", "w")
fieldnames = ["Timestamp", "Speed(m/s)", "Throttle", "Steering", "Brake", "Gear", "ImageName"]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# Set car client
client = CarClient()
client.confirmConnection()
car_controls = CarControls()
print('Connection established!')

car_state = client.getCarState()
while car_state.collision.has_collided:
    car_state = client.getCarState()
        
    responses = client.simGetImages([
        ImageRequest(0, AirSimImageType.Scene, False, False)
        ])
    image_response = responses[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    pil_image = Image.fromarray(image_rgba)

    writer.writerow({
        "Timestamp": car_state.timestamp,
        "Speed(m\s)": car_state.speed,
        "Throttle": car_control


    pil_image.save(os.path.join(image_save_dir, "test.png"))

csvfile.close()
