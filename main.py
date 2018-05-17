import torch

def main():

    # Trnasform
    # Crop the road(essential): img (256, 144) -> cropped (256, 70)
    transform = transform.Compose([
        transforms.Lambda(lambda img: img.crop((0, 74, 256, 144))),
        transforms.ToTensor(),
        ])

    # Augmentation ?
    # horizontal flip -> reverse steering angle
    # color jitter, particular brightness
