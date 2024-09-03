import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.img_dir = img_dir
        self.images = []
        self.controlimages = []
        self.load_images()
        self.img_size = img_size

    def load_images(self):
        for filename in os.listdir(self.img_dir):
            if filename.endswith(('.jpg', '.png')):  # Check for image file extensions
                if filename[0].isdigit():  # Check if the first character is a digit
                    self.images.append(os.path.join(self.img_dir, filename))
                elif filename[0].isalpha():  # Check if the first character is an alphabet
                    self.controlimages.append(os.path.join(self.img_dir, filename))
        self.images.sort()
        self.controlimages.sort()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            import ipdb; ipdb.set_trace()
            img = Image.open(self.images[idx])
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))

            hint = Image.open(self.controlimages[idx])
            hint = c_crop(hint)
            hint = hint.resize((self.img_size, self.img_size))


            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            
            json_path = self.images[idx].split('.')[0] + '.json'
            prompt = json_path.read().strip()
            import ipdb; ipdb.set_trace()
            return img, hint, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)