import os
import cv2
import json
import filelock
import numpy as np
from tqdm import tqdm

filelock.FileLock = filelock.SoftFileLock
from datasets import load_dataset

ds = load_dataset("raulc0399/open_pose_controlnet", split='train', cache_dir='openpose_data/', streaming=True)

# Specify the directory to save images
image_dir = '/home/mkocabas/projects/x-flux/openpose_data/'

# Create the directory if it doesn't exist
os.makedirs(image_dir, exist_ok=True)
# print(f'there are {len(ds['train'])} images in the dataset')

# for i in tqdm(range(1001)):
i = 0
for data in ds:
    cv2.imwrite(os.path.join(image_dir,f'{i}.png'), np.asarray(data['image'])[..., ::-1])
    cv2.imwrite(os.path.join(image_dir,f'c{i}.png'), np.asarray(data['conditioning_image']))
    with open (os.path.join(image_dir,f'{i}.json'), 'w') as jsonfile:
        json.dump(data['text'], jsonfile)

    i += 1

    if i > 100:
        break