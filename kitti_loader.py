import numpy as np
import os
import glob
# import cv2
import torch
from PIL import Image
from skimage.util import img_as_float, crop
from skimage.transform import resize

class DataLoader():
    '''
    raw_data_dir = dir containing extracted raw KITTI data (folder containing 2011-09-26 etc.)
    depth_maps_dir = dir containing extracted depth maps
    '''
    
    def __init__(self, raw_images_path, depth_images_path, mode="train", tfms=None):
        
        self.mode = mode
        self.tfms = tfms
        
        if self.mode == "train":
            with open('eigen_train_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == "test":
            with open('eigen_test_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == "val":
            with open('eigen_val_files.txt', 'r') as f:
                self.files = f.readlines()
            
        self.data = []
        for l in self.files:
            s = l.split()
            for im in s:
                self.data.append(raw_images_path + im)

        self.imgs, self.labels = [], []
        
        for img_path in self.data:
            img_name = img_path.split('.')[0] + '.png'
            tokens = img_name.split('/')
            path = depth_images_path + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            path = path.split('.')[0] + '.png'
#             print(path, img_path)
            if os.path.exists(path) and os.path.exists(img_name):
                self.imgs.append(img_name)
                self.labels.append(path)
                
        print('Found %d Images %d'%(len(self.imgs), len(self.labels)))
    
    def __len__(self):
        return len(self.imgs)
    
    def transform_depth(self, depth):
        print(np.min(depth), np.max(depth))
        depth = 1.0 / (depth + 1e-6)
        d_min = np.min(depth)
        d_max = np.max(depth)
        depth_relative = (d_max - depth) / ((d_max - d_min) + 1e-6)
        return depth_relative
    
    def load_data(self, img_file, label_file):
        x = Image.open(img_file)
        w, h = x.size
        x = x.crop((0, int(0.3*h), w, h))
        y = Image.open(label_file)
        y = y.crop((0, int(0.3*h), w, h))
        
        x = np.array(x)
        y = np.array(y)
        
#         print(np.max(y), np.min(y))
#         x = resize(x, (90, 270), anti_aliasing=True)
#         y = resize(y, (90, 270), anti_aliasing=True) 
#         y = y /100.0
#         y[y>80] = 80.0
        y[y<=1] = 1.0
        y = np.log(y)
        x = x.astype(np.uint8)
        y = y.astype(np.uint8)
        if self.tfms:
            tfmd_sample = self.tfms({"image":x, "depth":y})
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]
        
        depth = depth / torch.max(depth)
        
        return img, depth
        
    def get_one_batch(self, batch_size = 64):
        images = []
        labels = []

        while True:
            idx = np.random.choice(len(self.imgs), batch_size)
            for i in idx:
                x, y = self.load_data(self.imgs[i], self.labels[i])
                images.append(x)
                labels.append(y)
            yield torch.stack(images), torch.stack(labels)