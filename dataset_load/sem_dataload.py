import random
import pandas as pd
import numpy as np
import torch
import os
import glob
import cv2

class sem_dataload():
    def __init__(self, simulation_sem_paths, simulation_depth_paths):
        self.simulation_sem_paths = sorted(glob.glob(simulation_sem_paths + '*/*/*.png'))
        self.simulation_depth_paths = sorted(glob.glob(simulation_depth_paths + '*/*/*.png') + glob.glob(simulation_depth_paths + '*/*/*.png'))

        self.data_len = len(self.simulation_sem_paths)

        self.train_sem = self.simulation_sem_paths[:int(self.data_len * 0.8)]
        self.train_depth = self.simulation_depth_paths[:int(self.data_len * 0.8)]

    def __len__(self):
        return len(self.train_sem)

    def __getitem__(self, index):
        sem_path = self.train_sem[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        sem_img = sem_img / 255.

        if self.train_depth is not None:
            depth_path = self.train_depth[index]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
            depth_img = depth_img / 255.
            return torch.Tensor(sem_img), torch.Tensor(depth_img)  # B,C,H,W
        else:
            img_name = sem_path.split('/')[-1]
            return torch.Tensor(sem_img), img_name  # B,C,H,W

if __name__=='__main__':
    simulation_sem_paths = '/storage/mskim/samsung/open/simulation_data/SEM/'
    simulation_depth_paths = '/storage/mskim/samsung/open/simulation_data/Depth/'

    dataset_object_train = sem_dataload(simulation_sem_paths, simulation_depth_paths)

    x, y = dataset_object_train.__getitem__(0)