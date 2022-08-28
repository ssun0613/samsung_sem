import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import glob
import cv2

class train_dataload():
    def __init__(self, sem_paths, depth_paths):
        self.train_sem = sorted(glob.glob(sem_paths + '*/*/*.png'))

        self.train_depth = pd.read_csv(depth_paths)
        self.train_depth['3'] = self.train_depth['0'].str.split('0_').str[1]

        self.data_len = len(self.train_sem)

    def __len__(self):
        return len(self.train_sem)

    def __getitem__(self, index):
        sem_path = self.train_sem[index]
        sem_path_1 = self.train_sem[index].split('/')[7].split('_')[1]
        sem_path_2 = self.train_sem[index].split('/')[8]

        dep_val = 0
        for index in range(len(self.train_depth.index)):
            if self.train_depth.iloc[index, 1] == int(sem_path_1):
                for i in range(len(self.train_depth.index)):
                    if self.train_depth.iloc[i, 3] == sem_path_2:
                        dep_val = self.train_depth.iloc[i, 2]
                    else:
                        None
            else:
                None


        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        sem_img = sem_img / 255.

        sem_img = torch.Tensor(sem_img)
        dep_val = torch.Tensor(dep_val)

        return {'input' : sem_img , 'label' : dep_val}
class simulation_dataload():
    def __init__(self, simulation_sem_paths, simulation_depth_paths):
        self.train_sem = sorted(glob.glob(simulation_sem_paths + '*/*/*.png'))
        self.train_depth = sorted(glob.glob(simulation_depth_paths + '*/*/*.png') + glob.glob(simulation_depth_paths + '*/*/*.png'))

        self.data_len = len(self.train_sem)

    def __len__(self):
        return len(self.train_sem)

    def __getitem__(self, index):
        sem_path = self.train_sem[index]

        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)

        sem_img = sem_img / 255.

        depth_path = self.train_depth[index]
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
        depth_img = depth_img / 255.

        sem_img = torch.Tensor(sem_img)
        depth_img = torch.Tensor(depth_img)

        return {'input' : sem_img , 'label' : depth_img}
class simulation_sem_dataload():
    def __init__(self, simulation_sem_paths, simulation_depth_paths):
        self.train_sem = sorted(glob.glob(simulation_sem_paths + '*/*/*.png'))
        self.train_depth = sorted(glob.glob(simulation_depth_paths + '*/*/*.png')+glob.glob(simulation_depth_paths + '*/*/*.png'))

        self.data_len = len(self.train_depth)

    def __len__(self):
        return len(self.train_sem)

    def __getitem__(self, index):
        sem_path = self.train_sem[(2 * index) % self.data_len]
        sem_path_1 = self.train_sem[(2 * index + 1) % self.data_len]

        sem_img_1 = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img_2 = cv2.imread(sem_path_1, cv2.IMREAD_GRAYSCALE)


        sem_img_1 = np.expand_dims(sem_img_1, axis=-1).transpose(2, 0, 1)
        sem_img_2 = np.expand_dims(sem_img_2, axis=-1).transpose(2, 0, 1)

        sem_img = np.sum([sem_img_1, sem_img_2], axis=0) / 2.0
        sem_img = sem_img / 255.

        depth_path = self.train_depth[(2 * index) % self.data_len]
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
        depth_img = depth_img / 255.

        sem_img = torch.Tensor(sem_img)
        depth_img = torch.Tensor(depth_img)

        return {'input': sem_img, 'label': depth_img}
class simulation_depth_dataload():
    def __init__(self, simulation_depth_paths):
        self.train_depth_image = sorted(glob.glob(simulation_depth_paths + '*/*/*.png'))
        self.depth_len = len(self.train_depth_image)

    def __len__(self):
        return len(self.train_depth_image)

    def __getitem__(self, index):
        depth_path = self.train_depth_image[index]
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
        depth_img = depth_img / 255.

        depth_img = torch.Tensor(depth_img)

        depth_label = float(depth_path.split('/')[8])

        return { 'input' : depth_img, 'label' : depth_label }
class test_dataload():
    def __init__(self, test_sem_paths):
        self.test_sem_paths = sorted(glob.glob(test_sem_paths + 'SEM/*.png'))

        self.test_sem = self.test_sem_paths
    def __len__(self):
        return len(self.test_sem)

    def __getitem__(self, index):
        sem_path = self.test_sem[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        sem_img = sem_img / 255.

        sem_img = torch.Tensor(sem_img)
        img_name = sem_path.split('/')[-1]

        return {'sem' : sem_img , 'depth' : img_name}

if __name__=='__main__':
    simulation_sem_paths = '/storage/mskim/samsung/open/simulation_data/SEM/'
    simulation_depth_paths = '/storage/mskim/samsung/open/simulation_data/Depth/'

    train_sem_paths = '/storage/mskim/samsung/open/train/SEM/'
    train_depth_paths = '/storage/mskim/samsung/open/train/cv_train/cv_train_1.csv'
    test_sem_paths = '/storage/mskim/samsung/open/test/'

    datasize = (52, 52)
    dataset_object_train = simulation_depth_dataload(simulation_depth_paths)

    x, y = dataset_object_train.__getitem__(0)