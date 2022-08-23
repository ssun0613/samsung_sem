import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, stride = (1, 1), padding = 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
    def forward(self, x):
        return self.Conv(x)

class shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(shortcut, self).__init__()
        self.shortcut = Conv(in_channels, out_channels, kernel_size, stride, padding)

        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride = stride, padding = 1, kernel_size = (1, 1)),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU())
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        out = self.shortcut(x)
        return out + self.projection(x)

class sem_resnet(nn.Module):
    def __init__(self):
        super(sem_resnet, self).__init__()
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, stride=(1, 1), kernel_size=(3, 3)),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 shortcut(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=2),
                                 shortcut(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                 shortcut(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                 shortcut(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=2),
                                 shortcut(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
                                 shortcut(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=2),
                                 shortcut(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
                                 shortcut(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=2),
                                 shortcut(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
                                 shortcut(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=2, padding=2),
                                 shortcut(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=2),
                                 )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.decoder_1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1536),
            nn.ReLU(),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 2),
            nn.ConvTranspose2d(256, 512, 3, 2, [0, 1]),
            nn.ConvTranspose2d(512, 256, 3, 2),
            nn.ConvTranspose2d(256, 128, 2, 2, [1, 1]),
            nn.ConvTranspose2d(128, 64, 2, 1, [1, 1]),
            nn.ConvTranspose2d(64, 32, 2, 1, [1, 1]),
            nn.ConvTranspose2d(32, 32, 2, 1, [1, 1]),
            nn.ConvTranspose2d(32, 1, 2, 1, [1, 1]),
        )

    def set_input(self, x):
        self.input = x

    def forward(self):
        out = self.CNN(self.input)
        latent = self.linear(out)
        out = self.decoder_1(latent)
        out = out.view(-1, 128, 4, 3)
        self.output = self.decoder_2(out)

        return self.output

    def predict(self):
        return self.output

    def get_outputs(self):
        return self.output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_networks(self, net, net_type, device, weight_path=None):
        load_filename = 'latest_net_{}.pth'.format(net_type)
        if weight_path is None:
            ValueError('Should set the weight_path, which is the path to the folder including weights')
        else:
            load_path = os.path.join(weight_path, load_filename)
        net = net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device

        state_dict = torch.load(load_path, map_location=str(device))
        net.load_state_dict(state_dict['net'])

        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net

if __name__ == '__main__':
    print('Debug StackedAE')
    model = sem_resnet()
    model.init_weights()
    x = torch.rand(1,1,72,48)
    model.set_input(x)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')