import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd

class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBottleneck, self).__init__()
        self.DenseBottleneck = nn.Sequential(nn.BatchNorm2d(in_channels),
                                             nn.ReLU(),
                                             nn.Conv2d(in_channels, 4 * out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0),
                                             nn.BatchNorm2d(4 * out_channels),
                                             nn.ReLU(),
                                             nn.Conv2d(4 * out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.DenseBottleneck(x), self.shortcut(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(DenseBlock, self).__init__()
        self.DenseBlock = self.block(in_channels, out_channels, num_layer)

    def block(self, in_channels, out_channels, num_layer):
        block = []
        for i in range(num_layer):
            block.append(DenseBottleneck((in_channels + i * out_channels), out_channels))
        return nn.Sequential(*block)

    def forward(self, x):
        return self.DenseBlock(x)

class Trans_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Trans_layer, self).__init__()
        self.Trans_layer = nn.Sequential(nn.BatchNorm2d(in_channels),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                         nn.ReLU(),
                                         nn.AvgPool2d(kernel_size = (2, 2), stride = 2))

    def forward(self, x):
        return self.Trans_layer(x)

class sem_densenet(nn.Module):
    def __init__(self):
        super(sem_densenet, self).__init__()
        self.k = 12
        self.CNN = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 2 * self.k, kernel_size = (3, 3), stride = 1),
                                 nn.BatchNorm2d(2 * self.k),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size = (3, 3), stride = 1),
                                 DenseBlock(in_channels = 2 * self.k, out_channels = self.k, num_layer = 6),
                                 Trans_layer(in_channels = 2 * self.k + 6 * self.k, out_channels = self.k, kernel_size = (1, 1), stride = 1, padding = 0),
                                 DenseBlock(in_channels = self.k, out_channels = self.k, num_layer = 12),
                                 Trans_layer(in_channels = self.k + 12 * self.k, out_channels = self.k, kernel_size = (1, 1), stride = 1, padding = 0),
                                 DenseBlock(in_channels = self.k, out_channels = self.k, num_layer = 24),
                                 nn.BatchNorm2d(self.k + 24 * self.k),
                                 nn.ReLU()
                                 )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.k + 24 * self.k, self.k + 12 * self.k, 3, 2),
            nn.ConvTranspose2d(self.k + 12 * self.k, 2 * self.k + 6 * self.k, 3, 2),
            nn.ConvTranspose2d(2 * self.k + 6 * self.k, 2 * self.k, 3, 1, [1,1]),
            nn.ConvTranspose2d(2 * self.k, 2 * self.k, 3, 1, [1,1]),
            nn.ConvTranspose2d(2 * self.k, 1, 2, 1),
        )

    def set_input(self, x, label):
        self.input = x

    def forward(self):
        out = self.CNN(self.input)
        out = out.view(-1, 300, 17, 11)
        self.output = self.decoder(out)
        return self.output

    def predict(self):
        return self.output

    def get_outputs(self):
        return self.output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
    model = sem_densenet()
    model.init_weights()
    x = torch.rand(1,1,72,48)
    label = torch.rand(1, 1)
    model.set_input(x, label)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')