import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd

class BaseModel(nn.Module):
    def __init__(self, width, height):
        super(BaseModel, self).__init__()
        self.width = width
        self.height = height
        self.encoder = nn.Sequential(
            nn.Linear(self.width * self.height, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.width * self.height),
        )

    def set_input(self, x):
        self.input = x

    def forward(self):
        self.input = self.input.view(-1, self.width * self.height)
        self.output = self.decoder(self.encoder(self.input))
        self.output = self.output.view(-1, 1,self.width * self.height)
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

    def scores(self, label, pred):  # label은 test_y, pred는 예측값
        from sklearn.metrics import mean_squared_error

        label = label.cpu().detach().numpy()
        label = pd.DataFrame(label)
        pred = pred.cpu().detach().numpy()
        all_nrmse = []
        for idx in range(len(label.columns)):  # columns 수만큼 반복하고자 함
            rmse = mean_squared_error(label.iloc[:, idx], pred[:, idx],
                                      squared=False)  # label은 dataframe 형태, pred는 numpy 형태
            nrmse = rmse / np.mean(np.abs(label.iloc[:, idx]))
            all_nrmse.append(nrmse)
            score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])

        return score


if __name__ == '__main__':
    print('Debug StackedAE')
    model = neural_net()
    model.init_weights()
    x = torch.rand(52)
    model.set_input(x)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')