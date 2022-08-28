import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np

class Hook():
    def __init__(self, module, backward):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def closs(self):
        self.hook.remove()

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

class depth_resnet(nn.Module):
    def __init__(self):
        super(depth_resnet, self).__init__()
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
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(128, 1)

        self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]
        self.hookB = [Hook(layers[1], backward=True) for layers in list(self._modules.items())]

    def set_input(self, x, label):
        self.input = x
        self.label = label

    def _forward_reg(self, x):
        CNN = self.CNN(x)
        GAP = self.GAP(CNN)
        self.output = self.classification(self.flatten(GAP))
        return self.output

    def _make_att_map(self, out1):
        layer_index = np.argmax(np.array([name == 'CNN' for name in self._modules.keys()], dtype=np.int))
        feature_maps = self.hookF[layer_index].output

        # score = 1 / (1 + torch.abs(self.label - out1))
        # score.backward(gradient=score, retain_graph=True)

        score = 1 / (1 + torch.abs(self.label - out1))
        score.mean().backward(retain_graph=True)

        gradient = self.hookB[layer_index].output[0]
        weighted = gradient.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (gradient.shape[2] * gradient.shape[3])
        attention = F.relu((feature_maps.detach() * weighted.detach()).sum(dim=1).unsqueeze(dim=1))
        attention = F.interpolate(attention, size=(self.input.shape[2], self.input.shape[3]), mode='bicubic')

        attention = (attention - attention.min()) / (attention.max() - attention.min())

        return attention

    def _make_masked_img(self, attention):
        attention[attention <= 0.5] = 1
        attention[attention > 0.5] = 0

        self.input_1 = self.input * attention + (1 - attention) * 255

        return self.input_1

    def forward(self):
        self.out1 = self._forward_reg(self.input)
        attMap = self._make_att_map(self.out1)
        maskedImg = self._make_masked_img(attMap)
        self.out2 = self._forward_reg(maskedImg)
        self.loss_cl = F.mse_loss(self.label, self.out1)
        loss_am = 1 / (1 + F.mse_loss(self.label, self.out2))
        self.loss = self.loss_cl + loss_am

    def predict(self):
        return self.out1
    def predict_1(self):
        return self.out2
    def get_loss(self):
        return self.loss_cl
    def total_loss(self):
        return self.loss
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
    model = depth_resnet()
    model.init_weights()
    x = torch.rand(1,1,72,48)
    label = torch.rand(1, 1)
    model.set_input(x, label)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')