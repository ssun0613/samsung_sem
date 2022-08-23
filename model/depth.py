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

class depth_lenet(nn.Module):
    def __init__(self):
        super(depth_lenet, self).__init__()
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=(2, 2)),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=(2, 2)),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU()
                                 )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(64, 1)

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

        feature_maps = self.hookF[layer_index].output.detach().cpu()

        score = 1 / (1 + torch.abs(self.label - out1))
        score.backward(retain_graph=True)
        gradient = self.hookB[layer_index].output[0]
        weighted = gradient.sum(dim=2, keepdimm=True).sum(dim=3, keepdims=True) / (gradient.shape[2] * gradient.shape[3])

        attention = F.relu((feature_maps * weighted).sum(dim=1))
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

        return self.out1

    def predict(self):
        return self.out1

    def predict_1(self):
        return self.out2

    def get_loss(self):
        return self.loss

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
    model = depth_lenet()
    model.init_weights()
    x = torch.rand(1,1,72,48)
    model.set_input(x)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')