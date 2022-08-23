import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import cv2
from model.common import tensor2numpy

def CAM(net, input_label):
    original_size = (net.input.shape[2], net.input.shape[3])

    layer_index = np.argmax(np.array([name == 'CNN' for name in net._modules.keys()], dtype=np.int))

    prediction_class = torch.argmax(net.predict().detach().cpu(), dim=2)
    input_label = torch.argmax(input_label, dim=2)
    weighted = net._modules['classification']._parameters['weight'].detach().cpu()
    feature_maps = net.hookF[layer_index].output.detach().cpu()

    CAM = []
    input_image_pick = []
    input_label_pick = []
    prediction_pick = []
    for batch in range(net.input.shape[0]):
        prediction_index = torch.where(prediction_class[batch])[0]
        if len(prediction_index) > 0:
            prediction_index = 8 + prediction_index
            feature_map = feature_maps[batch]

            for i in range(1):
                cam = (weighted[prediction_index[i]].unsqueeze(dim=1).unsqueeze(dim=1) * feature_map).sum(dim=0)
                cam = cam - torch.min(cam)

                cam_image = cam / torch.max(cam)
                cam_image = (255 * cam_image).type(torch.uint8)
                cam_image = cv2.resize(cam_image.numpy(), original_size)
                cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)

                cam_map = 0.5 * tensor2numpy(net.input[batch]) + 0.5 * cam_image
                CAM.append(cam_map)

                input_image_pick.append(net.input[batch])
                input_label_pick.append(input_label[batch])
                prediction_pick.append(prediction_class[batch])

    return CAM, input_image_pick, input_label_pick, prediction_pick

def GradCAM(net, input_label):
    original_size = (net.input.shape[2], net.input.shape[3])
    layer_index = np.argmax(np.array([name == 'CNN' for name in net._modules.keys()], dtype=np.int))

    prediction_class = torch.argmax(net.predict().detach().cpu(), dim=2)
    input_label = torch.argmax(input_label, dim=2)
    feature_maps = net.hookF[layer_index].output.detach().cpu()

    net.forward()
    scores = net.output

    GradCAM = []
    input_image_pick = []
    input_label_pick = []
    prediction_pick = []
    for batch in range(net.input.shape[0]):
        prediction_index = torch.where(prediction_class[batch])[0]
        if len(prediction_index) > 0:
            prediction_index = 8 + prediction_index
            feature_map = feature_maps[batch]

            for i in range(1):
                score_prediction = scores[batch][prediction_index[i]]
                score_prediction.backward(retain_graph=True)
                gradient = net.hookB[layer_index].output[0][batch].detach().cpu()
                weighted = gradient.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / (gradient.shape[1] * gradient.shape[2])
                Gradcam = (weighted * feature_map).sum(dim=0)
                Gradcam = Gradcam - torch.min(Gradcam)

                Gradcam_image = Gradcam / torch.max(Gradcam)
                Gradcam_image = (255 * Gradcam_image).type(torch.uint8)
                Gradcam_image = cv2.resize(Gradcam_image.numpy(), original_size)
                Gradcam_image = cv2.applyColorMap(Gradcam_image, cv2.COLORMAP_JET)

                Gradcam_map = 0.5 * tensor2numpy(net.input[batch]) + 0.5 * Gradcam_image
                GradCAM.append(Gradcam_map)

                input_image_pick.append(net.input[batch])
                input_label_pick.append(input_label[batch])
                prediction_pick.append(prediction_class[batch])

    return GradCAM, input_image_pick, input_label_pick, prediction_pick

#
# def CAM(net, input_label):
#     original_size = (net.input.shape[2], net.input.shape[3])
#
#     layer_index = np.argmax(np.array([name == 'CNN' for name in net._modules.keys()], dtype=np.int))
#
#     prediction_class = torch.argmax(net.predict().detach().cpu(), dim=2)
#     input_label = torch.argmax(input_label, dim=2)
#     weighted = net._modules['classification']._parameters['weight'].detach().cpu()
#     feature_maps = net.hookF[layer_index].output.detach().cpu()
#
#     CAM = []
#     input_image_pick = []
#     input_label_pick = []
#     prediction_pick = []
#     for batch in range(net.input.shape[0]):
#         prediction_index = torch.where(prediction_class[batch])[0]
#         if len(prediction_index) > 0:
#             prediction_index = 8 + prediction_index
#             feature_map = feature_maps[batch]
#
#             if np.array_equal(input_label[batch].cpu().numpy(), prediction_class[batch].cpu().numpy()) == False:
#                 for i in range(1):
#                     cam = (weighted[prediction_index[i]].unsqueeze(dim=1).unsqueeze(dim=1) * feature_map).sum(dim=0)
#                     cam = cam - torch.min(cam)
#
#                     cam_image = cam / torch.max(cam)
#                     cam_image = (255 * cam_image).type(torch.uint8)
#                     cam_image = cv2.resize(cam_image.numpy(), original_size)
#                     cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
#
#                     cam_map = 0.5 * tensor2numpy(net.input[batch]) + 0.5 * cam_image
#                     CAM.append(cam_map)
#
#                     input_image_pick.append(net.input[batch])
#                     input_label_pick.append(input_label[batch])
#                     prediction_pick.append(prediction_class[batch])
#
#     return CAM, input_image_pick, input_label_pick, prediction_pick
#
# def GradCAM(net, input_label):
#     original_size = (net.input.shape[2], net.input.shape[3])
#     layer_index = np.argmax(np.array([name == 'CNN' for name in net._modules.keys()], dtype=np.int))
#
#     prediction_class = torch.argmax(net.predict().detach().cpu(), dim=2)
#     input_label = torch.argmax(input_label, dim=2)
#     feature_maps = net.hookF[layer_index].output.detach().cpu()
#
#     net.forward()
#     scores = net.output
#
#     GradCAM = []
#     input_image_pick = []
#     input_label_pick = []
#     prediction_pick = []
#     for batch in range(net.input.shape[0]):
#         prediction_index = torch.where(prediction_class[batch])[0]
#         if len(prediction_index) > 0:
#             prediction_index = 8 + prediction_index
#             feature_map = feature_maps[batch]
#
#             if np.array_equal(input_label[batch].cpu().numpy(), prediction_class[batch].cpu().numpy()) == False:
#                 for i in range(1):
#                     score_prediction = scores[batch][prediction_index[i]]
#                     score_prediction.backward(retain_graph=True)
#                     gradient = net.hookB[layer_index].output[0][batch].detach().cpu()
#                     weighted = gradient.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
#                     Gradcam = (weighted * feature_map).sum(dim=0)
#                     Gradcam = Gradcam - torch.min(Gradcam)
#
#                     Gradcam_image = Gradcam / torch.max(Gradcam)
#                     Gradcam_image = (255 * Gradcam_image).type(torch.uint8)
#                     Gradcam_image = cv2.resize(Gradcam_image.numpy(), original_size)
#                     Gradcam_image = cv2.applyColorMap(Gradcam_image, cv2.COLORMAP_JET)
#
#                     Gradcam_map = 0.5 * tensor2numpy(net.input[batch]) + 0.5 * Gradcam_image
#                     GradCAM.append(Gradcam_map)
#
#                     input_image_pick.append(net.input[batch])
#                     input_label_pick.append(input_label[batch])
#                     prediction_pick.append(prediction_class[batch])
#
#     return GradCAM, input_image_pick, input_label_pick, prediction_pick
#
