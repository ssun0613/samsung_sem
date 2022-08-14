import random
import numpy as np
import os
import glob
import cv2

import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from sklearn.metrics import mean_squared_error

from dataset_load.sem_dataload import sem_dataload
from model.sem_model import BaseModel

import warnings
warnings.filterwarnings(action='ignore')

import zipfile


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(41) # Seed 고정


simulation_sem_paths = '/storage/mskim/samsung/open/simulation_data/SEM/'
simulation_depth_paths = '/storage/mskim/samsung/open/simulation_data/Depth/'


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    result_name_list = []
    result_list = []
    with torch.no_grad():
        for sem, name in tqdm(iter(test_loader)):
            sem = sem.float().to(device)
            model_pred = model(sem)

            for pred, img_name in zip(model_pred, name):
                pred = pred.cpu().numpy().transpose(1, 2, 0) * 255.
                save_img_path = f'{img_name}'
                # cv2.imwrite(save_img_path, pred)
                result_name_list.append(save_img_path)
                result_list.append(pred)

    os.makedirs('/storage/mskim/samsung/open/submission', exist_ok=True)
    os.chdir("/storage/mskim/samsung/open/submission/")
    sub_imgs = []
    for path, pred_img in zip(result_name_list, result_list):
        cv2.imwrite(path, pred_img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("/storage/mskim/samsung/open/submission/submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

    # test_sem_path_list = sorted(glob.glob('/storage/mskim/samsung/open//test/SEM/*.png'))
    #
    # test_dataset = sem_dataload(test_sem_path_list, None)
    # test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)
    # inference(infer_model, test_loader, device)

if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------------------
    want_load = 'sem'
    best_score = 999999
    best_model = None
    batch_size = 2000
    epoch = 1
    datasize = (52, 52)
    width = 48
    height = 72
    bf_lr = 1e-4
    lr = 1e-4
    lr_min = 1e-8
    cpu_id = '1'
    continue_train = False
    device = torch.device("cuda:{}".format(cpu_id) if torch.cuda.is_available() else "cpu")

    import wandb
    # wandb.init(project=want_load)
    # --------------------------------------------------------------------------------------------------------------------
    print('want_load : {}\nbatch_size : {}\nepoch : {}\nbf_lr : {}\nlr : {}\n'.format(want_load, batch_size, epoch, bf_lr, lr))
    # --------------------------------------------------------------------------------------------------------------------

    net = BaseModel(width, height).to(device)

    if not continue_train:
        net.init_weights()
    else:
        net.load_networks(net=net, net_type='lr_{}_{}'.format(want_load, bf_lr), device=device,
                          weight_path='/home/ssunkim/PycharmProjects/LGAimers_Project/ssun/checkpoints')

    train_dataset = sem_dataload(simulation_sem_paths, simulation_depth_paths, datasize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    # ****** Optimizer, Scheduler setup ******
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=lr_min)
    fn_loss = nn.MSELoss()

    # ---- Training ----
    global_step = 0
    temp_loss = []
    temp_acc = []
    temp_score = []

    for curr_epoch in range(epoch):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))
        train_loss = []
        for sem, depth in enumerate(train_loader, 1):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            optimizer.zero_grad()

            net.set_input(sem)
            model_pred = net.forward(width, height)
            loss = fn_loss(model_pred, depth)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))
        #
        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])

        print('Save network...')
        torch.save({'net': net.state_dict()},
                   '/home/ssunkim/PycharmProjects/LGAimers_Project/ssun/checkpoints' + '/latest_net_lr_{}_{}.pth'.format(want_load, lr))
