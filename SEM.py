import os, random, cv2, zipfile, time
os.environ["CONFIG_PATHS"] = '/storage/mskim/wandb/'
os.environ["WANDB_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CACHE_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CONFIG_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_RUN_DIR"] = '/storage/mskim/wandb/'

import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from sklearn.metrics import mean_squared_error
from model.sem_model import autoencoder
from function import *

simulation_sem_paths = '/storage/mskim/samsung/open/simulation_data/SEM/'
simulation_depth_paths = '/storage/mskim/samsung/open/simulation_data/Depth/'

train_sem_paths = '/storage/mskim/samsung/open/train/SEM/'
train_depth_paths = '/storage/mskim/samsung/open/train/cv_train/cv_train_1.csv'
test_sem_paths = '/storage/mskim/samsung/open/test/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(41) # Seed 고정

def test(net, test_loader, device):
    net.to(device)
    net.eval()

    result_name_list = []
    result_list = []
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader, 1):
            sem = data['sem'].type('torch.FloatTensor').to(device)
            depth = data['depth']
            net.set_input(sem, depth)
            net.forward()
            output = net.predict()

            for pred, img_name in zip(output, depth):
                pred = pred.cpu().numpy().transpose(1, 2, 0) * 255.
                save_img_path = f'{img_name}'
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
    print('save submit file!')

def data_load(want_load):
    if want_load == 'train':
        from dataset_load.sem_dataload import train_dataload, test_dataload
        train_dataset = train_dataload(train_sem_paths, train_depth_paths)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = test_dataload(test_sem_paths)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    elif want_load == 'sem_depth_simulation':
        from dataset_load.sem_dataload import simulation_dataload, test_dataload
        train_dataset = simulation_dataload(simulation_sem_paths, simulation_depth_paths)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = test_dataload(test_sem_paths)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    elif want_load == 'sem_simulation':
        from dataset_load.sem_dataload import simulation_sem_dataload, test_dataload
        train_dataset = simulation_sem_dataload(simulation_sem_paths, simulation_depth_paths)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = test_dataload(test_sem_paths)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    elif want_load == 'depth_simulation':
        from dataset_load.sem_dataload import simulation_depth_dataload, test_dataload
        train_dataset = simulation_depth_dataload(simulation_depth_paths)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = test_dataload(test_sem_paths)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def network_load(want_net):
    if want_net == 'sem':
        from model.sem_model_resnet import sem_resnet
        from model.sem_model_densenet import sem_densenet
        # net_1 = sem_resnet().to(device)
        net_1 = sem_densenet().to(device)
        return net_1

    elif want_net == 'depth':
        from model.depth_model_resnet import depth_resnet
        net_1 = depth_resnet().to(device)
        return net_1

    elif want_net == 'sem_to_depth':
        from model.sem_model_resnet import sem_resnet
        from model.depth_model_resnet import depth_resnet
        net_1 = sem_resnet().to(device)
        net_2 = depth_resnet().to(device)
        return net_1, net_2

def sem_to_depth(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device):

    sem_net, depth_net = network_load(want_net)

    if not continue_train:
        depth_net.init_weights()
    else:
        depth_net.load_networks(net=depth_net, net_type='lr_{}'.format(lr), device=device,
                                weight_path='/home/ssunkim/PycharmProjects/samsung_sem/checkpoints')

    train_loader, test_loader = data_load(want_load)

    # test(net, test_loader, device)

    optimizer = torch.optim.Adam(sem_net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=lr_min)

    fn_loss = nn.MSELoss()

    # ---- Training ----
    global_step = 0
    train_loss = []
    for curr_epoch in range(epoch):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))

        for batch_id, data in enumerate(train_loader, 1):
            global_step += 1
            image = data['input'].type('torch.FloatTensor').to(device)
            label = data['label'].type('torch.FloatTensor').to(device)

            sem_net.set_input(image, label)
            sem_net.forward()
            depth_image = sem_net.predict()
            sem_net_loss = fn_loss(label, depth_image)

            with torch.no_grad():
                depth_net.set_input(depth_image, label)
                depth_net.forward()
                depth_net_loss = depth_net.get_loss()

            loss = sem_net_loss + depth_net_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().detach().numpy().item())
            #
            # if global_step % 10 == 0:
            #     loss_dict = dict()
            #     loss_dict['loss'] = np.mean(train_loss)
            #     loss_dict['lr'] = optimizer.param_groups[0]['lr']
            #     log(prefix='train', metrics_dict=loss_dict)
            # train_loss = []

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))
        #
        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])

        print('Save network...')
        torch.save({'net': sem_net.state_dict()},
                   '/home/ssunkim/PycharmProjects/samsung_sem/checkpoints' + '/latest_net_lr_{}.pth'.format(lr))
def sem(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device):
    net = network_load(want_net)

    if not continue_train:
        net.init_weights()
    else:
        net.load_networks(net=net, net_type='lr_{}'.format(lr), device=device,
                          weight_path='/home/ssunkim/PycharmProjects/samsung_sem/checkpoints/sem')

    train_loader, test_loader = data_load(want_load)

    test(net, test_loader, device)

    # ****** Optimizer, Scheduler setup ******
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=lr_min)

    fn_loss = nn.MSELoss()

    # ---- Training ----
    global_step = 0
    train_loss = []
    for curr_epoch in range(epoch):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))

        for batch_id, data in enumerate(train_loader, 1):
            global_step += 1
            image = data['input'].type('torch.FloatTensor').to(device)
            label = data['label'].type('torch.FloatTensor').to(device)

            net.set_input(image, label)
            net.forward()

            out = net.predict()
            loss = fn_loss(label, out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().detach().numpy().item())
            #
            if global_step % 50 == 0:
                loss_dict = dict()
                loss_dict['epoch'] = curr_epoch
                loss_dict['loss'] = np.mean(train_loss)
                loss_dict['lr'] = optimizer.param_groups[0]['lr']
                log(prefix='train', metrics_dict=loss_dict)

                train_loss = []

            if global_step % 100 == 0:
                log_image(image, label, out)

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))
        #
        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])

        print('Save network...')
        torch.save({'net': net.state_dict()},
                   '/home/ssunkim/PycharmProjects/samsung_sem/checkpoints/sem' + '/latest_net_lr_{}.pth'.format(lr))
def depth(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device):
    net = network_load(want_net)

    if not continue_train:
        net.init_weights()
    else:
        net.load_networks(net=net, net_type='lr_{}'.format(lr), device=device,
                          weight_path='/home/ssunkim/PycharmProjects/samsung_sem/checkpoints/depth')

    train_loader, test_loader = data_load(want_load)

    # test(net, test_loader, device)

    # ****** Optimizer, Scheduler setup ******
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=lr_min)

    # ---- Training ----
    global_step = 0
    train_loss = []
    for curr_epoch in range(epoch):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))

        for batch_id, data in enumerate(train_loader, 1):
            global_step += 1
            image = data['input'].type('torch.FloatTensor').to(device)
            label = data['label'].type('torch.FloatTensor').to(device)

            net.set_input(image, label)
            net.forward()

            out = net.predict()
            attention_map = net._make_att_map(out)
            loss = net.total_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().detach().numpy().item())
            #
            if global_step % 10 == 0:
                loss_dict = dict()
                loss_dict['epoch'] = curr_epoch
                loss_dict['loss'] = np.mean(train_loss)
                loss_dict['lr'] = optimizer.param_groups[0]['lr']
                log(prefix='train', metrics_dict=loss_dict)

                train_loss = []

            if global_step % 100 == 0:
                log_attention_map(image, attention_map)

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))
        #
        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])

        print('Save network...')
        torch.save({'net': net.state_dict()},'/home/ssunkim/PycharmProjects/samsung_sem/checkpoints/depth' + '/latest_net_lr_{}.pth'.format(lr))

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------------------------------
    want_net = 'sem'
    # [ 'sem' | 'depth' | 'sem_to_depth' ]
    batch_size = 40
    epoch = 200

    lr = 1e-4
    lr_min = 1e-8

    cpu_id = '0'
    continue_train = True
    device = torch.device("cuda:{}".format(cpu_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    import wandb
    wandb.init(project=want_net)
    # ------------------------------------------------------------------------------------------------------------------------------------------
    print('want_net : {}\nbatch_size : {}\nepoch : {}\nlr : {}\ncpu_id : {}\n'.format(want_net, batch_size, epoch, lr, cpu_id))
    # ------------------------------------------------------------------------------------------------------------------------------------------

    if want_net == 'sem':
        # want_load = 'sem_simulation'
        want_load = 'sem_depth_simulation'
        sem(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device)

    elif want_net == 'depth':
        want_load = 'depth_simulation'
        depth(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device)

    elif want_net == 'sem_to_depth':
        want_load = 'train'
        sem_to_depth(batch_size, epoch, lr, lr_min, want_net, want_load, continue_train, device)