# -*- coding: utf-8 -*-
import argparse
# import json
import json
import os

import numpy as np
import torch
import torch.optim as optim
# import csv
import torch.nn as nn
import random
from train_dataloader import VideoDataset_train, VideoDataset_val_test
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss

from torchvision import transforms
import time
from model import modular
from torch.amp import GradScaler
# from config import T2V_model


def main(config):

    for loop in range(config.total_loop):
        config.exp_version = loop
        print('%d round training starts here' % loop)
        seed = loop * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout':
            model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=config.feat_len)

        print('The current model is ' + config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout':
            model = model.float()

        if config.trained_model != 'none':
            # load the trained model
            print('loading the pretrained model: ',config.trained_model)
            model.load_state_dict(torch.load(config.trained_model,weights_only=1))

        # optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
        if config.loss_type == 'plcc':
            criterion = plcc_loss
        elif config.loss_type == 'plcc_rank':
            criterion = plcc_rank_loss
        elif config.loss_type == 'L2':
            criterion = nn.MSELoss().to(device)
        elif config.loss_type == 'L1':
            criterion = nn.L1Loss().to(device)
        elif config.loss_type == 'Huberloss':
            criterion = nn.HuberLoss().to(device)

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        # with open('paramLearn.txt','w') as f:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             f.write(f"Parameter Name: {name}\n")
        #             # f.write(f"Values: {param.data}\n")
        #             f.write("\n")
        # json.dump(tem,f,indent=4)

        transformations_train = transforms.Compose(  # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  transforms.Resize(config.resize)
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size), 
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        transformations_vandt = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
            transforms.CenterCrop(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


        # training data
        if config.database == 'FETV':

            spa_feat_dir = 'data/FETV_spatial_all_frames'
            tem_feat_dir = 'data/FETV_temporal_all_frames'
            imgs_dir = 'data/FETV_base_all_frames'
            mosfile = config.mosfile
            print('using the mos file: ',mosfile)
            trainset = VideoDataset_train(imgs_dir, tem_feat_dir, spa_feat_dir, mosfile,
                                          transformations_train, config.crop_size,
                                          prompt_num=config.prompt_num, seed=seed)
            valset = VideoDataset_val_test(imgs_dir, tem_feat_dir, spa_feat_dir, mosfile,
                                        transformations_vandt, 'val', config.crop_size,
                                        prompt_num=config.prompt_num, seed=seed)
        
        
        # dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                # batch_size=1,
                                                   shuffle=True, num_workers=config.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                 shuffle=False, num_workers=config.num_workers)

        best_val_criterion = -1  # SROCC min
        best_val_b, best_val_s, best_val_t, best_val_st = [], [], [], []

        print('Starting training:')

        scaler = GradScaler()
        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (vid_chunk_g, tem_feat_g, spa_feat_g, mos, _) in enumerate(train_loader):
                
                # x=len(list(filter(lambda i: i==0,flag)))
                # print(f'this batch has {x} eles use local dense, total num is {len(flag)}')
                
                optimizer.zero_grad()
                labels = mos.to(device).float()
                vid_chunk_g = vid_chunk_g.to(device)
                tem_feat_g = tem_feat_g.to(device)
                spa_feat_g = spa_feat_g.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs_b, outputs_s, outputs_t, outputs_st = model(
                        vid_chunk_g, tem_feat_g, spa_feat_g)
                    loss = criterion(labels, outputs_st)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(
                        batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                          (epoch + 1, config.epochs, loop + 1, len(trainset) // config.train_batch_size,
                           avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(
                        session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / \
                (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' %
                  (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))
            
        # ======================================
        # do validation after each epoch
            with torch.no_grad():
                model.eval()
                label = np.zeros([len(valset)])
                y_output_b = np.zeros([len(valset)])
                y_output_s = np.zeros([len(valset)])
                y_output_t = np.zeros([len(valset)])
                y_output_st = np.zeros([len(valset)])
                for i, (vid_chunk_g, vid_chunk_l, tem_feat_g, tem_feat_l,
                        spa_feat_g, spa_feat_l, mos, count) in enumerate(val_loader):
                    label[i] = mos.item()
                    outputs_b = outputs_s = outputs_t = outputs_st = 0
                    
                    for j in range(count):
                        vid_chunk_g[j] = vid_chunk_g[j].to(device)
                        tem_feat_g[j] = tem_feat_g[j].to(device)
                        spa_feat_g[j] = spa_feat_g[j].to(device)
                        b, s, t, st = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j])
                        outputs_b += b
                        outputs_s += s
                        outputs_t += t
                        outputs_st += st
                    count = count.to(device)
                    outputs_b, outputs_s, outputs_t, outputs_st = \
                        outputs_b/count, outputs_s/count, outputs_t/count, outputs_st/count

                    vid_chunk_l = vid_chunk_l.to(device)
                    tem_feat_l = tem_feat_l.to(device)
                    spa_feat_l = spa_feat_l.to(device)
                    b1, s1, t1, st1 = model(vid_chunk_l, tem_feat_l, spa_feat_l)
                    outputs_b = (outputs_b + b1) / 2
                    outputs_s = (outputs_s + s1) / 2
                    outputs_t = (outputs_t + t1) / 2
                    outputs_st = (outputs_st + st1) / 2

                    y_output_b[i] = outputs_b.item()
                    y_output_s[i] = outputs_s.item()
                    y_output_t[i] = outputs_t.item()
                    y_output_st[i] = outputs_st.item()

                val_PLCC_b, val_SRCC_b, val_KRCC_b, val_RMSE_b = performance_fit(label, y_output_b)
                val_PLCC_s, val_SRCC_s, val_KRCC_s, val_RMSE_s = performance_fit(label, y_output_s)
                val_PLCC_t, val_SRCC_t, val_KRCC_t, val_RMSE_t = performance_fit(label, y_output_t)
                val_PLCC_st, val_SRCC_st, val_KRCC_st, val_RMSE_st = performance_fit(label, y_output_st)

                print(
                    'Epoch {} completed. The result on the base validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        val_SRCC_b, val_KRCC_b, val_PLCC_b, val_RMSE_b))
                print(
                    'Epoch {} completed. The result on the S validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        val_SRCC_s, val_KRCC_s, val_PLCC_s, val_RMSE_s))
                print(
                    'Epoch {} completed. The result on the T validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        val_SRCC_t, val_KRCC_t, val_PLCC_t, val_RMSE_t))
                print(
                    'Epoch {} completed. The result on the ST validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,
                        val_SRCC_st, val_KRCC_st, val_PLCC_st, val_RMSE_st))

        # ===================
        # save model
                if val_SRCC_st > best_val_criterion:
                    print(
                        "Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = val_SRCC_st
                    best_val_b = [val_SRCC_b, val_KRCC_b,
                                  val_PLCC_b, val_RMSE_b]
                    best_val_s = [val_SRCC_s, val_KRCC_s,
                                  val_PLCC_s, val_RMSE_s]
                    best_val_t = [val_SRCC_t, val_KRCC_t,
                                  val_PLCC_t, val_RMSE_t]
                    best_val_st = [val_SRCC_st, val_KRCC_st,
                                   val_PLCC_st, val_RMSE_st]

                    print('Saving model...')
                    torch.save(model.state_dict(), f'ckpts_modular/{loop}.pth')

        print('Training completed.')

        print(
            'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_b[0], best_val_b[1], best_val_b[2], best_val_b[3]))
       
        print(
            'The best training result on the S validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_s[0], best_val_s[1], best_val_s[2], best_val_s[3]))
      
        print(
            'The best training result on the T validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_t[0], best_val_t[1], best_val_t[2], best_val_t[3]))

        print(
            'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))

        data = {"b":best_val_b,
                "s":best_val_s,
                "t":best_val_t,
                "st":best_val_st}

        with open(f'logs/log{loop}.json','w') as f:
            json.dump(data,f,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporal_dropout')
    parser.add_argument('--feat_len', type=int, default=8)
    parser.add_argument('--total_loop', type=int, default=5)
    parser.add_argument('--prompt_num', type=int, default=619)
    # training parameters

    # original 1e-5
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)

    # misc
    # parser.add_argument('--t2vmodel',type=str,default=['cogvideo',''])
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='plcc')
    parser.add_argument('--trained_model', type=str, default='none')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--mosfile',type=str,default='data/pyIQA_FETV_score/mosFile/temAVGmos.json')

    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
