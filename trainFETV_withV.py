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
        print('the %dth round training starts here' % (loop) )
        seed = loop 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout':
            model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=config.feat_len)

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
            print('loading the pretrained model: ', config.trained_model)
            model.load_state_dict(torch.load(
                config.trained_model, weights_only=1))

        # optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=0.0000001)

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

        model.clip.logit_scale.requires_grad = False

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

        transformations_train = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomCrop(config.crop_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        transformations_vandt = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(config.crop_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

        # training data
        if config.database == 'FETV':

            spa_feat_dir = 'data/FETV_spatial_all_frames'
            tem_feat_dir = 'data/FETV_temporal_all_frames'
            imgs_dir = 'data/FETV_base_all_frames'
            mosfile = config.mosfile
            print('using the mos file: ', mosfile)
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
            for i, (vid_chunk, tem_feat, spa_feat, prmt, mos) in enumerate(train_loader):

                optimizer.zero_grad()
                label=[]
                for i in range(len(mos)):
                    label.append(mos[i].to(device).float())
                    
                # label = mos.to(device).float()
                # labelt = mos[1].to(device).float()
                
                vid_chunk = vid_chunk.to(device)
                tem_feat = tem_feat.to(device)
                spa_feat = spa_feat.to(device)
                # prmt.to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, _, _, sta = model(vid_chunk, tem_feat, spa_feat, prmt)
                    # loss = criterion(labels, outputs_st)
                    # loss_t= criterion(labelt[0], st[0])
                    # loss_s= criterion(label[1], st[1])
                    loss = 0
                    for i in range(len(mos)):
                        loss += criterion(label[i],sta[i])
                    loss /= len(mos)

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
                labelt = np.zeros([len(valset)])
                labels = np.zeros([len(valset)])
                Tem_y_b = np.zeros([len(valset)])
                Tem_y_s = np.zeros([len(valset)])
                Tem_y_t = np.zeros([len(valset)])
                Tem_y_st = np.zeros([len(valset)])
                Spa_y_b = np.zeros([len(valset)])
                Spa_y_s = np.zeros([len(valset)])
                Spa_y_t = np.zeros([len(valset)])
                Spa_y_st = np.zeros([len(valset)])
                for i, (vid_chunk, vid_chunk_l, tem_feat, tem_feat_l,
                        spa_feat, spa_feat_l, t_mos, s_mos,count) in enumerate(val_loader):
                    labelt[i] = t_mos.item()
                    labels[i] = s_mos.item()
                    tmidb = tmids = tmidt = tmidst = 0
                    smidb = smids = smidt = smidst = 0
                    


                    for j in range(count):
                        vid_chunk[j] = vid_chunk[j].to(device)
                        tem_feat[j] = tem_feat[j].to(device)
                        spa_feat[j] = spa_feat[j].to(device)
                        b, s, t, st = model(vid_chunk[j], tem_feat[j], spa_feat[j])

                        smidb += b[0]
                        smids += s[0]
                        smidt += t[0]
                        smidst += st[0]
                        tmidb += b[1]
                        tmids += s[1]
                        tmidt += t[1]
                        tmidst += st[1]

                    count=count.to(device)

                    tmidb, tmids, tmidt, tmidst = tmidb/count, tmids/count, tmidt/count, tmidst/count
                    smidb, smids, smidt, smidst = smidb/count, smids/count, smidt/count, smidst/count

                    vid_chunk_l = vid_chunk_l.to(device)
                    tem_feat_l = tem_feat_l.to(device)
                    spa_feat_l = spa_feat_l.to(device)
                    b1, s1, t1, st1 = model(
                        vid_chunk_l, tem_feat_l, spa_feat_l)

                    tmidb = (tmidb + b1[1]) / 2
                    tmids = (tmids + s1[1]) / 2
                    tmidt = (tmidt + t1[1]) / 2
                    tmidst = (tmidst + st1[1]) / 2
                    smidb = (smidb + b1[0]) / 2
                    smids = (smids + s1[0]) / 2
                    smidt = (smidt + t1[0]) / 2
                    smidst = (smidst + st1[0]) / 2

                    Tem_y_b[i] = tmidb.item()
                    Tem_y_s[i] = tmids.item()
                    Tem_y_t[i] = tmidt.item()
                    Tem_y_st[i] = tmidst.item()
                    Spa_y_b[i] = smidb.item()
                    Spa_y_s[i] = smids.item()
                    Spa_y_t[i] = smidt.item()
                    Spa_y_st[i] = smidst.item()

                tPLCC_b, tSRCC_b, tKRCC_b, tRMSE_b = performance_fit(
                    labelt, Tem_y_b)
                tPLCC_s, tSRCC_s, tKRCC_s, tRMSE_s = performance_fit(
                    labelt, Tem_y_s)
                tPLCC_t, tSRCC_t, tKRCC_t, tRMSE_t = performance_fit(
                    labelt, Tem_y_t)
                tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_fit(
                    labelt, Tem_y_st)
                sPLCC_b, sSRCC_b, sKRCC_b, sRMSE_b = performance_fit(
                    labels, Spa_y_b)
                sPLCC_s, sSRCC_s, sKRCC_s, sRMSE_s = performance_fit(
                    labels, Spa_y_s)
                sPLCC_t, sSRCC_t, sKRCC_t, sRMSE_t = performance_fit(
                    labels, Spa_y_t)
                sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_fit(
                    labels, Spa_y_st)

                print('===============Tem==============')
                print(
                    'Epoch {} completed. The result on the base validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,tSRCC_b, tKRCC_b, tPLCC_b, tRMSE_b))
                print(
                    'Epoch {} completed. The result on the S validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,tSRCC_s, tKRCC_s, tPLCC_s, tRMSE_s))
                print(
                    'Epoch {} completed. The result on the T validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,tSRCC_t, tKRCC_t, tPLCC_t, tRMSE_t))
                print(
                    'Epoch {} completed. The result on the ST validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,tSRCC_st, tKRCC_st, tPLCC_st, tRMSE_st))
                print('===============Spa==============')
                print(
                    'Epoch {} completed. The result on the base validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,sSRCC_b, sKRCC_b, sPLCC_b, sRMSE_b))
                print(
                    'Epoch {} completed. The result on the S validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,sSRCC_s, sKRCC_s, sPLCC_s, sRMSE_s))
                print(
                    'Epoch {} completed. The result on the T validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,sSRCC_t, sKRCC_t, sPLCC_t, sRMSE_t))
                print(
                    'Epoch {} completed. The result on the ST validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1,sSRCC_st, sKRCC_st, sPLCC_st, sRMSE_st))
        # ===================
        # save model
                if (tSRCC_st+sSRCC_st)/2 > best_val_criterion:
                    print(
                        "Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = (tSRCC_st+sSRCC_st)/2
                    best_val_b = [tSRCC_b, tKRCC_b,
                                  tPLCC_b, tRMSE_b,
                                  sSRCC_b,sKRCC_b,
                                  sPLCC_b,sRMSE_b]
                    best_val_s = [tSRCC_s, tKRCC_s,
                                  tPLCC_s, tRMSE_s,
                                  sSRCC_s,sKRCC_s,
                                  sPLCC_s,sRMSE_s]
                    best_val_t = [tSRCC_t, tKRCC_t,
                                  tPLCC_t, tRMSE_t,
                                  sSRCC_t,sKRCC_t,
                                  sPLCC_t,sRMSE_t]
                    best_val_st = [tSRCC_st, tKRCC_st,
                                   tPLCC_st, tRMSE_st,
                                   sSRCC_st,sKRCC_st,
                                    sPLCC_st,sRMSE_st]

                    print('Saving model...')
                    torch.save(model.state_dict(), f'ckpts_modular/{loop}_{epoch}.pth')

        print('Training completed.')    
        print('===============BSET tem==============')

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

        print('===============BSET spa==============')
        print(
            'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_b[4], best_val_b[5], best_val_b[6], best_val_b[7]))

        print(
            'The best training result on the S validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_s[4], best_val_s[5], best_val_s[6], best_val_s[7]))

        print(
            'The best training result on the T validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_t[4], best_val_t[5], best_val_t[6], best_val_t[7]))

        print(
            'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_st[4], best_val_st[5], best_val_st[6], best_val_st[7]))
        # data = {"b":best_val_b,
        #         "s":best_val_s,
        #         "t":best_val_t,
        #         "st":best_val_st}

        # with open(f'logs/log{loop}.json','w') as f:
        #     json.dump(data,f,indent=4)


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
    parser.add_argument('--lr', type=float, default=1e-5)
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
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='plcc')
    parser.add_argument('--trained_model', type=str, default='none')
    # parser.add_argument('--save_path', type=str)
    parser.add_argument('--mosfile', type=str,
                        default='data/FETV.csv')

    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
