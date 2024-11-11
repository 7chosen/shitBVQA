# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import random
from train_dataloader import viCLIP_trainDT,viCLIP_vandtDT
from modular_utils import performance_fit
from modular_utils import plcc_loss, plcc_rank_loss
from torchvision import transforms
import time
from model import modular
from torch.amp import GradScaler
# from config import T2V_model
# from transformers import AutoModel
from ViCLIP_models.viclip import ViCLIP


def main(config):
    
    static = pd.read_csv('logs/static.csv')
    
    for loop in range(config.total_loop):
        config.exp_version = loop
        print('the %dth round training starts here' % (loop) )
        seed = loop 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout':
            model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=config.feat_len)
        else:
            model = ViCLIP()


        print('The current model is ' + config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        # model = model.float()

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

        # model.clip.logit_scale.requires_grad = False

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
            trainset = viCLIP_trainDT(imgs_dir, tem_feat_dir, spa_feat_dir, mosfile,
                                          transformations_train, config.crop_size,
                                          prompt_num=config.prompt_num, seed=seed)
            valset = viCLIP_vandtDT(imgs_dir, mosfile,'val', config.crop_size,
                                           prompt_num=config.prompt_num, seed=seed)

        # dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   # batch_size=1,
                                                   shuffle=True, num_workers=config.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                 shuffle=False, num_workers=config.num_workers)

        best_val_criterion = -1  # SROCC min
        # best_val_b, best_val_s, best_val_t, best_val_st = [], [], [], []

        print('Starting training:')

        scaler = GradScaler()
        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (v_l, prmt, mos) in enumerate(train_loader):

                optimizer.zero_grad()
                label=[]
                for l in range(len(mos)):
                    label.append(mos[l].to(device).float())
                v_l=v_l.to(device)                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    score = model(image=v_l,raw_text=prmt,idx=0,return_sims=True)
                    # print(score.shape)
                    loss = criterion(label[2],score) 
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
                label = np.zeros([len(valset),3])
                Ali_y = np.zeros(len(valset))
                
                for i, (v_l, v_g, prmt, mos, count) in enumerate(val_loader):
                    for j in range(len(mos)):
                        label[i][j] = mos[j].item()
                    score=0

                    for j in range(count):
                        v_l[j] = v_l[j].to(device)
                        score += model(image=v_l[j],raw_text=prmt,idx=0,return_sims=True)
                    count=count.to(device)
                    score/=count

                    v_g = v_g.to(device)
                    score += model(image=v_g,raw_text=prmt,idx=0,return_sims=True)
                    score/=2
                    score=score.to('cpu')
                    Ali_y[i]=score

                
                aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
                    label[:,2], Ali_y)
                
                new_row=[aSRCC_b, aKRCC_b, aPLCC_b, aRMSE_b]
                static.loc[len(static)]=new_row
                         

                print('Epoch {} completed. base val: SRCC: {:.4f}'.format(epoch + 1,aSRCC_b))
                    
                
                
        # ===================
        # save model
                if aSRCC_b > best_val_criterion:
                    print(
                        "Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = aSRCC_b
                    best_val_b = [aSRCC_b, aKRCC_b,
                                  aPLCC_b, aRMSE_b]

                    print('Saving model...')
                    torch.save(model.state_dict(), f'ckpts_modular/{loop}_{epoch}.pth')

        print('Training completed.')    
        print(
            'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_b[0], best_val_b[1], best_val_b[2], best_val_b[3 ]))
        static.to_csv('logs/static.csv',index=False)

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
                        default='viCLIP')
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
