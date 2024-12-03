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
from tqdm import tqdm
import yaml
from train_dataloader import Dataset_1mos,get_dataset
from modular_utils import performance_fit, performance_no_fit
from modular_utils import plcc_loss, plcc_rank_loss

from torchvision import transforms
import time
from modular_model import modular
from torch.amp import GradScaler


def main(config):
    
    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)

    stats = pd.read_csv('logs/ViTval.csv')

    for loop in range(opt["split"]):        
        if loop == 0 :
            continue
        
        print('the %dth round training starts here' % (loop) )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if opt["model"] == 'aveScore':
            model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=opt["feat_len"])
        elif opt["model"] == 'aveFeat':
            model = modular.ViTbCLIP_SpatialTemporal_dropout_meanpool(feat_len=opt["feat_len"])
        elif opt["model"] == 'hybrid':
            model = modular.ViTbCLIP_SpatialTemporal_dropout_hybrid(feat_len=opt["feat_len"])
        elif opt["model"] == 'old':
            model = modular.ViTbCLIP_SpatialTemporal_dropout_old(feat_len=opt["feat_len"])
        elif opt["model"] == 'exp':
            model = modular.ViTbCLIP_exp(feat_len=opt["feat_len"])
        print('The current model is ' + opt["model"])

        # if config.multi_gpu:
        #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        #     model = model.to(device)
        # else:
        model = model.to(device).float()
        
        if opt["pretrained_weights"] != 'none':
            print('loading the pretrained model from ', opt["pretrained_weights"])
            model.load_state_dict(torch.load(opt["pretrained_weights"], weights_only=1))

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=0.0000001)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt["decay_interval"], gamma=opt["decay_ratio"])
        if opt["loss_type"] == 'plcc':
            criterion = plcc_loss
        elif opt["loss_type"] == 'plcc_rank':
            criterion = plcc_rank_loss
        elif opt["loss_type"] == 'L2':
            criterion = nn.MSELoss().to(device)
        elif opt["loss_type"] == 'L1':
            criterion = nn.L1Loss().to(device)
        elif opt["loss_type"] == 'Huberloss':
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



        # dataloader
        print('using the mos file: ', opt["dataset"]["T2VQA"]["mos_file"])        
        trainset, valset, _ = get_dataset(opt["dataset"],loop)
            
        
        # trainset = Dataset_1mos(opt["dataset"]["T2VQA"], 'train', opt["dataset"]["T2VQA"]["vids_dir"], 
        #                         opt["dataset"]["T2VQA"]["tem_feat_dir"], opt["dataset"]["T2VQA"]["spa_feat_dir"], 
        #                                     opt["dataset"]["T2VQA"]["mos_file"], transformations_train,
        #                                    prompt_num = opt["dataset"]["T2VQA"]["prompt_num"], seed=loop)
        # valset = Dataset_1mos(opt["dataset"]["T2VQA"], 'val', opt["dataset"]["T2VQA"]["vids_dir"], 
        #                       opt["dataset"]["T2VQA"]["tem_feat_dir"], opt["dataset"]["T2VQA"]["spa_feat_dir"], 
        #                                     opt["dataset"]["T2VQA"]["mos_file"], transformations_vandt,
        #                                    prompt_num = opt["dataset"]["T2VQA"]["prompt_num"], seed=loop)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt["train_batch_size"],
                                                   shuffle=True, num_workers=opt["num_workers"], drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                 shuffle=False, num_workers=opt["num_workers"])

        best_val_criterion = -1  # SROCC min
        best_val_b, best_val_s, best_val_t, best_val_st = [], [], [], []

        print('Starting training:')

        scaler = GradScaler()
        for epoch in range(opt["epochs"]):
            print(f'=== Current epoch: {epoch} ===')
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            # for i, (vid_chunk, vid_chunk_g, tem_feat, tem_feat_g,\
                    # spa_feat, spa_feat_g, mos,count, prmt) in enumerate(tqdm(train_loader,desc='Training...')):
            for i, return_list in enumerate(tqdm(train_loader,desc='Training...')):
                optimizer.zero_grad()
                print(len(return_list))
                label=[]
                for _ in range(len(mos)):
                    label.append(mos[_].to(device).float())
                vid_chunk = vid_chunk.to(device)
                tem_feat = tem_feat.to(device)
                spa_feat = spa_feat.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):

                    if opt["model"] == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                        t, s, a, tm, sm, am = model(vid_chunk, tem_feat, spa_feat, prmt)
                        t = (t+tm)/2
                        s = (s+sm)/2
                        a = (a+am)/2
                    else:
                        t, s, a = model(vid_chunk, tem_feat, spa_feat, prmt)
                    
                    loss = criterion(label[0],(t[3]+s[3])/2)
                    
                    # loss = criterion(label[0],t[3]) \
                    #     +criterion(label[1],s[3]) \
                    #     +criterion(label[2],a[3])
                    loss /= len(mos)

                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            avg_loss = sum(batch_losses) / (len(trainset) // opt["train_batch_size"])
            print('Epoch %d averaged training loss: %.4f' %(epoch + 1, avg_loss))
            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))
                # if (i + 1) % (opt["print_samples"] // opt["train_batch_size"]) == 0:
                #     session_end_time = time.time()
                #     avg_loss_epoch = sum(batch_losses_each_disp) / (opt["print_samples"]  // opt["train_batch_size"])
                #     print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                #           (epoch + 1, opt["epochs"], i + 1, len(trainset)// opt["train_batch_size"],
                #            avg_loss_epoch))
                #     batch_losses_each_disp = []
                #     print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                #     session_start_time = time.time()
                
        # ======================================
        # do validation after each epoch
            with torch.no_grad():
                model.eval()
                label = np.zeros([len(valset),3])
                Tem_y, Spa_y, Ali_y = [np.zeros([len(valset), 4]) for _ in range(3)]
                
                for i, (vid_chunk, vid_chunk_g, tem_feat, tem_feat_g,\
                    spa_feat, spa_feat_g, mos,count, prmt) in enumerate(tqdm(val_loader, desc='Validating...')):

                    for j in range(len(mos)):
                        label[i][j] = mos[j].item()
                    
                    # mid_t stores xt,qt-t,qs-t,qst-t
                    mid_t, mid_s, mid_a = [torch.zeros(4) for _ in range(3)]
                    
                    for j in range(count):
                        vid_chunk[j] = vid_chunk[j].to(device)
                        tem_feat[j] = tem_feat[j].to(device)
                        spa_feat[j] = spa_feat[j].to(device)
                        if opt["model"] == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                            t, s, a, tm, sm, am = model(vid_chunk[j], tem_feat[j], spa_feat[j], prmt)
                            t = (t + tm) / 2
                            s = (s + sm) / 2
                            a = (a + am) / 2
                        else:
                            t, s, a = model(vid_chunk[j], tem_feat[j], spa_feat[j], prmt)
                        mid_t, mid_s, mid_a = mid_t+t, mid_s+s, mid_a+a
                    mid_t, mid_s, mid_a = mid_t/count, mid_s/count, mid_a/count

                    vid_chunk_g = vid_chunk_g.to(device)
                    tem_feat_g = tem_feat_g.to(device)
                    spa_feat_g = spa_feat_g.to(device)

                    if opt["model"] == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                        t, s, a, tm, sm, am = model(vid_chunk_g, tem_feat_g, spa_feat_g, prmt)
                        t = (t + tm) / 2
                        s = (s + sm) / 2
                        a = (a + am) / 2
                    else:
                        t, s, a = model(vid_chunk_g, tem_feat_g, spa_feat_g, prmt)
                    Tem_y[i], Spa_y[i], Ali_y[i] = (mid_t + t)/2, (mid_s + s)/2, (mid_a + a)/2

                # tPLCC_b, tSRCC_b, tKRCC_b, tRMSE_b = performance_fit(
                #     label[:,0], Tem_y[:,0])
                # tPLCC_t, tSRCC_t, tKRCC_t, tRMSE_t = performance_fit(
                #     label[:,0], Tem_y[:,1])
                # tPLCC_s, tSRCC_s, tKRCC_s, tRMSE_s = performance_fit(
                #     label[:,0], Tem_y[:,2])
                # tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_fit(
                #     label[:,0], Tem_y[:,3])
                
                # sPLCC_b, sSRCC_b, sKRCC_b, sRMSE_b = performance_fit(
                #     label[:,0], Spa_y[:,0])
                # sPLCC_t, sSRCC_t, sKRCC_t, sRMSE_t = performance_fit(
                #     label[:,0], Spa_y[:,1])
                # sPLCC_s, sSRCC_s, sKRCC_s, sRMSE_s = performance_fit(
                #     label[:,0], Spa_y[:,2])
                # sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_fit(
                #     label[:,0], Spa_y[:,3])
                
                # aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
                #     label[:,0], Ali_y[:,0])
                # aPLCC_t, aSRCC_t, aKRCC_t, aRMSE_t = performance_fit(
                #     label[:,0], Ali_y[:,1])
                # aPLCC_s, aSRCC_s, aKRCC_s, aRMSE_s = performance_fit(
                #     label[:,0], Ali_y[:,2])
                # aPLCC_st, aSRCC_st, aKRCC_st, aRMSE_st = performance_fit(
                #     label[:,0], Ali_y[:,3])

                PLCC_b, SRCC_b, KRCC_b, RMSE_b = performance_fit(
                    label[:,0], (Tem_y[:,0]+Spa_y[:,0])/2)
                PLCC_t, SRCC_t, KRCC_t, RMSE_t = performance_fit(
                    label[:,0], (Tem_y[:,1]+Spa_y[:,1])/2)
                PLCC_s, SRCC_s, KRCC_s, RMSE_s = performance_fit(
                    label[:,0], (Tem_y[:,2]+Spa_y[:,2])/2)
                PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
                    label[:,0], (Tem_y[:,3]+Spa_y[:,3])/2)    

                
                # stats.loc[len(stats)]=[tSRCC_st,sSRCC_st,aSRCC_st]
                         
                # print('===============Tem==============')
                # print(
                #     'Epoch {} completed. base val: SRCC: {:.4f}'.format(epoch + 1,tSRCC_b))
                # print(
                #     'Epoch {} completed. S val: SRCC: {:.4f}'.format(epoch + 1,tSRCC_s))
                # print(
                #     'Epoch {} completed. T val: SRCC: {:.4f}'.format(epoch + 1,tSRCC_t))
                # print(
                #     'Epoch {} completed. ST val: SRCC: {:.4f}'.format(epoch + 1,tSRCC_st))
                # print('===============Spa==============')
                print(f'Epoch {epoch+1} completed.')
                # print('===')
                print('base: {:.4f}\nT: {:.4f}\nS: {:.4f}\nST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                    SRCC_b ,SRCC_t ,SRCC_s, SRCC_st, KRCC_st, PLCC_st, RMSE_st))
                print('===')
                # print('S val: SRCC: {:.4f}'.format( epoch + 1,sSRCC_s))
                # print(
                #     'Epoch {} completed. T val: SRCC: {:.4f}'.format(epoch + 1,sSRCC_t))
                # print(
                #     'Epoch {} completed. ST val: SRCC: {:.4f}'.format(epoch + 1,sSRCC_st))

                # print('===============Ali==============')
                # print(
                #     'Epoch {} completed. base val: SRCC: {:.4f}'.format(epoch + 1,aSRCC_b))
                # print(
                #     'Epoch {} completed. S val: SRCC: {:.4f}'.format( epoch + 1,aSRCC_s))
                # print(
                #     'Epoch {} completed. T val: SRCC: {:.4f}'.format(epoch + 1,aSRCC_t))
                # print(
                #     'Epoch {} completed. ST val: SRCC: {:.4f}'.format(epoch + 1,aSRCC_st))
                    
                
                
        # ===================
        # save model
                if SRCC_st > best_val_criterion:
                    best_val_criterion = SRCC_st
                    best_val_st = [SRCC_st,KRCC_st,
                                    PLCC_st,RMSE_st]
                                    # aSRCC_st,aKRCC_st,
                                    # aPLCC_st,aRMSE_st]

                    if opt["save_model"] == True:
                        print(f'Save model using {epoch+1}th/{opt["epochs"]} training result')
                        torch.save(model.state_dict(), f'ckpts/{loop}.pth')
                print('the best SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))
                
                
            

            
            
            
        print('Training completed.')    
        # print('===============BSET tem==============')
        print(
            'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))

        # print('===============BSET spa==============')
        # print(
        #     'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
        #         best_val_st[4], best_val_st[5], best_val_st[6], best_val_st[7]))

        # print('===============BSET ali==============')
        # print(
        #     'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
        #         best_val_st[8], best_val_st[9], best_val_st[10], best_val_st[11]))
        
        
        # if config.save_model:
        #     stats.loc[len(stats)]=[0,0,0]
        #     stats.to_csv('logs/ViTval.csv',index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # # input parameters
    # parser.add_argument('--database', type=str, default='LGVQ')
    # parser.add_argument('--model_name', type=str,
    #                     default='aveScore')
    # parser.add_argument('--feat_len', type=int, default=8)
    # parser.add_argument('--total_loop', type=int, default=10)
    # # training parameters

    # # original 1e-5
    # parser.add_argument('--lr', type=float, default=5e-6)
    # parser.add_argument('--decay_ratio', type=float, default=0.9)
    # parser.add_argument('--decay_interval', type=int, default=2)
    # # parser.add_argument('--n_trial', type=int, default=0)
    # # parser.add_argument('--results_path', type=str)
    # # parser.add_argument('--exp_version', type=int)
    # parser.add_argument('--print_samples', type=int, default=2000)
    # parser.add_argument('--train_batch_size', type=int, default=8)
    # parser.add_argument('--num_workers', type=int, default=16)
    # parser.add_argument('--resize', type=int, default=256)
    # parser.add_argument('--crop_size', type=int, default=224)
    # parser.add_argument('--epochs', type=int, default=30)
    # # misc
    # # parser.add_argument('--ckpt_path', type=str, default=None)
    # # parser.add_argument('--multi_gpu', action='store_true')
    # # parser.add_argument('--gpu_ids', type=list, default=None)
    # parser.add_argument('--loss_type', type=str, default='plcc')
    # parser.add_argument('--trained_model', type=str, default='none')
    # parser.add_argument('--save_model',action='store_true')
    # parser.add_argument('--mosfile', type=str,
    #                     default='/home/user/Documents/vqadata/BVQAdata/T2VQA.csv')


    parser.add_argument(
        "-o", "--opt", type=str, default="./cfg.yml", help="the option file"
    )
    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)
