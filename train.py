# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import random
from tqdm import tqdm
import yaml
from train_dataloader import get_dataset
from modular_utils import performance_fit, plcc_loss, plcc_rank_loss
from modular_model import modular
from torch.amp import GradScaler


def main(config):
    
    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)

    # stats = pd.read_csv('logs/ViTval.csv')

    for loop in range(opt["split"]):        
        
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
            model = modular.ViTbCLIP_exp(opt["model_path"], opt["model_base"],
                feat_len=opt["feat_len"])
        print('The current model is ' + opt["model"])

        # if config.multi_gpu:
        #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        #     model = model.to(device)
        # else:
        model = model.to(device)
        
        if opt["pretrained_weights"] != None :
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



        # dataloader
        train_loader, val_loader, _ = get_dataset(opt,loop)            

        best_val_criterion = -1  # SROCC min
        SRCC_st = -1
        best_val_b, best_val_s, best_val_t, best_val_st = [], [], [], []

        print('Starting training:')

        scaler = GradScaler()
        for epoch in range(opt["epochs"]):
            print(f'=== Current epoch: {epoch} ===')
            model.train()
            for i, return_list in enumerate(tqdm(train_loader,desc='Training...')):
                for _ in return_list:
                    vid_chunk, vid_chunk_g, tem_feat, tem_feat_g,\
                        spa_feat, spa_feat_g, mos, count, prmt = _
                    # print(mos)
                    label=[]
                    for _ in range(len(mos)):
                        label.append(mos[_].to(device).float())
                    # vid_chunk = vid_chunk.to(device)
                    tem_feat = tem_feat.to(device)
                    spa_feat = spa_feat.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        t, s, a = model(vid_chunk, tem_feat, spa_feat, prmt)
                        if len(mos) == 1:
                            loss = criterion(label[0],(t[3]+s[3])/2)
                        elif len(mos) == 3:
                            loss = criterion(label[0],t[3]) \
                                +criterion(label[1],s[3]) \
                                +criterion(label[2],a[3])
                        else:
                            raise Exception('The number of mos is not correct')
                        loss /= len(mos)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            scheduler.step()
                
            # ======================================
            # do validation after each epoch
            with torch.no_grad():
                model.eval()
                for dataname in opt["dataset"]:
                    valset=val_loader[dataname]
                    Tem_y, Spa_y, Ali_y = [np.zeros([len(valset), 4]) for _ in range(3)]
                    label = np.zeros([len(valset),3])
                    for i, _ in enumerate(tqdm(valset,desc=f"{dataname} validating...")):
                        
                        vid_chunk, vid_chunk_g, tem_feat, tem_feat_g,\
                        spa_feat, spa_feat_g, mos, count, prmt = _[0]
                        for j in range(len(mos)):
                            label[i][j] = mos[j].item()
                        
                        # mid_t stores xt,qt-t,qs-t,qst-t
                        mid_t, mid_s, mid_a = [torch.zeros(4) for _ in range(3)]
                        
                        for j in range(count):
                            x = vid_chunk[:,j,...].to(device)
                            y = tem_feat[:,j,...].to(device)
                            z = spa_feat[:,j,...].to(device)
                            t, s, a = model(x, y, z, prmt)
                            mid_t, mid_s, mid_a = mid_t+t, mid_s+s, mid_a+a
                        mid_t, mid_s, mid_a = mid_t/count, mid_s/count, mid_a/count

                        x = vid_chunk_g.to(device)
                        y = tem_feat_g.to(device)
                        z = spa_feat_g.to(device)
                        t, s, a = model(x, y, z, prmt)
                        Tem_y[i], Spa_y[i], Ali_y[i] = (mid_t + t)/2, (mid_s + s)/2, (mid_a + a)/2


                    if dataname == 'T2VQA':
                        PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
                            label[:,0], (Tem_y[:,3]+Spa_y[:,3])/2)    
                        print('{} final ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                            dataname, SRCC_st, KRCC_st, PLCC_st, RMSE_st))
                        print('===')
                    else:
                        tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_fit(
                            label[:,0], Tem_y[:,3])
                        sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_fit(
                            label[:,1], Spa_y[:,3])
                        aPLCC_st, aSRCC_st, aKRCC_st, aRMSE_st = performance_fit(
                            label[:,2], Ali_y[:,3])
                        print('{} final tem ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                            dataname, tSRCC_st, tKRCC_st, tPLCC_st, tRMSE_st))
                        print('{} final spa ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                            dataname, sSRCC_st, sKRCC_st, sPLCC_st, sRMSE_st))
                        print('{} final ali ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                            dataname, aSRCC_st, aKRCC_st, aPLCC_st, aRMSE_st))
                        print('===')
                        SRCC_st = (tSRCC_st + sSRCC_st + aSRCC_st)/3
                    

            print(f'Epoch {epoch} completed.')
                
        # ===================
        # save model
            if SRCC_st > best_val_criterion:
                best_val_criterion = SRCC_st
                # best_val_st = [SRCC_st,KRCC_st,
                #                 PLCC_st,RMSE_st]
                #                 # aSRCC_st,aKRCC_st,
                #                 # aPLCC_st,aRMSE_st]
                if opt["save_model"] == True:
                    print(f'Save model using {epoch}th/{opt["epochs"]} training result')
                    torch.save(model.state_dict(), f'ckpts/{loop}.pth')
            # print('the best SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            #     best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))

        print('Training completed.')    
        # print('===============BSET tem==============')
        # print(
        #     'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
        #         best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))

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