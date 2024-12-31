import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn
from torchvision import transforms
from tqdm import tqdm
import yaml
from modular_model import modular
from modular_utils import performance_fit, performance_no_fit
from train_dataloader import get_dataset
from ViCLIP_models.viclip import ViCLIP


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)
    # stats = pd.read_csv('logs/ViTtest.csv')

    for loop in range(opt["split"]):
        if loop!=0:
            break
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
        print('The current model is: ' + opt["model"])
        
        # if config.multi_gpu:
        #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        #     model = model.to(device)
        # else:

        # load the trained model
        if opt["pretrained_weights"] != None :
            print('loading the pretrained model from ', opt["pretrained_weights"])
            # model.load_state_dict(torch.load(opt["pretrained_weights"]))
            model.load_state_dict(torch.load(f"./ckpts/{opt["model"]}_{loop}"))


        model = model.to(device)
        train_loader, val_loader,  test_loader = get_dataset(opt,loop)   


        with torch.no_grad():
            model.eval()
            for dataname in opt["dataset"]:
                testset=test_loader[dataname]
                Tem_y, Spa_y, Ali_y = [np.zeros([len(testset), 4]) for _ in range(3)]
                label = np.zeros([len(testset),3])
                for i, _ in enumerate(tqdm(testset,desc=f"{dataname} testing...")):
                    
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
                    KRCC_st = (tKRCC_st + sKRCC_st + aKRCC_st)/3
                    PLCC_st = (tPLCC_st + sPLCC_st + aPLCC_st)/3
                    RMSE_st = (tRMSE_st + sRMSE_st + aRMSE_st)/3
                    print('{} final ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                        dataname, SRCC_st, KRCC_st, PLCC_st, RMSE_st))
                    print('===')
                    

            # tPLCC_b, tSRCC_b, tKRCC_b, tRMSE_b = performance_fit(
            #     label[:,0], Tem_y[:,0])
            # tPLCC_t, tSRCC_t, tKRCC_t, tRMSE_t = performance_fit(
            #     label[:,0], Tem_y[:,1])
            # tPLCC_s, tSRCC_s, tKRCC_s, tRMSE_s = performance_fit(
            #     label[:,0], Tem_y[:,2])
            # tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_no_fit(
            #     label[:,0], Tem_y[:,3])
            
            # sPLCC_b, sSRCC_b, sKRCC_b, sRMSE_b = performance_fit(
            #     label[:,1], Spa_y[:,0])
            # sPLCC_t, sSRCC_t, sKRCC_t, sRMSE_t = performance_fit(
            #     label[:,1], Spa_y[:,1])
            # sPLCC_s, sSRCC_s, sKRCC_s, sRMSE_s = performance_fit(
            #     label[:,1], Spa_y[:,2])
            # sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_no_fit(
            #     label[:,1], Spa_y[:,3])
            
            # aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
            #     label[:,2], Ali_y[:,0])
            # aPLCC_t, aSRCC_t, aKRCC_t, aRMSE_t = performance_fit(
            #     label[:,2], Ali_y[:,1])
            # aPLCC_s, aSRCC_s, aKRCC_s, aRMSE_s = performance_fit(
            #     label[:,2], Ali_y[:,2])
            # aPLCC_st, aSRCC_st, aKRCC_st, aRMSE_st = performance_no_fit(
            #     label[:,2], Ali_y[:,3])    

            # PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
            #     label[:,0], (Tem_y[:,3]+Spa_y[:,3])/2) 

            # new_row=[
            #         tSRCC_st,tKRCC_st, tPLCC_st, tRMSE_st,
            #         sSRCC_st,sKRCC_st, sPLCC_st, sRMSE_st,
            #         aSRCC_st,aKRCC_st, aPLCC_st, aRMSE_st
            #         ]
            # stats.loc[len(stats)]=new_row

            # print('===============Tem==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(tSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(tSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(tSRCC_s))
            # print(
            #     'tem ST test: SRCC: {:.4f}'.format(tSRCC_st))
            # print('===============Spa==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(sSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(sSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(sSRCC_s))

            # print(
            #     'spa ST test: SRCC: {:.4f}'.format(sSRCC_st))

            # print('===============Ali==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(aSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(aSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(aSRCC_s))
            # print(
            #     'ST test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}'
            #     .format(aSRCC_st,aKRCC_st,aPLCC_st,aRMSE_st))
            
            # print('ST test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}'
            #     .format(SRCC_st,KRCC_st,PLCC_st,RMSE_st))

        # new_row=[0,0,0,0,
        #             0,0,0,0,
        #             0,0,0,0] 
        # stats.loc[len(stats)]=new_row 
        # stats.to_csv('logs/ViTtest.csv',index=False)        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="./cfg.yml", help="the option file")
    config = parser.parse_args()
    main(config)
