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
from train_dataloader import VideoDataset_val_test
from ViCLIP_models.viclip import ViCLIP


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)
    # stats = pd.read_csv('logs/ViTtest.csv')

    for loop in range(opt["split"]):
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
        else:
            model = ViCLIP()
        print('The current model is: ' + opt["model"])
        
        # if config.multi_gpu:
        #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        #     model = model.to(device)
        # else:
        model = model.to(device).float()

        # load the trained model
        ckpt_path=f'ckpts/{loop}.pth'
        print(f'Loading the trained model: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, weights_only=1))

        transformations_vandt = transforms.Compose(
            [transforms.Resize(opt["resize"], interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(opt["crop_size"]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


        # training data
        print('Using the mos file: ', opt["mos_file"])
        testset = VideoDataset_val_test(opt["database"], opt["imgs_dir"], opt["tem_feat_dir"], opt["spa_feat_dir"], 
                                            opt["mos_file"],transformations_vandt, 'test', opt["crop_size"],
                                           prompt_num = opt["prompt_num"], seed=loop)


        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=opt["num_workers"])

        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset),3])
            Tem_y, Spa_y, Ali_y = [np.zeros([len(testset), 4]) for _ in range(3)]
        
            for i, (vid_chunk_g, vid_chunk_l, tem_feat_g, tem_feat_l,
                    spa_feat_g, spa_feat_l, mos,count,prmt) in enumerate(tqdm(test_loader, leave=True)):
                for j in range(len(mos)):
                    label[i][j] = mos[j].item()
                mid_t, mid_s, mid_a = [torch.zeros(4) for _ in range(3)]
                
                for j in range(count):
                    vid_chunk_g[j] = vid_chunk_g[j].to(device)
                    tem_feat_g[j] = tem_feat_g[j].to(device)
                    spa_feat_g[j] = spa_feat_g[j].to(device)
                    if opt["model"] == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                        t, s, a, tm, sm, am = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j], prmt)
                        t = (t + tm) / 2
                        s = (s + sm) / 2
                        a = (a + am) / 2
                    else:
                        t, s, a = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j], prmt)
                    mid_t, mid_s, mid_a = mid_t+t, mid_s+s, mid_a+a
                mid_t, mid_s, mid_a = mid_t/count, mid_s/count, mid_a/count
                vid_chunk_l = vid_chunk_l.to(device)
                tem_feat_l = tem_feat_l.to(device)
                spa_feat_l = spa_feat_l.to(device)
                if opt["model"] == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                    t, s, a, tm, sm, am = model(vid_chunk_l, tem_feat_l, spa_feat_l, prmt)
                    t = (t + tm) / 2
                    s = (s + sm) / 2
                    a = (a + am) / 2
                else:
                    t, s, a = model(vid_chunk_l, tem_feat_l, spa_feat_l, prmt) 
                Tem_y[i], Spa_y[i], Ali_y[i] = (mid_t + t)/2, (mid_s + s)/2, (mid_a + a)/2

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

            PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
                label[:,0], (Tem_y[:,3]+Spa_y[:,3])/2) 

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
            
            print('ST test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}'
                .format(SRCC_st,KRCC_st,PLCC_st,RMSE_st))

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
