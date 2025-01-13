import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn
# from torchvision import transforms
from tqdm import tqdm
import yaml
from modular import modular_model
from modular.utils import performance_fit, performance_no_fit
from train_dataloader import get_dataset
# from ViCLIP_models.viclip import ViCLIP


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)

    t2vqa_s,t2vqa_p,t2vqa_r, t2vqa_k = [],[],[],[]
    lgvq_s1,lgvq_p1,lgvq_r1, lgvq_k1 = [],[],[],[]
    lgvq_s2,lgvq_p2,lgvq_r2, lgvq_k2 = [],[],[],[]
    lgvq_s3,lgvq_p3,lgvq_r3, lgvq_k3 = [],[],[],[]
    fetv_s1,fetv_p1,fetv_r1, fetv_k1 = [],[],[],[]
    fetv_s2,fetv_p2,fetv_r2, fetv_k2 = [],[],[],[]
    fetv_s3,fetv_p3,fetv_r3, fetv_k3 = [],[],[],[]
    for loop in range(opt["split"]):
        if opt["model"] == 'aveScore':
            model = modular_model.ViTbCLIP_SpatialTemporal_dropout(feat_len=opt["feat_len"])
        elif opt["model"] == 'exp':
            model = modular_model.ViTbCLIP_exp(opt["model_path"], opt["model_base"],
                feat_len=opt["feat_len"])
        print('The current model is: ' + opt["model"])
        
        # if config.multi_gpu:
        #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        #     model = model.to(device)
        # else:

        # load the trained model
        if opt["pretrained_weights"] != None :
            print('loading the pretrained model from ', f"./ckpts/{opt["model"]}_{loop}.pth")
            # model.load_state_dict(torch.load(opt["pretrained_weights"]))
            model.load_state_dict(torch.load(f"./ckpts/{opt["model"]}_{loop}.pth",weights_only=True))


        model = model.to(device).to(torch.float32)
        # _,_, test_loader = get_dataset(opt,loop)   
        train_loader , _, test_loader= get_dataset(opt,loop)   

        
        with torch.no_grad():
            model.eval()
            
            # srcc1,srcc2,srcc3=[],[],[]
            # label=[]
            # for i, return_list in enumerate(tqdm(train_loader,desc='Training...')):
            #     # for tag,_ in enumerate(return_list):
            #         # if tag == 2:
            #     vid_chunk, vid_chunk_g, tem_feat, tem_feat_g,\
            #         spa_feat, spa_feat_g, mos, count, prmt = return_list[0]
            #     # for _ in range(len(mos)):
            #     label.append(mos[0][0].numpy())
            #     vid_chunk = vid_chunk.to(device)
            #     tem_feat = tem_feat.to(device)
            #     spa_feat = spa_feat.to(device)
            #     t, s, a = model(vid_chunk, tem_feat, spa_feat, prmt, len(mos))
            #     tmp=(t[3]+s[3])/2
            #     srcc3.append(tmp.numpy())
            #     # if i == 500:
            #     #     break
            # srcc3=np.array(srcc3)
            # label=np.array(label)
            # # scoreA=pd.read_csv('./data/T2VQA.csv').iloc[:,2]
            # # print(srcc3[:10])
            # # print(label[:10])
            # PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
            #             label, srcc3) 
            # print(SRCC_st)
            # return
            for dataname in opt["dataset"]:
                testset=test_loader[dataname]
                Tem_y, Spa_y, Ali_y = [torch.zeros([len(testset), 4]) for _ in range(3)]
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
                        t, s, a = model(x, y, z, prmt,len(mos))
                        mid_t, mid_s, mid_a = mid_t+t, mid_s+s, mid_a+a
                    mid_t, mid_s, mid_a = mid_t/count, mid_s/count, mid_a/count

                    x = vid_chunk_g.to(device)
                    y = tem_feat_g.to(device)
                    z = spa_feat_g.to(device)
                    t, s, a = model(x, y, z, prmt,len(mos))
                    Tem_y[i], Spa_y[i], Ali_y[i] = (mid_t + t)/2, (mid_s + s)/2, (mid_a + a)/2

                Tem_y, Spa_y, Ali_y = Tem_y.cpu().numpy(), Spa_y.cpu().numpy(), Ali_y.cpu().numpy()
                if dataname == 'T2VQA':
                    PLCC_st, SRCC_st, KRCC_st, RMSE_st = performance_fit(
                        label[:,0], (Tem_y[:,3]+Spa_y[:,3])/2)    
                    # print('{} final ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                    #     dataname, SRCC_st, KRCC_st, PLCC_st, RMSE_st))
                    # print('===')
                    t2vqa_s.append(SRCC_st)
                    t2vqa_p.append(PLCC_st)
                    t2vqa_r.append(RMSE_st)
                    t2vqa_k.append(KRCC_st)
                else:
                    tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_fit(
                        label[:,0], Tem_y[:,3])
                    sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_fit(
                        label[:,1], Spa_y[:,3])
                    aPLCC_st, aSRCC_st, aKRCC_st, aRMSE_st = performance_fit(
                        label[:,2], Ali_y[:,3])
                    # print('{} final tem ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                    #     dataname, tSRCC_st, tKRCC_st, tPLCC_st, tRMSE_st))
                    # print('{} final spa ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                    #     dataname, sSRCC_st, sKRCC_st, sPLCC_st, sRMSE_st))
                    # print('{} final ali ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
                    #     dataname, aSRCC_st, aKRCC_st, aPLCC_st, aRMSE_st))
                    # print('===')
                    if dataname == 'LGVQ':
                        lgvq_s1.append(tSRCC_st)
                        lgvq_p1.append(tPLCC_st)
                        lgvq_r1.append(tRMSE_st)
                        lgvq_k1.append(tKRCC_st)
                        lgvq_s2.append(sSRCC_st)
                        lgvq_p2.append(sPLCC_st)
                        lgvq_r2.append(sRMSE_st)
                        lgvq_k2.append(sKRCC_st)
                        lgvq_s3.append(aSRCC_st)
                        lgvq_p3.append(aPLCC_st)
                        lgvq_r3.append(aRMSE_st)
                        lgvq_k3.append(aKRCC_st)
                    if dataname == 'FETV':
                        fetv_s1.append(tSRCC_st)
                        fetv_p1.append(tPLCC_st)
                        fetv_r1.append(tRMSE_st)
                        fetv_k1.append(tKRCC_st)
                        fetv_s2.append(sSRCC_st)
                        fetv_p2.append(sPLCC_st)
                        fetv_r2.append(sRMSE_st)
                        fetv_k2.append(sKRCC_st)
                        fetv_s3.append(aSRCC_st)
                        fetv_p3.append(aPLCC_st)
                        fetv_r3.append(aRMSE_st)
                        fetv_k3.append(aKRCC_st)
    print('LGVQ final tem ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(lgvq_s1), np.median(lgvq_k1), np.median(lgvq_p1), np.median(lgvq_r1)))
    print('LGVQ final spa ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(lgvq_s2), np.median(lgvq_k2), np.median(lgvq_p2), np.median(lgvq_r2)))
    print('LGVQ final ali ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(lgvq_s3), np.median(lgvq_k3), np.median(lgvq_p3), np.median(lgvq_r3)))

    print('FETV final tem ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format( 
        np.median(fetv_s1), np.median(fetv_k1), np.median(fetv_p1), np.median(fetv_r1)))
    print('FETV final spa ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(fetv_s2), np.median(fetv_k2), np.median(fetv_p2), np.median(fetv_r2)))
    print('FETV final ali ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(fetv_s3), np.median(fetv_k3), np.median(fetv_p3), np.median(fetv_r3)))


    print('T2VQA final ST: {:.4f}, {:.4f}, {:.4f}, {:.4f},'.format(
        np.median(t2vqa_s), np.median(t2vqa_k), np.median(t2vqa_p), np.median(t2vqa_r))
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="./cfg.yml", help="the option file")
    config = parser.parse_args()
    main(config)
