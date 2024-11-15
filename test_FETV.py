import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn
from torchvision import transforms
from modular_model import modular
# from modular_model import modular_v1
from modular_utils import performance_fit
from train_dataloader import VideoDataset_val_test


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    stats = pd.read_csv('logs/ViTtest.csv')

    for i in range(config.loop):
        if config.model_name == 'aveScore':
            model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=config.feat_len)
        elif config.model_name == 'aveFeat':
            model = modular.ViTbCLIP_SpatialTemporal_dropout_meanpool(feat_len=config.feat_len)
        elif config.model_name == 'Hybrid':
            model = modular.ViTbCLIP_SpatialTemporal_dropout_hybrid(feat_len=config.feat_len)

        print('using model: ', config.model_name)
        
        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)
        model = model.float()

        # load the trained model
        ckpt_path=f'ckpts/{i}.pth'
        print(f'loading the trained model {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, weights_only=1))

        transformations_vandt = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
            transforms.CenterCrop(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


        # training data
        if config.database == 'FETV':
            prompt_num=619
            lp_dir = 'data/FETV_spatial_all_frames'
            # temporal features
            feature_dir = 'data/FETV_temporal_all_frames'
            # extract frames
            imgs_dir = 'data/FETV_base_all_frames'
            datainfo = config.mosfile
            print('using the mos file: ', datainfo)

            testset = VideoDataset_val_test(imgs_dir, feature_dir,
                                        lp_dir, datainfo, transformations_vandt, 'test',
                                        config.crop_size, prompt_num=config.prompt_num, 
                                        frame_num=config.frame_num, seed=i)
        elif config.database == 'LGVQ':
            prompt_num=468
            imgs_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_frames'
            tem_feat_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_tem'
            spa_feat_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_spa'
            mosfile = config.mosfile
            print('using the mos file: ', mosfile)
            testset = VideoDataset_val_test(config.database, imgs_dir, tem_feat_dir, spa_feat_dir, mosfile,
                                           transformations_vandt, 'test', config.crop_size,
                                           prompt_num=prompt_num, seed=i)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=config.num_workers)

        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset),3])
            Tem_y = np.zeros([len(testset),4])
            Spa_y = np.zeros([len(testset),4])
            Ali_y = np.zeros([len(testset),4])
        
            for i, (vid_chunk_g, vid_chunk_l, tem_feat_g, tem_feat_l,
                    spa_feat_g, spa_feat_l, mos,count,prmt) in enumerate(test_loader):
                for j in range(len(mos)):
                    label[i][j] = mos[j].item()
                mid_t=torch.zeros(4).to(device)
                mid_s=torch.zeros(4).to(device)
                mid_a=torch.zeros(4).to(device)
                
                for j in range(count):
                    vid_chunk_g[j] = vid_chunk_g[j].to(device)
                    tem_feat_g[j] = tem_feat_g[j].to(device)
                    spa_feat_g[j] = spa_feat_g[j].to(device)
                    #t,s,a = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j], prmt)

                    if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                        t, s, a, tm, sm, am = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j], prmt)
                        t = (t + tm) / 2
                        s = (s + sm) / 2
                        a = (a + am) / 2
                    else:
                        t, s, a = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j], prmt)

                    mid_t+=t.squeeze(1)
                    mid_s+=s.squeeze(1)
                    mid_a+=a.squeeze(1)
                            
                count=count.to(device)
                mid_t/=count
                mid_s/=count
                mid_a/=count
                
                vid_chunk_l = vid_chunk_l.to(device)
                tem_feat_l = tem_feat_l.to(device)
                spa_feat_l = spa_feat_l.to(device)
                # t,s,a= model(
                #     vid_chunk_l, tem_feat_l, spa_feat_l, prmt)

                if config.model_name == 'ViTbCLIP_SpatialTemporal_dropout_hybrid':
                    t, s, a, tm, sm, am = model(vid_chunk_l, tem_feat_l, spa_feat_l, prmt)
                    t = (t + tm) / 2
                    s = (s + sm) / 2
                    a = (a + am) / 2
                else:
                    t, s, a = model(vid_chunk_l, tem_feat_l, spa_feat_l, prmt)

                mid_t = (mid_t + t.squeeze(1))/2
                mid_s = (mid_s + s.squeeze(1))/2
                mid_a = (mid_a + a.squeeze(1))/2
                
                Tem_y[i] = mid_t.to('cpu')
                Spa_y[i] = mid_s.to('cpu')
                Ali_y[i] = mid_a.to('cpu')

            # tPLCC_b, tSRCC_b, tKRCC_b, tRMSE_b = performance_fit(
            #     label[:,0], Tem_y[:,0])
            # tPLCC_t, tSRCC_t, tKRCC_t, tRMSE_t = performance_fit(
            #     label[:,0], Tem_y[:,1])
            # tPLCC_s, tSRCC_s, tKRCC_s, tRMSE_s = performance_fit(
            #     label[:,0], Tem_y[:,2])
            tPLCC_st, tSRCC_st, tKRCC_st, tRMSE_st = performance_fit(
                label[:,0], Tem_y[:,3])
            
            # sPLCC_b, sSRCC_b, sKRCC_b, sRMSE_b = performance_fit(
            #     label[:,1], Spa_y[:,0])
            # sPLCC_t, sSRCC_t, sKRCC_t, sRMSE_t = performance_fit(
            #     label[:,1], Spa_y[:,1])
            # sPLCC_s, sSRCC_s, sKRCC_s, sRMSE_s = performance_fit(
            #     label[:,1], Spa_y[:,2])
            sPLCC_st, sSRCC_st, sKRCC_st, sRMSE_st = performance_fit(
                label[:,1], Spa_y[:,3])
            
            # aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
            #     label[:,2], Ali_y[:,0])
            # aPLCC_t, aSRCC_t, aKRCC_t, aRMSE_t = performance_fit(
            #     label[:,2], Ali_y[:,1])
            # aPLCC_s, aSRCC_s, aKRCC_s, aRMSE_s = performance_fit(
            #     label[:,2], Ali_y[:,2])
            aPLCC_st, aSRCC_st, aKRCC_st, aRMSE_st = performance_fit(
                label[:,2], Ali_y[:,3])    


            new_row=[
                    tSRCC_st,tKRCC_st, tPLCC_st, tRMSE_st,
                    sSRCC_st,sKRCC_st, sPLCC_st, sRMSE_st,
                    aSRCC_st,aKRCC_st, aPLCC_st, aRMSE_st
                    ]
            stats.loc[len(stats)]=new_row

            # print('===============Tem==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(tSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(tSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(tSRCC_s))
            # print(
            #     'ST test: SRCC: {:.4f}'.format(tSRCC_st))
            # print('===============Spa==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(sSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(sSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(sSRCC_s))

            # print(
            #     'ST test: SRCC: {:.4f}'.format(sSRCC_st))

            # print('===============Ali==============')
            # print(
            #     'base test: SRCC: {:.4f}'.format(aSRCC_b))
            # print(
            #     'T test: SRCC: {:.4f}'.format(aSRCC_t))
            # print(
            #     'S test: SRCC: {:.4f}'.format(aSRCC_s))
            # print(
            #     'ST test: SRCC: {:.4f}'.format(aSRCC_st))

        # new_row=[0,0,0,0,
        #             0,0,0,0,
        #             0,0,0,0] 
        # stats.loc[len(stats)]=new_row 
        stats.to_csv('logs/ViTtest.csv',index=False)        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LGVQ')
    parser.add_argument('--model_name', type=str,
                        default='aveScore')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--frame_num', type=int, default=8)
    parser.add_argument('--feat_len', type=int, default=8)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='ckpts/0_16.pth')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--mosfile', type=str,
                        default='/home/user/Documents/vqadata/BVQAdata/LGVQ_sorted.csv')
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--loop', type=int,default=1)

    config = parser.parse_args()

    main(config)
