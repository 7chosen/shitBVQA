import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from train_dataloader import VideoDataset_val_test


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=8)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # load the trained model
    print(f'loading the trained model {config.trained_model}')
    model.load_state_dict(torch.load(config.trained_model, weights_only=1))

    # training data
    if config.database == 'FETV':
        lp_dir = 'data/FETV_spatial_all_frames'
        # temporal features
        feature_dir = 'data/FETV_temporal_all_frames'
        # extract frames
        imgs_dir = 'data/FETV_base_all_frames'
        datainfo = config.mosfile
        print('using the mos file: ', datainfo)

    transformations_vandt = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    testset = VideoDataset_val_test(imgs_dir, feature_dir,
                                    lp_dir, datainfo, transformations_vandt, 'test',
                                    config.crop_size, prompt_num=config.prompt_num, frame_num=config.frame_num, seed=config.seed)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        labelt = np.zeros([len(testset)])
        labels = np.zeros([len(testset)])
        Tem_y_b = np.zeros([len(testset)])
        Tem_y_s = np.zeros([len(testset)])
        Tem_y_t = np.zeros([len(testset)])
        Tem_y_st = np.zeros([len(testset)])
        Spa_y_b = np.zeros([len(testset)])
        Spa_y_s = np.zeros([len(testset)])
        Spa_y_t = np.zeros([len(testset)])
        Spa_y_st = np.zeros([len(testset)])
    
        for i, (vid_chunk_g, vid_chunk_l, tem_feat_g, tem_feat_l,
                spa_feat_g, spa_feat_l, t_mos, s_mos,count) in enumerate(test_loader):
            labelt[i] = t_mos.item()
            labels[i] = s_mos.item()
            tmidb = tmids = tmidt = tmidst = 0
            smidb = smids = smidt = smidst = 0
            
            for j in range(count):
                vid_chunk_g[j] = vid_chunk_g[j].to(device)
                tem_feat_g[j] = tem_feat_g[j].to(device)
                spa_feat_g[j] = spa_feat_g[j].to(device)
                b, s, t, st = model(vid_chunk_g[j], tem_feat_g[j], spa_feat_g[j])

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
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                tSRCC_b, tKRCC_b, tPLCC_b, tRMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                tSRCC_s, tKRCC_s, tPLCC_s, tRMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                tSRCC_t, tKRCC_t, tPLCC_t, tRMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                tSRCC_st, tKRCC_st, tPLCC_st, tRMSE_st))
        print('===============Spa==============')
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                sSRCC_b, sKRCC_b, sPLCC_b, sRMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                sSRCC_s, sKRCC_s, sPLCC_s, sRMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                sSRCC_t, sKRCC_t, sPLCC_t, sRMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                sSRCC_st, sKRCC_st, sPLCC_st, sRMSE_st))
        
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporal_modular_dropout')
    parser.add_argument('--prompt_num', type=int, default=619)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--frame_num', type=int, default=8)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='ckpts_modular/0_29.pth')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--mosfile', type=str,
                        default='data/FETV.csv')
    parser.add_argument('--seed', type=int,default=0)

    config = parser.parse_args()

    main(config)
