import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn
from torchvision import transforms
from modular_model import modular
from modular_utils import performance_fit
from train_dataloader import viCLIP_vandtDT, VideoDataset_val_test
from ViCLIP_models.viclip import ViCLIP

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    static = pd.read_csv('logs/ViTtest.csv')


    for loop in range(config.loop):
        model=ViCLIP()
        print('using model: ', config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
            model = model.float()

        # load the trained model
        ckpt_path=f'ckpts/{loop}.pth'
        print(f'loading the trained model {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, weights_only=1))
        
        # print(f'loading the trained model {config.trained_model}')
        # model.load_state_dict(torch.load(config.trained_model, weights_only=1))

        transformations_vandt = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
            transforms.CenterCrop(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

        # training data
        if config.database == 'FETV':
            lp_dir = 'data/FETV_spatial_all_frames'
            # temporal features
            feature_dir = 'data/FETV_temporal_all_frames'
            # extract frames
            imgs_dir = 'data/FETV_base_all_frames'
            datainfo = config.mosfile
            print('using the mos file: ', datainfo)

        elif config.database == 'LGVQ':
            prompt_num=468
            imgs_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_frames'
            tem_feat_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_tem'
            spa_feat_dir = '/home/user/Documents/vqadata/BVQAdata/LGVQ_spa'
            mosfile = config.mosfile
            print('using the mos file: ', mosfile)

        testset = VideoDataset_val_test(config.database, imgs_dir, tem_feat_dir, spa_feat_dir, mosfile,
                                            transformations_vandt, 'test', config.crop_size,
                                            prompt_num=prompt_num, seed=loop)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=config.num_workers)

        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset),3])
            Tem_y = np.zeros(len(testset))
            Spa_y = np.zeros(len(testset))
            Ali_y = np.zeros(len(testset))
        
            for i, (v_glo, v_loc, _,_,_,_, mos, count, prmt) in enumerate(test_loader):
                for j in range(len(mos)):
                    label[i][j] = mos[j].item()
                score=0
                score_1=0
                score_2=0

                for j in range(count):
                    v_glo[j] = v_glo[j].to(device)
                    tem = model(image=v_glo[j],raw_text=prmt,idx=0,return_sims=True)
                    score +=tem[0]
                    score_1+= tem[1]
                    score_2+= tem[2]
                count=count.to(device)
                score/=count
                score_1/=count
                score_2/=count

                v_loc = v_loc.to(device)
                tem= model(image=v_loc,raw_text=prmt,idx=0,return_sims=True)
                
                score+=tem[0]
                score/=2
                score=score.to('cpu')
                Tem_y[i]=score

                score_1+=tem[1]
                score_1/=2
                score_1=score_1.to('cpu')
                Spa_y[i]=score_1

                score_2+=tem[2]
                score_2/=2
                score_2=score_2.to('cpu')
                Ali_y[i]=score_2

            tPLCC_b, tSRCC_b, tKRCC_b, tRMSE_b = performance_fit(
                    label[:,0], Tem_y)
            sPLCC_b, sSRCC_b, sKRCC_b, sRMSE_b = performance_fit(
                    label[:,1], Spa_y)
            aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
                    label[:,2], Ali_y)
                
            new_row=[tSRCC_b,tKRCC_b,tPLCC_b,tRMSE_b,
                    sSRCC_b,sKRCC_b,sPLCC_b,sRMSE_b,
                    aSRCC_b,aKRCC_b,aPLCC_b,aRMSE_b]
            static.loc[len(static)]=new_row
            print('===============Tem==============')
            print(
                'base test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'
                .format(tSRCC_b,tKRCC_b,tPLCC_b,tRMSE_b))
            print('===============Spa==============')
            print(
                'base test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'
                .format(sSRCC_b,sKRCC_b,sPLCC_b,sRMSE_b))
            print('===============Ali==============')
            print(
                'base test: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'
                .format(aSRCC_b,aKRCC_b,aPLCC_b,aRMSE_b))
            
            # # new_row=[0,0,0,0] 
            # static.loc[len(static)]=new_row
            static.to_csv('logs/ViTval.csv',index=False)        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LGVQ')
    parser.add_argument('--model_name', type=str,
                        default='viCLIP')
    # parser.add_argument('--prompt_num', type=int, default=619)
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
                        default='/home/user/Documents/vqadata/BVQAdata/LGVQ_sorted.csv')
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--loop', type=int,default=1)

    config = parser.parse_args()

    main(config)
