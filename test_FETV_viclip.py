import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn
from torchvision import transforms
from modular_model import modular
from modular_utils import performance_fit
from train_dataloader import viCLIP_vandtDT
from ViCLIP_models.viclip import ViCLIP

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    static = pd.read_csv('logs/static_ret.csv')

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_dropout(feat_len=8)
    else:
        model=ViCLIP()

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

    testset = viCLIP_vandtDT(imgs_dir,datainfo, 'test',
                                    config.crop_size, prompt_num=config.prompt_num, frame_num=config.frame_num, seed=config.seed)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset),3])
        Ali_y = np.zeros(len(testset))
    
        for i, (v_l, v_g, prmt, mos, count) in enumerate(test_loader):
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
        
        aPLCC_b, aSRCC_b, aKRCC_b, aRMSE_b = performance_fit(
            label[:,2], Ali_y) 


        print('===============Ali==============')
        print(
            'base val: SRCC: {:.4f}'.format(aSRCC_b))

        static.to_csv('logs/static_ret.csv',index=False)        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--model_name', type=str,
                        default='viCLIP')
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
