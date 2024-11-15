import torch
from torchvision import  models
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import random
from argparse import ArgumentParser
from spatial_dataloader import VideoDataset,VideoDataset_LGVQ
from config import LGVQ_T2V_model,FETV_T2V_model
import scipy.io as scio


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return
    
class ResNet18_LP(torch.nn.Module):
    """Modified ResNet18 for feature extraction"""
    def __init__(self, layer=2):
        super(ResNet18_LP, self).__init__()
        if layer == 1:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-5])
        elif layer == 2:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-4])
        elif layer == 3:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-3])
        else:
            self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False 

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, layer=2, frame_batch_size=10, device='cuda'):
    """feature extraction"""
    extractor = ResNet18_LP(layer=layer).to(device)  #
    video_length = video_data.shape[0]
    # print(video_length)
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output = torch.cat((output1, output2), 1).squeeze()

    return output
    # return output1.squeeze()


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Laplacian Pyramids Features using Pre-Trained ResNet-18')
    parser.add_argument("--seed", type=int, default=20241017)
    parser.add_argument('--database', default='LGVQ', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--layer', type=int, default=2,
                        help='RN18 layer for feature extraction (default: 2)')
    parser.add_argument('--num_levels', type=int, default=6,
                        help='number of gaussian pyramids')
    # parser.add_argument('--t2vmodel',type=list,default=['cogvideo','text2video-zero','modelscope-t2v','zeroscope'])
    # parser.add_argument('--frame_num',type=int,default=8)
    parser.add_argument('--prompt_num',type=int,default=619)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.database == 'KoNViD-1k':
        vids_dir = 'data/konvid1k_image_all_fps1'
        save_folder = 'data/konvid1k_LP_ResNet18/'
        filename_path = 'data/KoNViD-1k_data.mat'

        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0].split('_')[0])
        # video_length_read = 8

    elif args.database == 'FETV':
        for t2vmdl in FETV_T2V_model:
            print('========================',t2vmdl)
            vids_dir = f'/home/user/Documents/vqadata/FETV/{t2vmdl}/videos'
            save_folder = f'data/FETV_spatial_all_frames/{t2vmdl}'
            # video_names = [0]
            # video_length_read = args.frame_num 
            # x=np.random.randint(10)
            dataset = VideoDataset(args.database, vids_dir, args.num_levels)
            # print(len(dataset))
            for i in range(args.prompt_num):
                print(f'process {i}th vid')
                current_data , video_length= dataset[i]
                # current_data[0] = eg. [40 (8 images * 5 layers)]
                features = get_features(current_data, args.layer, args.frame_batch_size, device)
                # start_index=0
                exit_folder(os.path.join(save_folder, str(i)))
                for j in range(video_length):
                    img_features = features[j*(args.num_levels-1) : (j+1)*(args.num_levels-1)]
                    np.save(os.path.join(save_folder, str(i), str(j)), img_features.to('cpu').numpy())
    
    elif args.database == 'LGVQ':
        for t2vmdl in LGVQ_T2V_model:
            print('======',t2vmdl,'======')
            vids_dir = '/home/user/Documents/vqadata/LGVQ/videos/'+t2vmdl
            save_folder=f'/home/user/Documents/vqadata/BVQAdata/LGVQ_spa/{t2vmdl}'
            dataset=VideoDataset_LGVQ(args.database, vids_dir, args.num_levels)

            for i in range(468):
                print(f'process {i}th vid')
                dt, f_len, nm = dataset[i]
                features=get_features(dt,args.layer, args.frame_batch_size,device)
                exit_folder(os.path.join(save_folder,nm))
                for j in range(f_len):
                    img_features = features[j*(args.num_levels-1) : (j+1)*(args.num_levels-1)]
                    np.save(os.path.join(save_folder, nm, str(j)), img_features.to('cpu').numpy())
            break
            

                    