# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from tem_dataloader import VideoDataset_temporal_slowfast
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
from config import FETV_T2V_model,FETV_vid_path,LGVQ_T2V_model


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Slow pathway samples the frame by looping every 4 frame
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(
            *list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(
                str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module(
            'slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module(
            # 'fast_avg_pool', slowfast_pretrained_features[5].pool[1])
            'fast_avg_pool',nn.AvgPool2d(kernel_size=7))
            # 'fast_avg_pool',nn.AdaptiveAvgPool3d((1,1,1)))

        self.adp_avg_pool.add_module(
            'adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            
            # 1*c*frames*h*w
            # print(x[1].shape)
            x = self.feature_extraction(x)
            
            # 1*256*frames*7*7
            # fast_feature=x[1]
            fast_feature=x[1].squeeze()
            fast_feature = self.fast_avg_pool(fast_feature)
            fast_feature=torch.unsqueeze(fast_feature,0)
            # print(fast_feature.shape)

            # 1*256*frames*1*1
            
        return fast_feature


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = slowfast()

    model = model.to(device)

    # resize = config.resize

    transformations = transforms.Compose([transforms.Resize((config.resize,config.resize),interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                               std=[0.225, 0.225, 0.225])])

    # training data
    if config.database == 'FETV':
        config.save_folder='/home/user/Documents/vqadata/BVQAdata/FETV_tem'
        if not os.path.exists(config.save_folder):
            os.makedirs(config.save_folder)
        vid_path='/home/user/Documents/vqadata/FETV'
        trainset=VideoDataset_temporal_slowfast(config.database, vid_path,
                                                transformations)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                        shuffle=False, num_workers=config.num_workers)
        with torch.no_grad():
            model.eval()
            for idx,(vid,nm) in enumerate(train_loader):
                nm=nm[0]
                print(f'process the {idx} vid')
                inputs=vid.permute(0,2,1,3,4)
                inputs = pack_pathway_output(inputs, device)
                fast_feature = model(inputs)
                np.save(f'{config.save_folder}/{nm}', fast_feature.to('cpu').numpy()) 

    elif config.database == 'LGVQ':
        config.save_folder='/home/user/Documents/vqadata/BVQAdata/LGVQ_tem'
        if not os.path.exists(config.save_folder ):
            os.makedirs(config.save_folder )
        vid_path='/home/user/Documents/vqadata/LGVQ'
        trainset=VideoDataset_temporal_slowfast(config.database, vid_path,
                                                transformations)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                        shuffle=False, num_workers=config.num_workers)
        with torch.no_grad():
            model.eval()
            for idx, (vid,nm) in enumerate(train_loader):
                nm=nm[0]
                print(f'process the {idx} vid')
                inputs=vid.permute(0,2,1,3,4)
                inputs = pack_pathway_output(inputs, device)
                fast_feature = model(inputs)
                np.save(f'{config.save_folder}/{nm}', fast_feature.to('cpu').numpy())
    elif config.database == 'T2VQA':
        config.save_folder='/home/user/Documents/vqadata/BVQAdata/T2VQA_tem'
        if not os.path.exists(config.save_folder):
            os.makedirs(config.save_folder)
        vid_path='/home/user/Documents/vqadata/T2VQA/videos'
        trainset=VideoDataset_temporal_slowfast(config.database, vid_path,
                                                transformations)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                        shuffle=False, num_workers=config.num_workers)
        with torch.no_grad():
            model.eval()
            for idx,(vid,nm) in enumerate(train_loader):
                nm=nm[0]
                print(f'process the {idx} vid')
                inputs=vid.permute(0,2,1,3,4)
                inputs = pack_pathway_output(inputs, device)
                fast_feature = model(inputs)
                np.save(f'{config.save_folder}/{nm}', fast_feature.to('cpu').numpy())          
            
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)
    # parser.add_argument('--num_frame', type=int, default=32)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--save_folder', type=str,
                        default='/home/user/Documents/vqadata/BVQAdata/FETV_tem')

    config = parser.parse_args()

    main(config)
