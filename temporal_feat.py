# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from tem_dataloader import VideoDataset_temporal_slowfast
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
from config import FETV_T2V_model,FETV_vid_path


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
            x = self.feature_extraction(x)
            
            # 1*256*frames*7*7
            # fast_feature=x[1]
            fast_feature=x[1].squeeze()
            fast_feature = self.fast_avg_pool(fast_feature)
            fast_feature=torch.unsqueeze(fast_feature,0)
            print(fast_feature.shape)

            # 1*256*frames*1*1
            
        return fast_feature


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = slowfast()

    model = model.to(device)

    resize = config.resize

    transformations = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                               std=[0.225, 0.225, 0.225])])

    # training data
    for n,t2vmdl in enumerate(FETV_T2V_model):

        videos_dir = FETV_vid_path[n]

        trainset = VideoDataset_temporal_slowfast(config.database,
                        videos_dir, transformations, resize)

        # dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                        shuffle=False, num_workers=config.num_workers)

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            for i, video in enumerate(train_loader):
                if not os.path.exists(config.feature_save_folder + f'/{t2vmdl}'):
                    dir_name = config.feature_save_folder + f'/{t2vmdl}'
                    os.makedirs(dir_name)
                
                # print(len(video))
                # print(video[0].shape)
                # swap channel and vid_length_clip
                # B*C*T*H*W
                # 1*3*all_frame*224*224
                video = video.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(video, device)
                fast_feature = model(inputs)
                np.save(dir_name +'/' + str(i),
                        fast_feature.to('cpu').numpy())

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)
    # parser.add_argument('--num_frame', type=int, default=32)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str,
                        default='data/FETV_temporal_all_frames')

    config = parser.parse_args()

    main(config)
