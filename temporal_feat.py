# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from tem_dataloader import VideoDataset_NR_SlowFast_feature

from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn


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
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():

            # print(x[0].shape)   
            # print(x[1].shape)   
            x = self.feature_extraction(x)
            # print(x[0].shape)   
            # print(x[1].shape)   
            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = slowfast()

    model = model.to(device)

    resize = config.resize

    transformations = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                               std=[0.225, 0.225, 0.225])])


    ## training data
    for t2vmdl in config.t2vmodel:

        videos_dir = f'/home/user/Documents/vqadata/FETV/{t2vmdl}/videos'
        datainfo = 'data/LiveVQC_data.mat'
        
        trainset = VideoDataset_NR_SlowFast_feature(config.database, 
            videos_dir, datainfo, transformations, resize, config.num_frame)


        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                shuffle=False, num_workers=config.num_workers)

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            for i, (video, mos, video_name) in enumerate(train_loader):
                video_name = str(i)
                # print(len(video))
                # print(video[0].shape) 
                if not os.path.exists(config.feature_save_folder + t2vmdl + '/' +video_name):
                    dir_name=config.feature_save_folder + t2vmdl+'/' +video_name
                    os.makedirs(dir_name)

                for idx, ele in enumerate(video):
                    # swap channel and vid_length_clip
                    # B*C*T*H*W
                    # 1*3*num_frames*224*224
                    ele = ele.permute(0, 2, 1, 3, 4)
                    # print(ele.shape)
                    inputs = pack_pathway_output(ele, device)
                    # print(inputs[0].shape)
                    slow_feature, fast_feature = model(inputs)
                    
                    np.save(dir_name+ '/'+str(idx) + 'slow',
                            slow_feature.to('cpu').numpy())
                    np.save(dir_name+'/' + str(idx) + 'fast',
                            fast_feature.to('cpu').numpy())
                    # print('done')
                            
                            
        #         break
        # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='FETV')
    # parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_frame', type=int, default=32)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='data/FETVtemporal_16frames/')
    parser.add_argument('--t2vmodel',type=list,default=['cogvideo','text2video-zero','modelscope-t2v','zeroscope'])

    config = parser.parse_args()

    main(config)