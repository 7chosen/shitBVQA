import glob
import json
import os
from sys import prefix

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2
import skvideo.io
import csv
import math


def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset_temporal_slowfast(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, database_name,
                 data_dir, transform, resize=224):
        super(VideoDataset_temporal_slowfast, self).__init__()

        # if database_name == 'KoNViD-1k':
        #     m = scio.loadmat(filename_path)
        #     video_names = []
        #     score = []
        #     index_all = m['index'][0]
        #     for i in index_all:
        #         video_names.append(m['video_names'][i][0]
        #                            [0].split('_')[0] + '.mp4')
        #         score.append(m['scores'][i][0])
        #     dataInfo = pd.DataFrame(video_names)
        #     dataInfo['score'] = score
        #     dataInfo.columns = ['file_names', 'MOS']
        #     self.video_names = dataInfo['file_names']
        #     self.score = dataInfo['MOS']

        # if database_name == 'FETV':
        #     # video_names = []
        #     # for i in range(619):
        #     #     if data_dir.find('cogvid') != -1:
        #     #         video_names.append(f'{i}.gif')
        #     #     else:
        #     #         video_names.append(f'{i}'+'.mp4')
        #     # dataInfo = pd.DataFrame(video_names)
        #     # dataInfo.columns = ['file_names']
        #     # self.video_names = dataInfo['file_names']
        #     self.video_names=glob.glob(f'{data_dir}/*.mp4')
        
        # if database_name == 'LGVQ':
        #     # video_names=glob.glob(f'{data_dir}/*.mp4')            
        #     # dataInfo = pd.DataFrame(video_names)
        #     # dataInfo.columns=['file_names']
        #     # self.video_names = dataInfo['file_names']
        #     self.video_names=glob.glob(f'{data_dir}/*.mp4')
        
        # if database_name == 'T2VQA':
        self.video_names=glob.glob(f'{data_dir}/*.mp4')
            

        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        # video_name = self.video_names.iloc[idx]

        filename = self.video_names[idx]
        vid_nm=filename.split('/')[-1][:-4]
        # print(filename)

        cap = cv2.VideoCapture(filename)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_length_round = video_length if video_length%8==0 else (video_length//8 +1 )*8
        if video_frame_rate == 0:
            raise Exception('no frame detect')
        video_channel = 3
        transformed_frame_all = torch.zeros(
            [video_length_round, video_channel, self.resize, self.resize])

        for i in range(video_length):
            has_frames, frame = cap.read()
            if not has_frames:
                raise Exception('no frame detect')
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = self.transform(read_frame)
            # print(read_frame.shape)
            transformed_frame_all[i] = read_frame
        cap.release()

        # 0-39 40
        # 0-32 33 || 33: == 32
        if video_length_round != video_length:
            transformed_frame_all[video_length:,:]=transformed_frame_all[video_length-1,:]
        
        print('len: ', transformed_frame_all.shape[0], ' fr: ', video_frame_rate, ' name: ', vid_nm)
        # # 1-8 9-16
        # if video_length % 8 != 0 :
        #     tmp=transformed_frame_all.clone()
        #     # print(transformed_frame_all[video_length])
        #     # video_length=video_length_round+8
        #     # last_8_ele=transformed_frame_all[video_length-9:video_length-1]
        #     transformed_frame_all[-8:]=tmp[video_length-9:video_length-1]
        #     # transformed_frame_all=torch.cat((transformed_frame_all,last_8_ele))
            
        # transformed_video_all = []

        # video_length=video_length/video_second
        # store {video_second} clips to a list, and every clip have {video_length_clip} frames
        # the last clip will follow rule that the last few frames of this clip will duplicate the last frame of the video  
        # for i in range(video_second):
        #     # clip_len*3*224*224
        #     transformed_video = torch.zeros(
        #         [video_frame_rate, video_channel, self.resize, self.resize])
        #     if (i * video_frame_rate + video_length) <= video_length:
        #         transformed_video = transformed_frame_all[
        #             i * video_frame_rate: (i * video_frame_rate + video_length)]
        #     # if a clip is greater then the original video length, then instead it use
        #     else:
        #         transformed_video[:(video_length - i * video_frame_rate)
        #                           ] = transformed_frame_all[i * video_frame_rate:]
        #         for j in range((video_length - i * video_frame_rate), video_length):
        #             transformed_video[j] = transformed_video[video_length -
        #                                                      i * video_frame_rate - 1]

                # transformed_video_all.append(transformed_video)
        # transformed_video_all.append(transformed_frame_all)

        # if video_second < vid_sec_min:
        #     for i in range(video_second, vid_sec_min):
        #         transformed_video_all.append(
        #             transformed_video_all[video_second - 1])

        # frame_sum*3*224*224
        # print(transformed_frame_all.shape)

        # 8*32*3*224*224
        # print(transformed_video_all[0].shape)
        
        return transformed_frame_all, vid_nm