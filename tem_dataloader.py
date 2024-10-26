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



class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, database_name, 
                 data_dir, filename_path, transform, resize, num_frame):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'KoNViD-1k':
            m = scio.loadmat(filename_path)
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                score.append(m['scores'][i][0])
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']

        elif database_name == 'FETV':
            video_names = []
            score =[]
            for i in range(619):
                if data_dir.find('cogvid') != -1:
                    video_names.append(f'{i}.gif')
                else:
                    video_names.append(f'{i}'+'.mp4')
                score.append(0)
            dataInfo=pd.DataFrame(video_names)
            dataInfo['score']=score
            dataInfo.columns=['file_names','MOS']
            self.video_names=dataInfo['file_names']
            self.score=dataInfo['MOS']
            
        
        self.database_name = database_name
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)
        self.num_frame = num_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        # video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))

        filename = os.path.join(self.videos_dir, video_name)
        
        
        print(filename)
        if filename[-1] == '4':
            cap = cv2.VideoCapture(filename)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        elif filename[-1] == 'f':
            cap=Image.open(filename)
            video_length=cap.n_frames
            frame_duration=cap.info['duration']
            video_frame_rate=int(round(1000/frame_duration))
        else:
            raise Exception('vid path is not right')
        
        print('len: ',video_length,' fr: ', video_frame_rate)

        video_second = int(video_length / video_frame_rate)
        # one clip for one second video
        if video_frame_rate == 0:
            raise Exception('no frame detect')
        # else:
            # vid_sec_min = int(video_length / video_frame_rate)
        vid_sec_min = 8 
        # 32
        video_channel = 3
        video_length_clip = self.num_frame
        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])
        transformed_video_all = []
        
        if filename[-1] == '4':       
            for i in range(video_length):
                has_frames, frame = cap.read()
                if not has_frames:
                    raise Exception('no frame detect')
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[i] = read_frame
            cap.release()
        elif filename[-1] == 'f':
            current_idx=0
            while 1:
                try:
                    cap.seek(current_idx)
                    frame_rgb=cap.convert("RGB")
                    # frame_np=np.array(frame_rgb)
                    frame_np=self.transform(frame_rgb)
                    transformed_frame_all[current_idx]=frame_np
                    current_idx+=1
                except EOFError:
                    break
        else:
            raise Exception('vid path NOT right')
        
        # print(transformed_frame_all[0].shape)

        # store {video_second} clips to a list, and every clip have {video_length_clip} frames
        for i in range(video_second):
            # 32*3*224*224
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            # if a clip is greater then the origin video length, then instead it use
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
                    
            transformed_video_all.append(transformed_video)



        if video_second < vid_sec_min:
            for i in range(video_second, vid_sec_min):
                transformed_video_all.append(transformed_video_all[video_second - 1])


        # 16*3*224*224
        # print(transformed_frame_all.shape)
        
        # 8*32*3*224*224
        # print(transformed_video_all[0].shape)
 
        
        # print(1)
        return transformed_video_all, video_score, video_name


class VideoDataset_LQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, database_name, data_dir, filename_path, transform, resize, num_frame):
        super(VideoDataset_LQ_SlowFast_feature, self).__init__()
        if database_name == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']
            
        elif database_name == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            dataInfo = pd.DataFrame({'video_names':video_names, 'MOS':MOS})
            self.video_names = dataInfo['video_names']
            self.score = dataInfo['MOS']

        elif database_name == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['file_name']
            self.score = dataInfo['MOS']
        
        elif database_name == 'LIVE-livestreaming':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['video']
            self.score = dataInfo['MOS']

        self.database_name = database_name
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)
        self.num_frame = num_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        # video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))

        filename = os.path.join(self.videos_dir, video_name)
        print(filename)

        if self.database_name == 'LIVE-Qualcomm':
            video_clip_min = 15
            video_height = 1080  # the heigh of frames
            video_width = 1920  # the width of frames
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
            video_frame_rate = video_data.shape[0] // 15

        elif self.database_name == 'BVI-SR':
            video_clip_min = 5
            video_height = 2160  # the heigh of frames
            video_width = 3840  # the width of frames
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})
            video_frame_rate = 60

        elif self.database_name == 'BVIHFR':
            video_clip_min = 10
            video_height = 1080  # the heigh of frames
            video_width = 1920  # the width of frames
            filename = os.path.join(self.videos_dir, video_name) + '-360-1920x1080.yuv'
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})

            video_frame_rate = int(video_name.split('-')[1][:-3])

        elif self.database_name == 'LIVE-livestreaming':
            video_clip_min = 7
            video_height = 2160  # the heigh of frames
            video_width = 3840  # the width of frames
            filename = os.path.join(self.videos_dir, video_name)
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})

            video_list = video_name.split('_')
            for name_str in video_list:
                if name_str[-3:]=='fps':
                    video_frame_rate = int(name_str[:-3])

        video_length = video_data.shape[0]
        video_channel = 3

        print(video_length)
        print(video_frame_rate)

        video_clip = int(video_length / video_frame_rate)

        video_length_clip = self.num_frame

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            frame = video_data[i]
            read_frame = Image.fromarray(frame)
            read_frame = self.transform(read_frame)
            transformed_frame_all[video_read_index] = read_frame
            video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]


        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name