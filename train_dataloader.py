import json
import os
import csv
import random

import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import math
# from random import shuffle
from config import T2V_model


def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset_train_val(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 database_name, crop_size, feature_type, prompt_num, frame_num=8, seed=0):
        super(VideoDataset_train_val, self).__init__()

        with open(mosfile_path, 'r') as f:
            mos = json.load(f)
        self.video_names = []
        self.score = []
        random.seed(seed)
        np.random.seed(seed)

        index_rd = np.random.permutation(prompt_num)
        train_index = index_rd[0:int(prompt_num*0.7)]
        val_index = index_rd[int(prompt_num * 0.7):int(prompt_num * 0.8)]

        if database_name == 'train':
            for idx, t2vmdl in enumerate(T2V_model):
                for i in train_index:
                    suffix = "{:03}".format(i)
                    self.video_names.append(t2vmdl+suffix)
                    self.score.append(mos[idx][i])
        elif database_name == 'val':
            for idx, t2vmdl in enumerate(T2V_model):
                for i in val_index:
                    suffix = "{:03}".format(i)
                    self.video_names.append(t2vmdl+suffix)
                    self.score.append(mos[idx][i])


        dataInfo = pd.DataFrame(self.video_names)
        dataInfo['score'] = self.score
        dataInfo.columns = ['file_names', 'MOS']
        self.video_names = dataInfo['file_names'].tolist()
        self.score = dataInfo['MOS'].tolist()

        self.crop_size = crop_size
        self.imgs_dir = imgs_dir
        self.temporalFeat = temporalFeat
        self.spatialFeat = spatialFeat
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.frame_num = frame_num
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        t2vmodel_name = video_name[:-3]
        # remove leading zeros
        videoname_idx = int(video_name[-3:])
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        img_path_name = os.path.join(
            self.imgs_dir, t2vmodel_name, str(videoname_idx))
        spatial_feat_name = os.path.join(
            self.spatialFeat, t2vmodel_name, str(videoname_idx))
        temporal_feat_name = os.path.join(
            self.temporalFeat, t2vmodel_name, str(videoname_idx))

        video_channel = 3

        # 224
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        # fix random
        # seed = np.random.randint(20231001)
        # random.seed(seed)

        frame_len = len(os.listdir(img_path_name))
        start_index = random.randint(0, frame_len-self.frame_num)
        random_range=list(range(start_index,start_index+8))

        # first_file=sorted(tmp,key=lambda x:int(x.split('.')[0]))[0]
        transformed_video = torch.zeros(
            [self.frame_num, video_channel, video_height_crop, video_width_crop])
        resized_lp = torch.zeros([self.frame_num, 5*256])
        for i in range(start_index, start_index+self.frame_num):
            imge_name = os.path.join(img_path_name, f'{i}.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i-start_index] = read_frame
            resized_lp[i-start_index] = torch.from_numpy(
            np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.temporalFeat, videoname_idx)
            transformed_feature = torch.zeros([self.frame_num, 2048])
            for i in range(self.frame_num):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            # feature_folder_name = os.path.join(temporal_feat_name, )
            
            # transformed_feature = torch.zeros([self.frame_num, 256])
            # for i in range(self.frame_num):
            #     # print(i)
            #     i_index = i   #TODO
            #     feature_3D = np.load(os.path.join(temporal_feat_name,str(i_index) + '.npy'))
            #     feature_3D = torch.from_numpy(feature_3D)
            #     feature_3D = feature_3D.squeeze()
            #     transformed_feature[i] = feature_3D
            
            # 1*256*frames*1*1   
            feature_3D=np.load(temporal_feat_name+'.npy')
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D=feature_3D[:,:,random_range,:,:]
            feature_3D = feature_3D.squeeze().permute(1,0)
            # self.frame_num*256
            
            
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.temporalFeat, videoname_idx)
            transformed_feature = torch.zeros([self.frame_num, 2048+256])
            for i in range(self.frame_num):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D
        # transformed_feature = []

        return transformed_video, feature_3D, video_score, resized_lp


class VideoDataset_test(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 crop_size, feature_type, prompt_num, frame_num=8, seed=0):
        super(VideoDataset_test, self).__init__()

        with open(mosfile_path, 'r') as f:
            mos = json.load(f)
        self.video_names = []
        self.score = []
        random.seed(seed)
        np.random.seed(seed)
        # video_num=dataInfo.shape[0]

        index_rd = np.random.permutation(prompt_num)
        test_index = index_rd[int(prompt_num * 0.8):]

        for idx, t2vmdl in enumerate(T2V_model):
            for i in test_index:
                suffix = "{:03}".format(i)
                self.video_names.append(t2vmdl+suffix)
                self.score.append(mos[idx][i])

        dataInfo = pd.DataFrame(self.video_names)
        dataInfo['score'] = self.score
        dataInfo.columns = ['file_names', 'MOS']
        self.video_names = dataInfo['file_names'].tolist()
        self.score = dataInfo['MOS'].tolist()

        self.crop_size = crop_size
        self.imgs_dir = imgs_dir
        self.temporalFeat = temporalFeat
        self.spatialFeat = spatialFeat
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        t2vmodel_name = video_name[:-3]
        # remove leading zeros
        videoname_idx = int(video_name[-3:])
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        img_path_name = os.path.join(
            self.imgs_dir, t2vmodel_name, str(videoname_idx))
        spatial_feat_name = os.path.join(
            self.spatialFeat, t2vmodel_name, str(videoname_idx))
        temporal_feat_name = os.path.join(
            self.temporalFeat, t2vmodel_name, str(videoname_idx))

        video_channel = 3

        # 224
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        # fix random
        # seed = np.random.randint(20231001)
        # random.seed(seed)

        frame_len = len(os.listdir(img_path_name))

        # test all frames
        start_index = 0
        count = frame_len//self.frame_num
        # print(f'start from {start_index}th frame, collect {frame_len} frames')
        # first_file=sorted(tmp,key=lambda x:int(x.split('.')[0]))[0]
        # start_index=int(first_file[:-4])
        transformed_video_all=[]
        resized_lp_all=[]
        
        for j in range(count):
            transformed_video = torch.zeros(
                [self.frame_num, video_channel, video_height_crop, video_width_crop])
            resized_lp = torch.zeros([self.frame_num, 5*256])
            for i in range(start_index, start_index+self.frame_num):
                imge_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i-start_index] = read_frame
                resized_lp[i-start_index] = torch.from_numpy(
                np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)
            transformed_video_all.append(transformed_video)
            resized_lp_all.append(resized_lp)
            start_index+=self.frame_num
            
        # read 3D features
        # if self.feature_type == 'Slow':
        #     feature_folder_name = os.path.join(self.temporalFeat, videoname_idx)
        #     transformed_feature = torch.zeros([self.frame_num, 2048])
        #     for i in range(self.frame_num):
        #         i_index = i
        #         feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
        #         feature_3D = torch.from_numpy(feature_3D)
        #         feature_3D = feature_3D.squeeze()
        #         transformed_feature[i] = feature_3D
        # elif self.feature_type == 'Fast':
        #     # feature_folder_name = os.path.join(temporal_feat_name, )
        #     transformed_feature = torch.zeros([self.frame_num, 256])
        #     for i in range(self.frame_num):
        #         # print(i)
        #         i_index = i   #TODO
        #         feature_3D = np.load(os.path.join(temporal_feat_name,str(i_index) + 'fast_feature.npy'))
        #         feature_3D = torch.from_numpy(feature_3D)
        #         feature_3D = feature_3D.squeeze()
        #         transformed_feature[i] = feature_3D
        # elif self.feature_type == 'SlowFast':
        #     feature_folder_name = os.path.join(self.temporalFeat, videoname_idx)
        #     transformed_feature = torch.zeros([self.frame_num, 2048+256])
        #     for i in range(self.frame_num):
        #         i_index = i
        #         feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
        #         feature_3D_slow = torch.from_numpy(feature_3D_slow)
        #         feature_3D_slow = feature_3D_slow.squeeze()
        #         feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
        #         feature_3D_fast = torch.from_numpy(feature_3D_fast)
        #         feature_3D_fast = feature_3D_fast.squeeze()
        #         feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
        #         transformed_feature[i] = feature_3D
        transformed_feature = []

        return transformed_video_all, transformed_feature, video_score, resized_lp_all, count
