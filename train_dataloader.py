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
from config import FETV_T2V_model


def read_float_with_comma(num):
    return float(num.replace(",", "."))


class VideoDataset_train(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 crop_size, prompt_num, frame_num=8, seed=0):
        super(VideoDataset_train, self).__init__()

        with open(mosfile_path, 'r') as f:
            mos = json.load(f)
        self.video_names = []
        self.score = []
        random.seed(seed)
        np.random.seed(seed)

        index_rd = np.random.permutation(prompt_num)
        train_index = index_rd[0:int(prompt_num*0.7)]
        for idx, t2vmdl in enumerate(FETV_T2V_model):
            for i in train_index:
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
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        t2vmodel_name = video_name[:-3]
        # remove leading zeros
        videoname_idx = int(video_name[-3:])
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        img_path_name = os.path.join(self.imgs_dir, t2vmodel_name, str(videoname_idx))
        spatial_feat_name = os.path.join(self.spatialFeat, t2vmodel_name, str(videoname_idx))
        temporal_feat_name = os.path.join(self.temporalFeat, t2vmodel_name, str(videoname_idx))

        video_channel = 3

        # 224
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        # if flag == 1, use the global sparse: random pick {self.frame_num} consecutive frames, return a list
        # else use the local sparse: select all frames by {self.frame_num} frames a chunk, return a 2-dim list
        feature_3D = np.load(temporal_feat_name+'.npy')
        feature_3D = torch.from_numpy(feature_3D)
        frame_len=len(os.listdir(img_path_name))
        count=int(frame_len/self.frame_num)

        # flag = random.randint(0, 1)
        flag=1
        
        trans_img_all = []
        resized_lap_all = []
        fast_feature_all = []
        trans_img_1 = []
        resized_lap_1 = []
        fast_feature_1 = []
        

        # using local
        if flag == 0:
 
            start_index=0

            for i in range(count):
                trans_img = torch.zeros([self.frame_num, video_channel, video_height_crop, video_width_crop])
                lap_feat = torch.zeros([self.frame_num, 5*256])
                fast_feature = torch.zeros([self.frame_num, 256])
                specified_rge = list(
                    range(start_index, start_index + self.frame_num))
                for j in range(start_index, start_index + self.frame_num):
                    img_name = os.path.join(img_path_name, f'{j}.png')
                    read_frame = Image.open(img_name).convert('RGB')
                    read_frame = self.transform(read_frame)
                    trans_img[j-start_index] = read_frame
                    lap_feat[j-start_index] = torch.from_numpy(
                        np.load(os.path.join(spatial_feat_name, f'{j}.npy'))).view(-1)
                fast_feature = feature_3D[:, :, specified_rge, :, :]
                fast_feature = fast_feature.squeeze().permute(1, 0)
                trans_img_all.append(trans_img)
                fast_feature_all.append(fast_feature)
                resized_lap_all.append(lap_feat)
                start_index += self.frame_num
            if 0 < frame_len - start_index < self.frame_num:
                count += 1
                trans_img = torch.zeros([self.frame_num, video_channel, video_height_crop, video_width_crop])
                lap_feat = torch.zeros([self.frame_num, 5*256])
                fast_feature = torch.zeros([self.frame_num, 256])
                start_index = frame_len-self.frame_num
                for i in range(start_index, frame_len):
                    img_name = os.path.join(img_path_name, f'{i}.png')
                    read_frame = Image.open(img_name).convert('RGB')
                    read_frame = self.transform(read_frame)
                    trans_img[i-start_index] = read_frame
                    lap_feat[i-start_index] = torch.from_numpy(
                        np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)
                fast_feature = feature_3D[:, :, -8:, :, :]
                fast_feature = fast_feature.squeeze().permute(1, 0)
                trans_img_all.append(trans_img)
                fast_feature_all.append(fast_feature)
                resized_lap_all.append(lap_feat)

        # flage = 1, use random {self.frame_num} consecutive imgs
        else:
            
            frame_len = len(os.listdir(img_path_name))-1
            start_index = random.randint(0, frame_len-self.frame_num)
            random_range = list(range(start_index, start_index+self.frame_num))

            trans_img_1 = torch.zeros(
                [self.frame_num, video_channel, video_height_crop, video_width_crop])
            resized_lap_1 = torch.zeros([self.frame_num, 5*256])
            for i in range(start_index, start_index + self.frame_num):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                trans_img_1[i-start_index] = read_frame
                resized_lap_1[i-start_index] = torch.from_numpy(
                    np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)

            # read 3D features
            # 1*256*frames*1*1
            fast_feature_1= feature_3D[:, :, random_range, :, :]
            fast_feature_1= fast_feature_1.squeeze().permute(1, 0)
            # return shape of [8,256]
            # self.frame_num*256
        # print(1)

        # print('1 ',len(trans_img_all))
        # print('2 ',len(fast_feature_all))
        # print('3 ',len(resized_lap_all))
 
        return trans_img_all, trans_img_1, fast_feature_all, fast_feature_1, \
                resized_lap_all, resized_lap_1, video_score, count, flag


class VideoDataset_val_test(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 database_name, crop_size, prompt_num, seed=0, frame_num=8):
        super(VideoDataset_val_test, self).__init__()

        with open(mosfile_path, 'r') as f:
            mos = json.load(f)
        self.video_names = []
        self.score = []
        random.seed(seed)
        np.random.seed(seed)

        index_rd = np.random.permutation(prompt_num)
        val_index = index_rd[int(prompt_num * 0.7):int(prompt_num * 0.8)]
        test_index = index_rd[int(prompt_num * 0.8):]
        if database_name == 'val':
            for idx, t2vmdl in enumerate(FETV_T2V_model):
                for i in val_index:
                    suffix = "{:03}".format(i)
                    self.video_names.append(t2vmdl+suffix)
                    self.score.append(mos[idx][i])
        else:
            for idx, t2vmdl in enumerate(FETV_T2V_model):
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
        frame_len = len(os.listdir(img_path_name))
        count = int(frame_len/self.frame_num)        

        # test all clips of the video to get the average score
        # global sparse
        trans_img_glob = []
        spa_glob = []
        tem_glob = []
        feature_3D = np.load(temporal_feat_name+'.npy')
        feature_3D = torch.from_numpy(feature_3D)
        start_index = 0
        for i in range(count):
            trans_img = torch.zeros([self.frame_num, video_channel, video_height_crop, video_width_crop])
            lap_feat = torch.zeros([self.frame_num, 5*256])
            fast_feature = torch.zeros([self.frame_num, 256])
            specified_rge = list(range(start_index, start_index+self.frame_num))
            for j in range(start_index, start_index+self.frame_num):
                img_name = os.path.join(img_path_name, f'{j}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                trans_img[j-start_index] = read_frame
                lap_feat[j-start_index] = torch.from_numpy(
                    np.load(os.path.join(spatial_feat_name, f'{j}.npy'))).view(-1)
            fast_feature = feature_3D[:, :, specified_rge, :, :]
            fast_feature = fast_feature.squeeze().permute(1, 0)
            trans_img_glob.append(trans_img)
            tem_glob.append(fast_feature)
            spa_glob.append(lap_feat)
            start_index += self.frame_num
        if 0 < frame_len - start_index < self.frame_num:
            count += 1
            trans_img = torch.zeros(
                [self.frame_num, video_channel, video_height_crop, video_width_crop])
            lap_feat = torch.zeros([self.frame_num, 5*256])
            fast_feature = torch.zeros([self.frame_num, 256])
            start_index = frame_len-self.frame_num
            for i in range(start_index, frame_len):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name).convert('RGB')
                read_frame = self.transform(read_frame)
                trans_img[i-start_index] = read_frame
                lap_feat[i-start_index] = torch.from_numpy(
                    np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)
            fast_feature = feature_3D[:, :, -8:, :, :]
            fast_feature = fast_feature.squeeze().permute(1, 0)
            trans_img_glob.append(trans_img)
            tem_glob.append(fast_feature)
            spa_glob.append(lap_feat)

        # test {self.frame_num} consecutive frames, a clip of the vid
        '''
        frame_len -= 1
        start_index = random.randint(0, frame_len-self.frame_num)
        random_range = list(range(start_index, start_index+self.frame_num))
        trans_img_global = torch.zeros(
            [self.frame_num, video_channel, video_height_crop, video_width_crop])
        resized_lap_global = torch.zeros([self.frame_num, 5*256])
        for i in range(start_index, start_index+self.frame_num):
            img_name = os.path.join(img_path_name, f'{i}.png')
            read_frame = Image.open(img_name).convert('RGB')
            read_frame = self.transform(read_frame)
            trans_img_global[i-start_index] = read_frame
            resized_lap_global[i-start_index] = torch.from_numpy(
                np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)
        feature_3D_global = feature_3D[:, :, random_range, :, :]
        feature_3D_global = feature_3D_global.squeeze().permute(1, 0)
        '''

        # test select {self.frame_num} frames with a interval T
        # local dense
        tem_local = torch.zeros([self.frame_num,256])
        spa_local = torch.zeros([self.frame_num,256*5])
        trans_img_local = torch.zeros([self.frame_num,video_channel,video_height_crop,video_width_crop])
        # make sure always select the first 8 frames
        count2=int(frame_len/self.frame_num)
        frame_len=count2*self.frame_num
        fixed_range=list([x] for x in range(0,frame_len,count2))
        mid_value = feature_3D[:,:,fixed_range,:,:].squeeze().permute(1,0)
        tem_local=mid_value
        for idx,i in enumerate(range(0,frame_len,count2)):
            img_name = os.path.join(img_path_name,f'{i}.png')
            mid_value1=Image.open(img_name).convert('RGB')
            mid_value1=self.transform(mid_value1)
            trans_img_local[idx]=mid_value1
            mid_value3 = torch.from_numpy(np.load(os.path.join(spatial_feat_name,f'{i}.npy'))).view(-1)    
            spa_local[idx]=mid_value3

        # ========
        
        if len(spa_glob) != count:
            raise Exception('global sparse sample is not right')
        
        if len(spa_local) != self.frame_num:
            print(len(spa_local))
            print(self.frame_num)
            raise Exception('local dense sample is not right')

    #   return img0,img1,tem0,tem1,spa0,spa1,mos,count
        return trans_img_glob, trans_img_local, tem_glob, tem_local, \
            spa_glob, spa_local, video_score, count

