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
# from config import FETV_T2V_model

class VideoDataset_train(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, database, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 crop_size, prompt_num, frame_num=8, seed=0):
        super(VideoDataset_train, self).__init__()

        data_file=pd.read_csv(mosfile_path)
        if database == 'FETV':
            model_name=data_file.iloc[:,1]
            prompt_file=data_file.iloc[:,2]
            spa_file=data_file.iloc[:,3]
            tem_file=data_file.iloc[:,4]
            ali_file=data_file.iloc[:,5]
        elif database == 'LGVQ':
            model_name=data_file.iloc[:,0]
            prompt_file = data_file.iloc[:,1]
            spa_file=data_file.iloc[:,2]
            tem_file=data_file.iloc[:,3]
            ali_file=data_file.iloc[:,4]     

        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(prompt_num)
        train_index = index_rd[0:int(prompt_num*0.7)]
        
        self.img_path_dir=[]
        self.s_feat_name=[]
        self.t_feat_name=[]
        self.prompt_name=[]
        self.score=[]
        
        for idx in train_index:
            idx_copy=idx
            for j in range(len(os.listdir(imgs_dir))):
                mdl=model_name[idx_copy]
                
                self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                self.prompt_name.append(prompt_file[idx_copy])
                
                self.img_path_dir.append(os.path.join(imgs_dir,mdl,prompt_file[idx_copy]))
                self.s_feat_name.append(os.path.join(spatialFeat,mdl,prompt_file[idx_copy]))
                self.t_feat_name.append(os.path.join(temporalFeat,mdl,prompt_file[idx_copy]))
                idx_copy+=prompt_num
        
        self.crop_size = crop_size
        self.transform = transform
        self.frame_num = frame_num

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        
        score=self.score[idx]
        for ele in score:
            ele=torch.FloatTensor(np.array(float(ele)))
        prmt=self.prompt_name[idx]
        img_path_name = self.img_path_dir[idx]
        spatial_feat_name = self.s_feat_name[idx]
        temporal_feat_name = self.t_feat_name[idx] 

        video_channel = 3
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
        feature_3D = np.load(temporal_feat_name+'.npy')
        feature_3D = torch.from_numpy(feature_3D)
        frame_len=len(os.listdir(img_path_name))
        count=int(frame_len/self.frame_num)
        final_trans_img = torch.zeros([self.frame_num,video_channel,video_height_crop,video_width_crop])
        final_spa = torch.zeros(self.frame_num,5*256)
        final_tem = []

        # if flag == 1, use the local sparse: random pick {self.frame_num} consecutive frames, return a list
        # else use the global sparse: select all frames by {self.frame_num} frames a chunk, return a 2-dim list
        flag = random.randint(0, 1)
        # global dense
        # select {self.frame_num} imgs with a interval
        if flag == 0:
            # make sure it always select the first {self.frame_num} frames
            frame_len = self.frame_num * count
            random_range = list([x] for x in range(0,frame_len,count))
            for idx,i in enumerate(range(0,frame_len,count)):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                final_trans_img[idx] = read_frame
                final_spa[idx] = torch.from_numpy(
                    np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)
            
        # flage = 1, use random {self.frame_num} consecutive imgs
        # local sparse
        else:
            if frame_len - self.frame_num == 0:
                start_index=0
            else:
                frame_len = frame_len - 1
                start_index = random.randint(0, frame_len-self.frame_num)
                
            random_range = list(range(start_index, start_index+self.frame_num))
            for idx,i in enumerate(range(start_index, start_index + self.frame_num)):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                final_trans_img[idx] = read_frame
                final_spa[idx] = torch.from_numpy(
                    np.load(os.path.join(spatial_feat_name, f'{i}.npy'))).view(-1)

        # read 3D features
        # 1*256*frames*1*1 ->  self.frame_num*256
        final_tem=feature_3D[:,:,random_range,:,:].squeeze().permute(1,0)
        
        if len(final_spa) != self.frame_num:
            raise Exception('sample is not right')
        
        return final_trans_img,  final_tem,  final_spa, prmt, score


class VideoDataset_val_test(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, database, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 database_name, crop_size, prompt_num, seed=0, frame_num=8):
        super(VideoDataset_val_test, self).__init__()

        data_file=pd.read_csv(mosfile_path)
        if database == 'FETV':
            model_name=data_file.iloc[:,1]
            prompt_file=data_file.iloc[:,2]
            spa_file=data_file.iloc[:,3]
            tem_file=data_file.iloc[:,4]
            ali_file=data_file.iloc[:,5]
        elif database == 'LGVQ':
            model_name=data_file.iloc[:,0]
            prompt_file = data_file.iloc[:,1]
            spa_file=data_file.iloc[:,2]
            tem_file=data_file.iloc[:,3]
            ali_file=data_file.iloc[:,4] 
        
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(prompt_num)
        val_index = index_rd[int(prompt_num * 0.7):int(prompt_num * 0.8)]
        test_index = index_rd[int(prompt_num * 0.8):]

        self.img_path_dir=[]
        self.s_feat_name=[]
        self.t_feat_name=[]
        self.prompt_name=[]
        self.score=[]
        
        if database_name == 'val':
            for idx in val_index:
                # str_idx=str(idx)
                # idx_copy=idx
                # for j in range(4):
                #     mdl=model_name[idx_copy]
                #     self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                #     self.prompt_name.append(prompt_file[idx_copy])

                #     self.img_path_dir.append(os.path.join(imgs_dir,mdl,str_idx))
                #     self.s_feat_name.append(os.path.join(spatialFeat,mdl,str_idx))
                #     self.t_feat_name.append(os.path.join(temporalFeat,mdl,str_idx))
                #     idx_copy+=prompt_num
                idx_copy=idx
                for j in range(len(os.listdir(imgs_dir))):
                    mdl=model_name[idx_copy]
                    
                    self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                    self.prompt_name.append(prompt_file[idx_copy])
                    
                    self.img_path_dir.append(os.path.join(imgs_dir,mdl,prompt_file[idx_copy]))
                    self.s_feat_name.append(os.path.join(spatialFeat,mdl,prompt_file[idx_copy]))
                    self.t_feat_name.append(os.path.join(temporalFeat,mdl,prompt_file[idx_copy]))
                    idx_copy+=prompt_num
                    
        if database_name == 'test':
            for idx in test_index:
                idx_copy=idx
                for j in range(len(os.listdir(imgs_dir))):
                    mdl=model_name[idx_copy]
                    
                    self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                    self.prompt_name.append(prompt_file[idx_copy])
                    
                    self.img_path_dir.append(os.path.join(imgs_dir,mdl,prompt_file[idx_copy]))
                    self.s_feat_name.append(os.path.join(spatialFeat,mdl,prompt_file[idx_copy]))
                    self.t_feat_name.append(os.path.join(temporalFeat,mdl,prompt_file[idx_copy]))
                    idx_copy+=prompt_num 

                    
        self.crop_size = crop_size
        self.transform = transform
        self.frame_num = frame_num
        
    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):

        score=self.score[idx]
        for ele in score:
            ele=torch.FloatTensor(np.array(float(ele)))
        prmt=self.prompt_name[idx]
        
        
        img_path_name = self.img_path_dir[idx]
        spatial_feat_name = self.s_feat_name[idx]
        temporal_feat_name = self.t_feat_name[idx] 

        video_channel = 3
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
        feature_3D = np.load(temporal_feat_name+'.npy')
        feature_3D = torch.from_numpy(feature_3D)
        frame_len=len(os.listdir(img_path_name))
        count=int(frame_len/self.frame_num)

        # test all clips of the video to get the average score
        # local
        img_l = []
        s_l = []
        t_l = []
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
            img_l.append(trans_img)
            t_l.append(fast_feature)
            s_l.append(lap_feat)
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
            img_l.append(trans_img)
            t_l.append(fast_feature)
            s_l.append(lap_feat)

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
        # global 
        t_g = torch.zeros([self.frame_num,256])
        s_g = torch.zeros([self.frame_num,256*5])
        img_g = torch.zeros([self.frame_num,video_channel,video_height_crop,video_width_crop])
        # make sure always select the first {self.frame_num} frames
        count2=int(frame_len/self.frame_num)
        frame_len=count2*self.frame_num
        fixed_range=list([x] for x in range(0,frame_len,count2))
        mid_value = feature_3D[:,:,fixed_range,:,:].squeeze().permute(1,0)
        t_g=mid_value
        for idx,i in enumerate(range(0,frame_len,count2)):
            img_name = os.path.join(img_path_name,f'{i}.png')
            mid_value1=Image.open(img_name).convert('RGB')
            mid_value1=self.transform(mid_value1)
            img_g[idx]=mid_value1
            mid_value3 = torch.from_numpy(np.load(os.path.join(spatial_feat_name,f'{i}.npy'))).view(-1)    
            s_g[idx] = mid_value3

        # ========
        
        if len(s_l) != count:
            raise Exception('local sample is not right')
        
        if len(s_g) != self.frame_num:
            print(len(s_g))
            print(self.frame_num)
            raise Exception('global sample is not right')

    #   return img0,img1,tem0,tem1,spa0,spa1,mos,count
        return img_l, img_g, t_l, t_g, \
            s_l, s_g, score, count, prmt





v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    # step = len(vid_list) // fnum
    # vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0,1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    # print(vid_tube.shape)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).float()
    vid_tube=vid_tube.squeeze(0)
    return vid_tube


class viCLIP_trainDT(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, temporalFeat, spatialFeat, mosfile_path, transform,
                 crop_size, prompt_num, frame_num=8, seed=0):
        super(viCLIP_trainDT, self).__init__()

        data_file=pd.read_csv(mosfile_path)
        model_name=data_file.iloc[:,1]
        prompt_file=data_file.iloc[:,2]
        spa_file=data_file.iloc[:,3]
        tem_file=data_file.iloc[:,4]
        ali_file=data_file.iloc[:,5]

        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(prompt_num)
        train_index = index_rd[0:int(prompt_num*0.7)]
        
        self.img_path_dir=[]
        self.prompt_name=[]
        self.score=[]
        
        for idx in train_index:
            str_idx=str(idx)
            idx_copy=idx
            for j in range(4):
                mdl=model_name[idx_copy]
                self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                self.prompt_name.append(prompt_file[idx_copy])
                self.img_path_dir.append(os.path.join(imgs_dir,mdl,str_idx))
                idx_copy+=prompt_num
        
        self.transform=transform
        self.crop_size = crop_size
        self.frame_num = frame_num

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        
        score=self.score[idx]
        for ele in score:
            ele=torch.FloatTensor(np.array(float(ele)))
        prmt=self.prompt_name[idx]
        img_path_name = self.img_path_dir[idx]
        frame_len=len(os.listdir(img_path_name))
        count=int(frame_len/self.frame_num)
        # final_trans_img = torch.zeros([self.frame_num,video_channel,video_height_crop,video_width_crop])
        final_trans_img=[]


        # if flag == 1, use the local sparse: random pick {self.frame_num} consecutive frames
        # else flag = 0 use the global sparse: select all frames by {self.frame_num} frames a chunk
        flag = random.randint(0, 1)
        # global dense
        # select {self.frame_num} imgs with a interval
        if flag == 0:
            # make sure it always select the first {self.frame_num} frames
            frame_len = self.frame_num * count
            for idx,i in enumerate(range(0,frame_len,count)):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                final_trans_img.append(read_frame)

            
        # flage = 1, use random {self.frame_num} consecutive imgs
        # local sparse
        else:
            frame_len = frame_len - 1
            start_index = random.randint(0, frame_len-self.frame_num)
            for idx,i in enumerate(range(start_index, start_index + self.frame_num)):
                img_name = os.path.join(img_path_name, f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                final_trans_img.append(read_frame)
        imgs=torch.stack(final_trans_img)
        # print(imgs.shape)
        
        
        return imgs, prmt, score

        
class viCLIP_vandtDT(data.Dataset):
    """Read data from the original dataset for feature extraction
    """

    def __init__(self, imgs_dir, mosfile_path, transfrom,
                 database_name,crop_size, prompt_num, frame_num=8, seed=0):
        super(viCLIP_vandtDT, self).__init__()

        data_file=pd.read_csv(mosfile_path)
        model_name=data_file.iloc[:,1]
        prompt_file=data_file.iloc[:,2]
        spa_file=data_file.iloc[:,3]
        tem_file=data_file.iloc[:,4]
        ali_file=data_file.iloc[:,5]

        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(prompt_num)
        val_index = index_rd[int(prompt_num * 0.7):int(prompt_num * 0.8)]
        test_index = index_rd[int(prompt_num * 0.8):]
        
        self.img_path_dir=[]
        self.prompt_name=[]
        self.score=[]
        
        if database_name == 'val':
            for idx in val_index:
                str_idx=str(idx)
                idx_copy=idx
                for j in range(4):
                    mdl=model_name[idx_copy]
                    self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                    self.prompt_name.append(prompt_file[idx_copy])                    
                    self.img_path_dir.append(os.path.join(imgs_dir,mdl,str_idx))
                    idx_copy+=prompt_num
        if database_name == 'test':
            for idx in test_index:
                str_idx=str(idx)
                idx_copy=idx
                for j in range(4):
                    mdl=model_name[idx_copy]
                    self.score.append([tem_file[idx_copy],spa_file[idx_copy],ali_file[idx_copy]])
                    self.prompt_name.append(prompt_file[idx_copy])                    
                    self.img_path_dir.append(os.path.join(imgs_dir,mdl,str_idx))
                    idx_copy+=prompt_num
        self.crop_size = crop_size
        self.frame_num = frame_num
        self.transform=transfrom

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        
        score=self.score[idx]
        for ele in score:
            ele=torch.FloatTensor(np.array(float(ele)))
        prmt=self.prompt_name[idx]
        img_path_name = self.img_path_dir[idx]
        frame_len=len(os.listdir(img_path_name))
        count=int(frame_len/self.frame_num)
        # local
        img_l=[]
        start_idx=0

        for i in range(count):
            tem_img=[]
            for j in range(start_idx,start_idx+self.frame_num):
                img_name=os.path.join(img_path_name,f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                tem_img.append(read_frame)
            ret=torch.stack(tem_img)
            img_l.append(ret)
            start_idx+=self.frame_num
        if 0< frame_len-start_idx < self.frame_num:
            count+=1
            tem_img=[]
            start_idx=frame_len-self.frame_num
            for j in range(start_idx,frame_len):
                img_name=os.path.join(img_path_name,f'{i}.png')
                read_frame = Image.open(img_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                tem_img.append(read_frame)
            ret=torch.stack(tem_img)
            img_l.append(ret)


        # global
        img_g = []
        count2=int(frame_len/self.frame_num)
        frame_len=count2*self.frame_num
        for idx,i in enumerate(range(0,frame_len,count2)):
            img_name = os.path.join(img_path_name,f'{i}.png')
            read_frame = Image.open(img_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            img_g.append(read_frame)
        img_g=torch.stack(img_g)
        
        
        return img_l,img_g,prmt,score,count