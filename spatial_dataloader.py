import glob
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
# import random
# from config import T2V_model
import cv2


def pyramidsGL(image, num_levels, dim=224):
    ''' Creates Gaussian (G) and Laplacian (L) pyramids of level "num_levels" from image im. 
    G and L are list where G[i], L[i] stores the i-th level of Gaussian and Laplacian pyramid, respectively. '''
    o_width = image.shape[1]
    o_height = image.shape[0]
    
    # resize
    
    # if both greater than 
    if o_width>(dim+num_levels) and o_height>(dim+num_levels) :
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width*f_height)/o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height*f_width)/o_width)
        else:
            f_width = f_height = dim

        height_step = int((o_height-f_height)/(num_levels-1))*(-1)
        width_step = int((o_width-f_width)/(num_levels-1))*(-1)
        height_list = [i for i in range(o_height, f_height-1, height_step)]
        width_list = [i for i in range(o_width, f_width-1, width_step)]
    
    # if both equal to
    elif o_width==dim or o_height==dim :
        height_list = [o_height for i in range(num_levels)]
        width_list = [o_width for i in range(num_levels)]

    else:
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width*f_height)/o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height*f_width)/o_width)
        else:
            f_width = f_height = dim
        image = cv2.resize(image, (f_width, f_height), interpolation = cv2.INTER_CUBIC)
        height_list = [f_height for i in range(num_levels)]
        width_list = [f_width for i in range(num_levels)]

    layer = image.copy()
    gaussian_pyramid = [layer]    #Gaussian Pyramid
    # print(gaussian_pyramid[2])
    
    laplacian_pyramid = []         # Laplacian Pyramid

    for i in range(num_levels-1):
        
        blur = cv2.GaussianBlur(gaussian_pyramid[i], (5,5), 5)
        layer = cv2.resize(blur, (width_list[i+1], height_list[i+1]), interpolation = cv2.INTER_CUBIC)
        gaussian_pyramid.append(layer)

        uplayer = cv2.resize(blur, (width_list[i], height_list[i]), interpolation = cv2.INTER_CUBIC)
        laplacian = cv2.subtract(gaussian_pyramid[i], uplayer)
        laplacian_pyramid.append(laplacian)

    gaussian_pyramid.pop(-1)
    return gaussian_pyramid, laplacian_pyramid


def resizedpyramids(gaussian_pyramid, laplacian_pyramid, num_levels, width, height):
    gaussian_pyramid_resized, laplacian_pyramid_resized=[],[]
    for i in range(num_levels-1):
        # img_gaussian_pyramid = cv2.resize(gaussian_pyramid[i],(width, height), interpolation = cv2.INTER_CUBIC)
        img_laplacian_pyramid = cv2.resize(laplacian_pyramid[i],(width, height), interpolation = cv2.INTER_CUBIC)
        # gaussian_pyramid_resized.append(img_gaussian_pyramid)
        laplacian_pyramid_resized.append(img_laplacian_pyramid)
    return gaussian_pyramid_resized, laplacian_pyramid_resized



class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, database_name, vids_dir,num_levels=6):
        super(VideoDataset, self).__init__()
        self.database_name = database_name
        self.vids_dir = vids_dir
        self.num_levels = num_levels
        # self.num_frames = num_frame

    def __len__(self):
        return len(os.listdir(self.vids_dir))

    def __getitem__(self, idx):
        
        if 'cogvid' in self.vids_dir:
            vid=f'{idx}.gif'
        else:
            vid=f'{idx}.mp4'
            
        vid_path=os.path.join(self.vids_dir,vid)
        if vid[-1] == '4':
            cap=cv2.VideoCapture(vid_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            cap=Image.open(vid_path)
            video_length=cap.n_frames
        if video_length == 0:
            raise Exception('no frame in this vid')
        print(f'total ftame is {video_length}')
        video_chunk=[]
        if vid[-1] == '4':
            while 1:
                ret,frame=cap.read()
                if not ret:
                    break
                video_chunk.append(frame)
        else:
            try:
                for i in range(cap.n_frames):
                    cap.seek(i)
                    frame_rgb=cap.convert('RGB')
                    frame_np=np.array(frame_rgb)
                    video_chunk.append(frame_np)
            except EOFError:
                pass 
                
        video_width = video_chunk[0].shape[0]
        video_height = video_chunk[0].shape[1]
        # ##########
        transformed_video = torch.zeros([video_length*(self.num_levels-1), 3, video_height, video_width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
#       
        # seed = np.random.randint(0)
        # np.random.seed(seed)
        # random.seed(0)
        for idx,extravt_frame in enumerate(video_chunk):
            gaussian_pyramid,laplacian_pyramid = pyramidsGL(extravt_frame, self.num_levels)
            _, laplacian_pyramid_resized = resizedpyramids(gaussian_pyramid, 
                    laplacian_pyramid, self.num_levels, video_width, video_height)
            for j in range(len(laplacian_pyramid_resized)):
                lp = laplacian_pyramid_resized[j]
                lp = cv2.cvtColor(lp, cv2.COLOR_BGR2RGB) # 
                lp = transform(lp)
                transformed_video[idx*(self.num_levels-1)+j] = lp

        return transformed_video,video_length 

class VideoDataset_LGVQ(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, database_name, vids_dir,num_levels=6):
        super(VideoDataset_LGVQ, self).__init__()
        self.database_name = database_name
        self.vids_dir = glob.glob(f'{vids_dir}/*.mp4')
        self.num_levels = num_levels
        # self.num_frames = num_frame

    def __len__(self):
        return len(os.listdir(self.vids_dir))

    def __getitem__(self, idx):

        vid_path = self.vids_dir[idx]
        vid_name = vid_path.split('/')[-1]
        vid_name=vid_name[:-4]
        cap=cv2.VideoCapture(vid_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_length == 0:
            raise Exception('no frame in this vid')
        print(f'total ftame is {video_length}')
        video_chunk=[]
        while 1:
            ret,frame=cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            video_chunk.append(frame)
            
        video_width = video_chunk[0].shape[0]
        video_height = video_chunk[0].shape[1]
        # ##########
        transformed_video = torch.zeros([video_length*(self.num_levels-1), 3, video_height, video_width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # seed = np.random.randint(0)
        # np.random.seed(seed)
        # random.seed(0)
        for i,extravt_frame in enumerate(video_chunk):
            gaussian_pyramid,laplacian_pyramid = pyramidsGL(extravt_frame, self.num_levels)
            _, laplacian_pyramid_resized = resizedpyramids(gaussian_pyramid, 
                    laplacian_pyramid, self.num_levels, video_width, video_height)
            for j in range(len(laplacian_pyramid_resized)):
                lp = laplacian_pyramid_resized[j]
                lp = cv2.cvtColor(lp, cv2.COLOR_BGR2RGB) # 
                lp = transform(lp)
                transformed_video[i*(self.num_levels-1)+j] = lp

        return transformed_video,video_length, vid_name