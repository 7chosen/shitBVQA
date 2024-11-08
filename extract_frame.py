import numpy as np
import os
import scipy.io as scio
from config import FETV_T2V_model,LGVQ_T2V_model
from PIL import Image
import cv2


def get_index(filepath):
    frame_num = len(os.listdir(filepath))
    file=os.listdir(filepath)
    video_name=sorted(file,key=lambda x:int(x.split('.')[0]))[0]
    first_idx=int(video_name[:-4])
    return first_idx,frame_num



def extract_frame_spa(spa_feat_path,video_path ,save_folder):
    '''
        vid_path: eg. cogvideo/0
    '''
    for feat_folder in os.listdir(spa_feat_path):
        print(f'process the {feat_folder}th video')
        first_idx,frame_num=get_index(os.path.join(spa_feat_path,feat_folder))
        rge=list(range(first_idx,first_idx+frame_num))
        save_fd=save_folder+f'/{feat_folder}'
        if not os.path.exists(save_fd):
            os.makedirs(save_fd)
          
        img_folder=[]
            
        if 'cogvid' in video_path:
            video_name=os.path.join(video_path,feat_folder+'.gif')
            cap=Image.open(video_name)
            try:
                for i in range(cap.n_frames):
                    cap.seek(i)
                    if i in rge:
                        img_folder.append(cap.copy())
                        # frame_rgb=cap.convert('RGB')
            except EOFError:
                raise Exception('no frame')
            for frame in img_folder:
                frame.save(save_fd+f'/{first_idx}.png')
                first_idx+=1
        else:
            video_name=os.path.join(video_path,feat_folder+'.mp4')
            cap=cv2.VideoCapture(video_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES,first_idx)
            for idx in range(first_idx,first_idx+frame_num):
                ret,frame = cap.read()
                if not ret:
                    raise Exception('no frame detect')
                # img_folder.append(frame)
                cv2.imwrite(save_fd+f'/{first_idx}.png',frame)
                first_idx+=1


def extract_frame(video_path,save_folder):
    
    for num,video_name in enumerate(os.listdir(video_path)):
        print(f'process the {num}th video')
        save_fd=save_folder+f'/{video_name[:-4]}'
        if not os.path.exists(save_fd):
            os.makedirs(save_fd)
        video_name=os.path.join(video_path,video_name)
        cap=cv2.VideoCapture(video_name)
        name=0
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            cv2.imwrite(save_fd+f'/{name}.png',frame)
            name+=1
    
if __name__ == "__main__":
    
    # spa_feat_path='data/FETV_spatial_all_frames/'
    save_folder='test/'
    video_path='/home/user/Documents/vqadata/FETV/'
    # if not os.path.exists(save_folder):
        # os.makedirs(save_folder)
    for t2vmdl in FETV_T2V_model:
        # if t2vmdl == 'cogvideo':
        #     pass
        print(f'========{t2vmdl}=========')
        # spa=spa_feat_path+t2vmdl
        v_p=video_path+t2vmdl+'/videos'
        s_f=save_folder+t2vmdl
        # extract_frame_spa(spa,v_p,s_f)
        
        extract_frame(v_p,s_f)
        # break