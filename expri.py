
import scipy.io as scio
import json
import os
import pandas as pd
import numpy as np
import torch

from utils import performance_fit

from config import LGVQ_T2V_model
# with open(file_path, 'r') as f:
#     dt = json.load(f)


# model = ['cogvideo', 'text2video-zero', 'modelscope-t2v', 'zeroscope']
# all_mos = []
# for mdl in model:
#     current_mos = []
#     for i in range(619):
#         current_mos.append(dt[mdl][str(i)]['static_quality'])
# #     all_mos.append(current_mos)

# with open('data/spaAVGmos.json', 'w') as f:
#     json.dump(all_mos, f, indent=4)
# prefix_path='/home/user/Document/'

   
    


 
# with open('data/pyIQA_LGVQ_score/prompt_cls.json', 'r') as f:
#     dt = json.load(f)
# dt=dt["data"]
# # for item in dt:
# #     vid_path=item["path"]
# #     vid_name=item['filename']
# #     spa_mos=item['spatial_quality']
# y_s=[]
# label=[]
# for idx, name in enumerate(LGVQ_T2V_model):
#     df = pd.read_csv(f'data/pyIQA_LGVQ_score/clipiqaplus_csv/{name}.csv', header=None)
#     df.columns = ['name', 'mos']
#     iqa_mos = df['mos'].tolist()
#     iqa_name = df['name'].tolist()
#     for i in range(468):
#         y_s.append(iqa_mos[i])
#         # print(i)
#         for item in dt:
#             if name == item['code'] :
#                 if iqa_name[i] == item['filename'][:-4]:
#                     spa_mos=item['spatial_quality']
#                     label.append(spa_mos)
#     print(name)
#     # break
# PLCC,SRCC,KRCC,RMSE=performance_fit(label,y_s)
# print(PLCC,' ',SRCC,' ',KRCC,' ',RMSE)    





tem=np.load('data/FETV_temporal_all_frames/zeroscope/1.npy')
tem=torch.from_numpy(tem)
print(tem.shape)
