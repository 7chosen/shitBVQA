
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

   
    


 
with open('data/pyIQA_LGVQ_score/prompt_cls.json', 'r') as f:
    dt = json.load(f)
dt=dt["data"]
# for item in dt:
#     vid_path=item["path"]
#     vid_name=item['filename']
#     spa_mos=item['spatial_quality']
y_s=[]
label=[]
for idx, name in enumerate(LGVQ_T2V_model):
    df = pd.read_csv(f'data/pyIQA_LGVQ_score/liqe_csv/{name}.csv', header=None)
    df.columns = ['name', 'mos']
    iqa_mos = df['mos'].tolist()
    iqa_name = df['name'].tolist()
    for i in range(468):
        y_s.append(iqa_mos[i])
        # print(i)
        for item in dt:
            if name == item['code'] :
                if iqa_name[i] == item['filename'][:-4]:
                    spa_mos=item['spatial_quality']
                    label.append(spa_mos)
    print(name)
    # break
PLCC,SRCC,KRCC,RMSE=performance_fit(label,y_s)
print(SRCC,' ',KRCC,' ',PLCC,' ',RMSE)    




# # Step 1: Read the file and extract values
# srcc_values = []
# krcc_values = []
# plcc_values = []
# rmse_values = []

# with open('log.txt', 'r') as file:
#     for line in file:
#         # Split the line and extract values
#         parts = line.strip().split(', ')
#         srcc = float(parts[0].split(': ')[1])
#         krcc = float(parts[1].split(': ')[1])
#         plcc = float(parts[2].split(': ')[1])
#         rmse = float(parts[3].split(': ')[1])
        
#         # Step 2: Append to respective lists
#         srcc_values.append(srcc)
#         krcc_values.append(krcc)
#         plcc_values.append(plcc)
#         rmse_values.append(rmse)

# # Step 3: Compute medians
# srcc_median = np.median(srcc_values)
# krcc_median = np.median(krcc_values)
# plcc_median = np.median(plcc_values)
# rmse_median = np.median(rmse_values)

# # Print the medians
# print(f'SRCC Median: {srcc_median}')
# print(f'KRCC Median: {krcc_median}')
# print(f'PLCC Median: {plcc_median}')
# print(f'RMSE Median: {rmse_median}')


