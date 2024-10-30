import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from train_dataloader import VideoDataset_val_test


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # load the trained model
    print(f'loading the trained model {config.trained_model}')
    model.load_state_dict(torch.load(config.trained_model,weights_only=1))

    # training data
    if config.database == 'FETV':
        lp_dir = 'data/FETV_spatial_all_frames'
        # temporal features
        feature_dir = 'data/FETV_temporal_all_frames'
        # extract frames
        imgs_dir = 'data/FETV_base_all_frames'
        datainfo = config.mosfile
        print('using the mos file: ',datainfo )
         
    transformations_vandt = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    testset = VideoDataset_val_test(imgs_dir, feature_dir,
                                    lp_dir, datainfo, transformations_vandt, 'test',
                                    config.crop_size, prompt_num=config.prompt_num, frame_num=config.frame_num)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()

        label = np.zeros([len(testset)])
        y_output_b = np.zeros([len(testset)])
        y_output_s = np.zeros([len(testset)])
        y_output_t = np.zeros([len(testset)])
        y_output_st = np.zeros([len(testset)])
        for i, (vid_chunk_g, vid_chunk_l, tem_g, tem_l,
                spa_g, spa_l, mos, count) in enumerate(test_loader):
            outputs_b = outputs_s = outputs_t = outputs_st = 0
            label[i] = mos.item()
            for j in range(count):
                vid_chunk_g[j] = vid_chunk_g[j].to(device)
                tem_g[j] = tem_g[j].to(device)
                spa_g[j] = spa_g[j].to(device)
                b, s, t, st = model(vid_chunk_g[j], tem_g[j], spa_g[j])
                outputs_b += b
                outputs_s += s
                outputs_t += t
                outputs_st += st
                # print(f'testing... current count is {j}, the ret is {st}')
            count = count.to(device)
            outputs_b, outputs_s, outputs_t, outputs_st = \
                outputs_b/count, outputs_s/count, outputs_t/count, outputs_st/count

            vid_chunk_l = vid_chunk_l.to(device)
            tem_l = tem_l.to(device)
            spa_l = spa_l.to(device)
            b1, s1, t1, st1 = model(vid_chunk_l, tem_l, spa_l)
            outputs_b = (outputs_b + b1) / 2
            outputs_s = (outputs_s + s1) / 2
            outputs_t = (outputs_t + t1) / 2
            outputs_st = (outputs_st + st1) / 2

            y_output_b[i] = outputs_b.item()
            y_output_s[i] = outputs_s.item()
            y_output_t[i] = outputs_t.item()
            y_output_st[i] = outputs_st.item()

        test_PLCC_b, test_SRCC_b, test_KRCC_b, test_RMSE_b = performance_fit(
            label, y_output_b)
        test_PLCC_s, test_SRCC_s, test_KRCC_s, test_RMSE_s = performance_fit(
            label, y_output_s)
        test_PLCC_t, test_SRCC_t, test_KRCC_t, test_RMSE_t = performance_fit(
            label, y_output_t)
        test_PLCC_st, test_SRCC_st, test_KRCC_st, test_RMSE_st = performance_fit(
            label, y_output_st)

        # print(config.database)
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='FETV')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporal_modular_dropout')
    parser.add_argument('--prompt_num', type=int, default=619)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--frame_num',type=int,default=8)

    # misc
    parser.add_argument('--trained_model', type=str,default='ckpts_modular/first_train.pth')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--mosfile',type=str,default='data/pyIQA_FETV_score/mosFile/temAVGmos.json')

    config = parser.parse_args()

    main(config)
