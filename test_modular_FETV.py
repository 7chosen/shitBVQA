import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from train_dataloader import VideoDataset_test


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(
            feat_len=8)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # load the trained model
    print(f'loading the trained model {config.trained_model}')
    model.load_state_dict(torch.load(config.trained_model))

    # training data

    if config.database == 'KoNViD-1k':
        videos_dir = 'data/konvid1k_image_all_fps1'
        datainfo = 'data/KoNViD-1k_data.mat'
        lp_dir = 'data/konvid1k_LP_ResNet18'
        feature_dir = 'data/KoNViD-1k_slowfast'

    elif config.database == 'FETV':
        lp_dir = 'data/FETV_spatial_all_frames'
        # temporal features
        feature_dir = 'data/FETVtemporal'
        # extract frames
        videos_dir = 'data/base_all_frames'
        datainfo = 'data/spaAVGmos.json'

    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  # transforms.Resize(config.resize),
         transforms.CenterCrop(config.crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    testset = VideoDataset_test(videos_dir, feature_dir,
                                     lp_dir, datainfo, transformations_test,
                                     config.crop_size, 'Fast', prompt_num=config.prompt_num)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()

        label = np.zeros([len(testset)])
        y_output_b = np.zeros([len(testset)])
        y_output_s = np.zeros([len(testset)])
        y_output_t = np.zeros([len(testset)])
        y_output_st = np.zeros([len(testset)])
        for i, (video, feature_3D, mos, lp, count) in enumerate(test_loader):
            outputs_b=outputs_s=outputs_t=outputs_st=0
            label[i] = mos.item()
            for j in range(count):
                video[j] = video[j].to(device)
                # feature_3D = feature_3D.to(device)
                lp[j] = lp[j].to(device)
                b, s, t, st = model(video[j], feature_3D, lp[j])
                outputs_b+=b
                outputs_s+=s
                outputs_t+=t
                outputs_st+=st
                # print(f'testing... current count is {j}, the ret is {st}')
            count=count.to(device)
            outputs_b, outputs_s, outputs_t, outputs_st = \
                    outputs_b/count, outputs_s/count, outputs_t/count, outputs_st/count

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
        # print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
        # print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='FETV')
    # parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str,
                        default='ViTbCLIP_SpatialTemporal_modular_dropout')
    parser.add_argument('--prompt_num', type=int, default=619)

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str,
                        default='ckpts_modular/8frames_spa_no_weight.pth')
    # parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--feature_type', type=str, default='fast')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    config = parser.parse_args()

    main(config)
