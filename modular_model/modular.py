from collections import defaultdict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.models as models
import clip
from itertools import product
import PIL.Image as Image
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

class ViTbCLIP_SpatialTemporal_dropout(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_dropout, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        # clip_vit_b_pretrained_features = ViT_B_16.visual
        # self.feature_extraction = clip_vit_b_pretrained_features
        
        self.clip=ViT_B_16
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048


        self.sr = sr
        self.tr = tr


    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block
    
    def forward(self, x, tem_feat, spa_feat, prmt):
        x_size = x.shape
        # x: (batch * frames) x 3-channel x height x width
        # eg. 128*3*224*224
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # Using clip.text
        # input shape (batch_size*frame_num)*c*h*w, which h and w must be 224
        image_features = self.clip.encode_image(x)
        # (batch_size*frame_num) * 512

        # Normalize image features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # bs * frames * 512
        image_features = image_features.view(x_size[0], x_size[1], -1)

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        input_texts = []
        for i in range(x_size[0]):
            prompt = [prmt[i]]
            texts = [f"a photo with {s} spatial quality and {t} temporal quality, which matches {p}"
                     for s, t, p in product(qualitys, qualitys, prompt)]
            input_texts.append(torch.cat([clip.tokenize(texts)]))

        input_texts = torch.cat(input_texts, dim=0)

        text_features =  self.clip.encode_text(input_texts.to(x.device))
        # 200 * 512
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # bs * 25 * 512
        text_features = text_features.view(x_size[0],-1, text_features.size(1))

        x_all = []
        x_all_presoftmax = []
        for i in range(x_size[0]):
            visual_feat = image_features[i, ...]
            text_feat = text_features[i, ...]
            cos_sim = logit_scale * visual_feat @ text_feat.t()
            cos_sim_pre = cos_sim
            # print(cos_sim_pre.shape)
            cos_sim = torch.nn.functional.softmax(cos_sim, dim=1)
            x_all.append(cos_sim.unsqueeze(0))
            x_all_presoftmax.append(cos_sim_pre.unsqueeze(0))

        x_all = torch.cat(x_all, 0)
        x_all_presoftmax = torch.cat(x_all_presoftmax, 0)
        # 8*8*25
        
        x_all = x_all.view(-1, x_all.size(2))
        x_all_presoftmax = x_all_presoftmax.view(-1, x_all_presoftmax.size(2))

        logits_all = x_all.view(-1, len(qualitys), len(qualitys))

        xs = logits_all.sum(2)
        xt = logits_all.sum(1)
        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]
        # print(xt.shape)

        xs = xs.view(x_size[0],-1)
        xt = xt.view(x_size[0],-1)
        xa = xa.view(x_size[0], -1)
        # batch*frame_num
        xs = torch.mean(xs, dim=1).unsqueeze(1)
        xt = torch.mean(xt, dim=1).unsqueeze(1)
        xa = torch.mean(xa, dim=1).unsqueeze(1)

        assert xa.shape == xs.shape == xt.shape, "shape is not same"
        ones = torch.ones_like(xa)

        # spatial rectifier
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            spatial_t = self.spatialRec2(spa_feat)
            spatial_a = self.spatialRec3(spa_feat)

            # ax+b
            alphaS1 = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS1 = torch.add(alphaS1, ones)
            betaS1 = torch.chunk(spatial_s, 2, dim=1)[1]

            alphaS2 = torch.chunk(spatial_t, 2, dim=1)[0]
            alphaS2 = torch.add(alphaS2, ones)
            betaS2 = torch.chunk(spatial_t, 2, dim=1)[1]

            alphaS3 = torch.chunk(spatial_a, 2, dim=1)[0]
            alphaS3 = torch.add(alphaS3, ones)
            betaS3 = torch.chunk(spatial_a, 2, dim=1)[1]
        else:
            raise Exception('not implement')
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(alphaS1), xs), betaS1).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(alphaS2), xt), betaS2).squeeze(1)
        qs_a = torch.add(torch.mul(torch.abs(alphaS3), xa), betaS3).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal_s = self.temporalRec1(tem_feat)
            temporal_t = self.temporalRec2(tem_feat)
            temporal_a = self.temporalRec3(tem_feat)

            # ax+b
            alphaT1= torch.chunk(temporal_s, 2, dim=1)[0]
            alphaT1 = torch.add(alphaT1, ones)
            betaT1 = torch.chunk(temporal_s, 2, dim=1)[1]


            alphaT2= torch.chunk(temporal_t, 2, dim=1)[0]
            alphaT2 = torch.add(alphaT2, ones)
            betaT2 = torch.chunk(temporal_t, 2, dim=1)[1]

            alphaT3= torch.chunk(temporal_a, 2, dim=1)[0]
            alphaT3 = torch.add(alphaT3, ones)
            betaT3 = torch.chunk(temporal_a, 2, dim=1)[1]

        else:
            raise Exception('not implement')
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(alphaT1), xs), betaT1).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(alphaT2), xt), betaT2).squeeze(1)
        qt_a = torch.add(torch.mul(torch.abs(alphaT3), xa), betaT3).squeeze(1)


        if self.sr and self.tr:
            st_a1 = torch.sqrt(torch.abs(torch.mul(alphaS1, alphaT1)))
            st_b1 = torch.div(torch.add(betaS1, betaT1), 2)

            st_a2 = torch.sqrt(torch.abs(torch.mul(alphaS2, alphaT2)))
            st_b2 = torch.div(torch.add(betaS2, betaT2), 2)

            st_a3 = torch.sqrt(torch.abs(torch.mul(alphaS3, alphaT3)))
            st_b3 = torch.div(torch.add(betaS3, betaT3), 2)
            # modular_a_a = torch.sqrt(torch.abs(torch.mul(sa_t, ta_t)))


            qst_s = torch.add(torch.mul(st_a1, xs), st_b1).squeeze(1)
            qst_t = torch.add(torch.mul(st_a2, xt), st_b2).squeeze(1)
            qst_a = torch.add(torch.mul(st_a3, xa), st_b3).squeeze(1)
        # elif self.sr:
            # qst = qs
        # elif self.tr:
            # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)


        # 4 * batch_size
        t=torch.stack((xt.squeeze(1),qt_t,qs_t,qst_t))
        s=torch.stack((xs.squeeze(1),qt_s,qs_s,qst_s))
        a=torch.stack((xa.squeeze(1),qt_a,qs_a,qst_a))

        # if batch_size == 1, then return shape[4] directly
        if(x_size[0]==1):
            t,s,a = t.squeeze(1).to('cpu'),s.squeeze(1).to('cpu'),a.squeeze(1).to('cpu')

        return t, s, a


class ViTbCLIP_SpatialTemporal_dropout_meanpool(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_dropout_meanpool, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        # ckpt = torch.load('pickscore_vitb16.pt')
        # ViT_B_16.load_state_dict(ckpt['model_state_dict'])

        # clip_vit_b_pretrained_features = ViT_B_16.visual
        # self.feature_extraction = clip_vit_b_pretrained_features

        self.clip = ViT_B_16
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

        self.sr = sr
        self.tr = tr
    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def forward(self, x, tem_feat, spa_feat, prmt):
        x_size = x.shape
        # x: batch * frames x 3-channel x height x width
        # eg. 16*8*3*224*224

        input_texts = []
        for i in range(x_size[0]):
            prompt = [prmt[i]]
            texts = [f"a photo with {s} spatial quality and {t} temporal quality, which matches '{p}'"
                     for s, t, p in product(qualitys, qualitys, prompt)]
            input_texts.append(torch.cat([clip.tokenize(texts)]))

        input_texts = torch.cat(input_texts, dim=0)

        text_features =  self.clip.encode_text(input_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        text_features = text_features.view(x_size[0], -1, text_features.size(1))

        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        image_features = self.clip.encode_image(x)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.view(x_size[0], x_size[1], -1)
        image_features = torch.mean(image_features, dim=1)

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        x_all = []
        x_all_presoftmax = []
        for i in range(x_size[0]):
            visual_feat = image_features[i, ...]
            text_feat = text_features[i, ...]
            cos_sim = logit_scale * visual_feat @ text_feat.t()
            cos_sim_pre = cos_sim
            cos_sim = torch.nn.functional.softmax(cos_sim,dim=0)
            x_all.append(cos_sim.unsqueeze(0))
            x_all_presoftmax.append(cos_sim_pre.unsqueeze(0))

        x_all = torch.cat(x_all, 0)
        x_all_presoftmax = torch.cat(x_all_presoftmax, 0)

        logits_all = x_all.view(-1, len(qualitys), len(qualitys))

        xs = logits_all.sum(2)
        xt = logits_all.sum(1)
        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]

        xs = xs.unsqueeze(1)
        xt = xt.unsqueeze(1)
        xa = xa.unsqueeze(1)


        assert xa.shape == xs.shape == xt.shape, "shape is not same"
        ones = torch.ones_like(xa)

        # spatial rectifier
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            spatial_t = self.spatialRec2(spa_feat)
            spatial_a = self.spatialRec3(spa_feat)

            # ax+b
            alphaS1 = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS1 = torch.add(alphaS1, ones)
            betaS1 = torch.chunk(spatial_s, 2, dim=1)[1]

            alphaS2 = torch.chunk(spatial_t, 2, dim=1)[0]
            alphaS2 = torch.add(alphaS2, ones)
            betaS2 = torch.chunk(spatial_t, 2, dim=1)[1]

            alphaS3 = torch.chunk(spatial_a, 2, dim=1)[0]
            alphaS3 = torch.add(alphaS3, ones)
            betaS3 = torch.chunk(spatial_a, 2, dim=1)[1]
        else:
            raise Exception('not implement')
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(alphaS1), xs), betaS1).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(alphaS2), xt), betaS2).squeeze(1)
        qs_a = torch.add(torch.mul(torch.abs(alphaS3), xa), betaS3).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal_s = self.temporalRec1(tem_feat)
            temporal_t = self.temporalRec2(tem_feat)
            temporal_a = self.temporalRec3(tem_feat)

            # ax+b
            alphaT1 = torch.chunk(temporal_s, 2, dim=1)[0]
            alphaT1 = torch.add(alphaT1, ones)
            betaT1 = torch.chunk(temporal_s, 2, dim=1)[1]

            alphaT2 = torch.chunk(temporal_t, 2, dim=1)[0]
            alphaT2 = torch.add(alphaT2, ones)
            betaT2 = torch.chunk(temporal_t, 2, dim=1)[1]

            alphaT3 = torch.chunk(temporal_a, 2, dim=1)[0]
            alphaT3 = torch.add(alphaT3, ones)
            betaT3 = torch.chunk(temporal_a, 2, dim=1)[1]

        else:
            raise Exception('not implement')
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(alphaT1), xs), betaT1).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(alphaT2), xt), betaT2).squeeze(1)
        qt_a = torch.add(torch.mul(torch.abs(alphaT3), xa), betaT3).squeeze(1)

        if self.sr and self.tr:
            st_a1 = torch.sqrt(torch.abs(torch.mul(alphaS1, alphaT1)))
            st_b1 = torch.div(torch.add(betaS1, betaT1), 2)

            st_a2 = torch.sqrt(torch.abs(torch.mul(alphaS2, alphaT2)))
            st_b2 = torch.div(torch.add(betaS2, betaT2), 2)

            st_a3 = torch.sqrt(torch.abs(torch.mul(alphaS3, alphaT3)))
            st_b3 = torch.div(torch.add(betaS3, betaT3), 2)
            # modular_a_a = torch.sqrt(torch.abs(torch.mul(sa_t, ta_t)))

            qst_s = torch.add(torch.mul(st_a1, xs), st_b1).squeeze(1)
            qst_t = torch.add(torch.mul(st_a2, xt), st_b2).squeeze(1)
            qst_a = torch.add(torch.mul(st_a3, xa), st_b3).squeeze(1)


        # elif self.sr:
        # qst = qs
        # elif self.tr:
        # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)


        # 4 * batch_size
        t = torch.stack((xt.squeeze(1), qt_t, qs_t, qst_t))
        s = torch.stack((xs.squeeze(1), qt_s, qs_s, qst_s))
        a = torch.stack((xa.squeeze(1), qt_a, qs_a, qst_a))

        # return x, qt, qs, qst
        return t, s, a


class ViTbCLIP_SpatialTemporal_dropout_hybrid(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_dropout_hybrid, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        self.clip = ViT_B_16
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(5 * 256 * self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier((256) * self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

        self.sr = sr
        self.tr = tr

    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def forward(self, x, tem_feat, spa_feat, prmt):
        x_size = x.shape
        # x: (batch * frames) x 3-channel x height x width
        # eg. 128*3*224*224
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # Using clip.text
        # input shape (batch_size*frame_num)*c*h*w, which h and w must be 224
        image_features = self.clip.encode_image(x)
        # (batch_size*frame_num) * 51

        # Normalize image features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        input_texts = []
        for i in range(x_size[0]):
            prompt = [prmt[i]]
            texts = [f"a photo with {s} spatial quality and {t} temporal quality, which matches {p}"
                     for s, t, p in product(qualitys, qualitys, prompt)]
            input_texts.append(torch.cat([clip.tokenize(texts)]))

        input_texts = torch.cat(input_texts, dim=0)

        text_features = self.clip.encode_text(input_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        image_features = image_features.view(x_size[0], x_size[1], -1)
        mean_features = torch.mean(image_features, dim=1)
        text_features = text_features.view(x_size[0], -1, text_features.size(1))

        x_all = []
        x_all_presoftmax = []

        x_all_mean = []
        x_all_presoftmax_mean = []


        for i in range(x_size[0]):
            visual_feat = image_features[i, ...]
            text_feat = text_features[i, ...]
            cos_sim = logit_scale * visual_feat @ text_feat.t()
            cos_sim_pre = cos_sim
            cos_sim = torch.nn.functional.softmax(cos_sim, dim=1)
            x_all.append(cos_sim.unsqueeze(0))
            x_all_presoftmax.append(cos_sim_pre.unsqueeze(0))

            mean_feat = mean_features[i, ...]
            cos_sim_mean = logit_scale * mean_feat @ text_feat.t()
            cos_sim_pre_mean = cos_sim_mean
            cos_sim_mean = torch.nn.functional.softmax(cos_sim_mean,dim=0)
            x_all_mean.append(cos_sim_mean.unsqueeze(0))
            x_all_presoftmax_mean.append(cos_sim_pre_mean.unsqueeze(0))

        x_all = torch.cat(x_all, 0)
        x_all_presoftmax = torch.cat(x_all_presoftmax, 0)
        x_all = x_all.view(-1, x_all.size(2))
        x_all_presoftmax = x_all_presoftmax.view(-1, x_all_presoftmax.size(2))

        x_all_mean = torch.cat(x_all_mean, 0)
        x_all_presoftmax_mean = torch.cat(x_all_presoftmax_mean, 0)

        # frame-level
        logits_all = x_all.view(-1, len(qualitys), len(qualitys))

        xs = logits_all.sum(2)
        xt = logits_all.sum(1)
        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]

        xs = xs.view(x_size[0], -1)
        xt = xt.view(x_size[0], -1)
        xa = xa.view(x_size[0], -1)
        # batch*frame_num
        xs = torch.mean(xs, dim=1).unsqueeze(1)
        xt = torch.mean(xt, dim=1).unsqueeze(1)
        xa = torch.mean(xa, dim=1).unsqueeze(1)

        #video-level
        logits_all_mean = x_all_mean.view(-1, len(qualitys), len(qualitys))

        xsm = logits_all_mean.sum(2)
        xtm = logits_all_mean.sum(1)
        xam = x_all_presoftmax_mean.mean(1)

        xsm = 1 * xsm[:, 0] + 2 * xsm[:, 1] + 3 * xsm[:, 2] + 4 * xsm[:, 3] + 5 * xsm[:, 4]
        xtm = 1 * xtm[:, 0] + 2 * xtm[:, 1] + 3 * xtm[:, 2] + 4 * xtm[:, 3] + 5 * xtm[:, 4]

        xsm = xsm.view(x_size[0], -1)
        xtm = xtm.view(x_size[0], -1)
        xam = xam.view(x_size[0], -1)

        assert xa.shape == xs.shape == xt.shape, "shape is not same"
        assert xam.shape == xsm.shape == xtm.shape, "shape is not same"
        ones = torch.ones_like(xa)

        # spatial rectifier
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            spatial_t = self.spatialRec2(spa_feat)
            spatial_a = self.spatialRec3(spa_feat)

            # ax+b
            alphaS1 = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS1 = torch.add(alphaS1, ones)
            betaS1 = torch.chunk(spatial_s, 2, dim=1)[1]

            alphaS2 = torch.chunk(spatial_t, 2, dim=1)[0]
            alphaS2 = torch.add(alphaS2, ones)
            betaS2 = torch.chunk(spatial_t, 2, dim=1)[1]

            alphaS3 = torch.chunk(spatial_a, 2, dim=1)[0]
            alphaS3 = torch.add(alphaS3, ones)
            betaS3 = torch.chunk(spatial_a, 2, dim=1)[1]
        else:
            raise Exception('not implement')
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(alphaS1), xs), betaS1).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(alphaS2), xt), betaS2).squeeze(1)
        qs_a = torch.add(torch.mul(torch.abs(alphaS3), xa), betaS3).squeeze(1)


        qs_sm = torch.add(torch.mul(torch.abs(alphaS1), xsm), betaS1).squeeze(1)
        qs_tm = torch.add(torch.mul(torch.abs(alphaS2), xtm), betaS2).squeeze(1)
        qs_am = torch.add(torch.mul(torch.abs(alphaS3), xam), betaS3).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal_s = self.temporalRec1(tem_feat)
            temporal_t = self.temporalRec2(tem_feat)
            temporal_a = self.temporalRec3(tem_feat)

            # ax+b
            alphaT1 = torch.chunk(temporal_s, 2, dim=1)[0]
            alphaT1 = torch.add(alphaT1, ones)
            betaT1 = torch.chunk(temporal_s, 2, dim=1)[1]

            alphaT2 = torch.chunk(temporal_t, 2, dim=1)[0]
            alphaT2 = torch.add(alphaT2, ones)
            betaT2 = torch.chunk(temporal_t, 2, dim=1)[1]

            alphaT3 = torch.chunk(temporal_a, 2, dim=1)[0]
            alphaT3 = torch.add(alphaT3, ones)
            betaT3 = torch.chunk(temporal_a, 2, dim=1)[1]

        else:
            raise Exception('not implement')
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(alphaT1), xs), betaT1).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(alphaT2), xt), betaT2).squeeze(1)
        qt_a = torch.add(torch.mul(torch.abs(alphaT3), xa), betaT3).squeeze(1)

        qt_sm = torch.add(torch.mul(torch.abs(alphaT1), xsm), betaT1).squeeze(1)
        qt_tm = torch.add(torch.mul(torch.abs(alphaT2), xtm), betaT2).squeeze(1)
        qt_am = torch.add(torch.mul(torch.abs(alphaT3), xam), betaT3).squeeze(1)

        if self.sr and self.tr:
            st_a1 = torch.sqrt(torch.abs(torch.mul(alphaS1, alphaT1)))
            st_b1 = torch.div(torch.add(betaS1, betaT1), 2)

            st_a2 = torch.sqrt(torch.abs(torch.mul(alphaS2, alphaT2)))
            st_b2 = torch.div(torch.add(betaS2, betaT2), 2)

            st_a3 = torch.sqrt(torch.abs(torch.mul(alphaS3, alphaT3)))
            st_b3 = torch.div(torch.add(betaS3, betaT3), 2)
            # modular_a_a = torch.sqrt(torch.abs(torch.mul(sa_t, ta_t)))

            qst_s = torch.add(torch.mul(st_a1, xs), st_b1).squeeze(1)
            qst_t = torch.add(torch.mul(st_a2, xt), st_b2).squeeze(1)
            qst_a = torch.add(torch.mul(st_a3, xa), st_b3).squeeze(1)

            qst_sm = torch.add(torch.mul(st_a1, xsm), st_b1).squeeze(1)
            qst_tm = torch.add(torch.mul(st_a2, xtm), st_b2).squeeze(1)
            qst_am = torch.add(torch.mul(st_a3, xam), st_b3).squeeze(1)
        # elif self.sr:
        # qst = qs
        # elif self.tr:
        # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)

        # 4 * batch_size
        t = torch.stack((xt.squeeze(1), qt_t, qs_t, qst_t))
        s = torch.stack((xs.squeeze(1), qt_s, qs_s, qst_s))
        a = torch.stack((xa.squeeze(1), qt_a, qs_a, qst_a))

        tm = torch.stack((xtm.squeeze(1), qt_tm, qs_tm, qst_tm))
        sm = torch.stack((xsm.squeeze(1), qt_sm, qs_sm, qst_sm))
        am = torch.stack((xam.squeeze(1), qt_am, qs_am, qst_am))

        # return x, qt, qs, qst
        return t, s, a, tm, sm, am

        
class ViTbCLIP_SpatialTemporal_dropout_old(torch.nn.Module):

    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_dropout_old, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        # clip_vit_b_pretrained_features = ViT_B_16.visual
        # self.feature_extraction = clip_vit_b_pretrained_features
        
        self.clip=ViT_B_16
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048


        self.sr = sr
        self.tr = tr


    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block
    
    def forward(self, x, tem_feat, spa_feat, prmt):
        
        
        # self.spa_texts = torch.cat([clip.tokenize(
        #     f'a photo with {q} spatial quality, which matches {p}') for q,p in product(qualitys,prmt)])
        # self.tem_texts = torch.cat([clip.tokenize(
        #     f'a photo with {q} temporal quality, which matches {p}') for q,p in product(qualitys,prmt)])
        # ali_texts=torch.cat([clip.tokenize(
        #     f'a photo with {q} alignment quality, which matches {p}') for q,p in product(qualitys,prmt)])
        
        spa_texts=[]
        tem_texts=[]
        ali_texts=[]
        for i in range(len(prmt)):
            spa_texts.append(clip.tokenize([f'a photo with {q} spatial quality, which matches {prmt[i]}' for q in qualitys]))
            tem_texts.append(clip.tokenize([f'a photo with {q} temporal quality, which matches {prmt[i]}' for q in qualitys]))
            ali_texts.append(clip.tokenize([f'a photo with {q} alignment quality, which matches {prmt[i]}' for q in qualitys]))
        # bs*5*77
        spa_texts=torch.stack(spa_texts)
        spa_texts=spa_texts.view(-1,spa_texts.shape[2])
        tem_texts=torch.stack(tem_texts)
        tem_texts=tem_texts.view(-1,tem_texts.shape[2]) 
        ali_texts=torch.stack(ali_texts)
        ali_texts=ali_texts.view(-1,ali_texts.shape[2])

        
        x_size = x.shape
        # x: batch * frames x 3-channel x height x width
        # eg. 16*8*3*224*224
        x = x.view(-1, x_size[2], x_size[3], x_size[4])        
        image_features = self.clip.encode_image(x)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.view(x_size[0], x_size[1], -1)
        # batch * frames * 512

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        # Encode text features
        s_text_features = self.clip.encode_text(spa_texts.to(x.device))
        t_text_features = self.clip.encode_text(tem_texts.to(x.device))
        a_text_features = self.clip.encode_text(ali_texts.to(x.device))
        # print(a_text_features.shape)
        # Normalize text features
        s_text_features = s_text_features / s_text_features.norm(dim=1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=1, keepdim=True)
        a_text_features = a_text_features / a_text_features.norm(dim=1, keepdim=True)
        # b_s * 5 * 512
        s_text_features=s_text_features.view(x_size[0],-1,s_text_features.shape[1])
        t_text_features=t_text_features.view(x_size[0],-1,t_text_features.shape[1])
        a_text_features=a_text_features.view(x_size[0],-1,a_text_features.shape[1])

        # Compute cosine similarity as logits
        xs_all=[]
        xt_all=[]
        xa_all=[]
        for i in range(x_size[0]):
            visual_feat = image_features[i, ...]

            s_text_feat = s_text_features[i, ...]
            s_cos_sim = logit_scale * visual_feat @ s_text_feat.t()
            xs=torch.nn.functional.softmax(s_cos_sim,dim=1)
            # 8*5
            xs=xs.unsqueeze(0)
            xs_all.append(xs)
            
            t_text_feat = t_text_features[i,...]
            t_cos_sim = logit_scale * visual_feat @ t_text_feat.t()
            xt=torch.nn.functional.softmax(t_cos_sim,dim=1)
            xt=xt.unsqueeze(0)
            xt_all.append(xt)

            a_text_feat = a_text_features[i, ...]
            a_cos_sim = logit_scale * visual_feat @ a_text_feat.t()
            xa=a_cos_sim.unsqueeze(0)
            xa_all.append(xa)

        # Weighted sum of outputs
        # Assuming you want to apply specific weights to the classes
        xs=torch.cat(xs_all,0)
        xs=xs.view(-1,xs.size(2))
        # 64*5
        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xs=xs.view(x_size[0],-1)
        xs=torch.mean(xs,dim=1)
        
        xt=torch.cat(xt_all,0)
        xt=xt.view(-1,xt.size(2))
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]
        xt=xt.view(x_size[0],-1)
        xt=torch.mean(xt,dim=1)
        
        xa=torch.cat(xa_all,0)
        xa=xa.view(-1,xa.size(2))
        xa = 1 * xa[:, 0] + 2 * xa[:, 1] + 3 * xa[:, 2] + 4 * xa[:, 3] + 5 * xa[:, 4]
        xa=xa.view(x_size[0],-1)
        xa=xa.mean(dim=1)
                
        assert xa.shape == xs.shape == xt.shape, "shape is not same"
        xs = xs.unsqueeze(1)
        xt = xt.unsqueeze(1)
        xa = xa.unsqueeze(1)
        ones = torch.ones_like(xa)
        

        # spatial rectifier 
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            spatial_t = self.spatialRec2(spa_feat)
            spatial_a = self.spatialRec3(spa_feat)
            
            # ax+b
            alphaS1 = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS1 = torch.add(alphaS1, ones)
            betaS1 = torch.chunk(spatial_s, 2, dim=1)[1]

            alphaS2 = torch.chunk(spatial_t, 2, dim=1)[0]
            alphaS2 = torch.add(alphaS2, ones)
            betaS2 = torch.chunk(spatial_t, 2, dim=1)[1]

            alphaS3 = torch.chunk(spatial_a, 2, dim=1)[0]
            alphaS3 = torch.add(alphaS3, ones)
            betaS3 = torch.chunk(spatial_a, 2, dim=1)[1]           
        else:
            raise Exception('not implement')
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(alphaS1), xs), betaS1).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(alphaS2), xt), betaS2).squeeze(1)
        qs_a = torch.add(torch.mul(torch.abs(alphaS3), xa), betaS3).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier 
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal_s = self.temporalRec1(tem_feat)
            temporal_t = self.temporalRec2(tem_feat)
            temporal_a = self.temporalRec3(tem_feat)

            # ax+b
            alphaT1= torch.chunk(temporal_s, 2, dim=1)[0]
            alphaT1 = torch.add(alphaT1, ones)
            betaT1 = torch.chunk(temporal_s, 2, dim=1)[1]

            
            alphaT2= torch.chunk(temporal_t, 2, dim=1)[0]
            alphaT2 = torch.add(alphaT2, ones)
            betaT2 = torch.chunk(temporal_t, 2, dim=1)[1]

            alphaT3= torch.chunk(temporal_a, 2, dim=1)[0]
            alphaT3 = torch.add(alphaT3, ones)
            betaT3 = torch.chunk(temporal_a, 2, dim=1)[1]           

        else:
            raise Exception('not implement')
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(alphaT1), xs), betaT1).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(alphaT2), xt), betaT2).squeeze(1)
        qt_a = torch.add(torch.mul(torch.abs(alphaT3), xa), betaT3).squeeze(1)


        if self.sr and self.tr:
            st_a1 = torch.sqrt(torch.abs(torch.mul(alphaS1, alphaT1)))
            st_b1 = torch.div(torch.add(betaS1, betaT1), 2)

            st_a2 = torch.sqrt(torch.abs(torch.mul(alphaS2, alphaT2)))
            st_b2 = torch.div(torch.add(betaS2, betaT2), 2)

            st_a3 = torch.sqrt(torch.abs(torch.mul(alphaS3, alphaT3)))
            st_b3 = torch.div(torch.add(betaS3, betaT3), 2)
            # modular_a_a = torch.sqrt(torch.abs(torch.mul(sa_t, ta_t)))

            
            qst_s = torch.add(torch.mul(st_a1, xs), st_b1).squeeze(1)
            qst_t = torch.add(torch.mul(st_a2, xt), st_b2).squeeze(1)
            qst_a = torch.add(torch.mul(st_a3, xa), st_b3).squeeze(1)


        # elif self.sr:
            # qst = qs
        # elif self.tr:
            # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)

        # x=torch.concat((xt.unsqueeze(0),xs.unsqueeze(0),xa.unsqueeze(0)),dim=0)
        # qt=torch.concat((qt_t.unsqueeze(0),qt_s.unsqueeze(0),qt_a.unsqueeze(0)),dim=0)
        # qs=torch.concat((qs_t.unsqueeze(0),qs_s.unsqueeze(0),qs_a.unsqueeze(0)),dim=0)
        # qst=torch.concat((qst_t.unsqueeze(0),qst_s.unsqueeze(0),qst_a.unsqueeze(0)),dim=0)


        # 4 * batch_size 
        t=torch.stack((xt.squeeze(1),qt_t,qs_t,qst_t))
        s=torch.stack((xs.squeeze(1),qt_s,qs_s,qst_s))
        a=torch.stack((xa.squeeze(1),qt_a,qs_a,qst_a))
        
        # return x, qt, qs, qst
        return t, s, a


strings = [
    "How would you judge the quality of this video? ",
    "Can you rate the quality of this video? ",
    "Rate the quality of this video. ",
    "Could you evaluate the quality of this video? ",
    "What do you think about the quality of this video? ",
    "What is your quality rating for this video? ",
    "How would you rate the quality of this video? ",
    "How do you assess the quality of this video? ",
    "What's your opinion on the quality of this video? "
]
def wa5(logits):
    
    logprobs = np.array([logits["excellent"], logits["good"], logits["fair"], logits["poor"], logits["bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def expand2square(pil_img, background_color):
    # tensor
    # h*w*c
    pil_img = pil_img.numpy()
    if pil_img.max() <= 1.0:
        # If tensor is normalized (values between 0-1)
        pil_img = (pil_img * 255).astype(np.uint8)
    else:
        # If tensor already has values between 0-255
        pil_img = pil_img.astype(np.uint8)
    # Convert to PIL Image
    pil_img = Image.fromarray(pil_img)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class ViTbCLIP_exp(torch.nn.Module):
    def __init__(self, model_path, model_base, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_exp, self).__init__()
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(model_path, model_base, model_name)
        
        self.tokenizer = tokenizer
        self.model = model 
        self.image_processor = image_processor
        self.conv_mode="mplug_owl2"
        
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048


        self.sr = sr
        self.tr = tr


    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block
    
    def forward(self, x, tem_feat, spa_feat, prmt):
        
        x_size = x.shape        
        ret=torch.zeros(x_size[0]).to('cuda')
        for i in range(x_size[0]):
            image = [expand2square(img, tuple(int(t*255) for t in self.image_processor.image_mean)) for img in x[i]]
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to('cuda')
            conv = conv_templates[self.conv_mode].copy()
            inp = random.choice(strings) + prmt[i] + '\n' + DEFAULT_IMAGE_TOKEN
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " The quality of the video is"
            toks = ["good", "poor", "fair", "bad", "excellent"]
            ids_ = [id_[1] for id_ in self.tokenizer(toks)["input_ids"]]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
            llddata={}
            llddata["logits"] = defaultdict(float)
            with torch.inference_mode():
                output_logits = self.model(input_ids,
                    images=[image_tensor])["logits"][:,-1]
                for tok, id_ in zip(toks, ids_):
                    llddata["logits"][tok] += output_logits.mean(0)[id_].item()
                llddata["score"] = wa5(llddata["logits"])
            ret[i]=llddata["score"]
        xa=xs=xt=ret.unsqueeze(1)
        ones = torch.ones_like(xa)
        
        # spatial rectifier
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            spatial_t = self.spatialRec2(spa_feat)
            spatial_a = self.spatialRec3(spa_feat)

            # ax+b
            alphaS1 = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS1 = torch.add(alphaS1, ones)
            betaS1 = torch.chunk(spatial_s, 2, dim=1)[1]

            alphaS2 = torch.chunk(spatial_t, 2, dim=1)[0]
            alphaS2 = torch.add(alphaS2, ones)
            betaS2 = torch.chunk(spatial_t, 2, dim=1)[1]

            alphaS3 = torch.chunk(spatial_a, 2, dim=1)[0]
            alphaS3 = torch.add(alphaS3, ones)
            betaS3 = torch.chunk(spatial_a, 2, dim=1)[1]
        else:
            raise Exception('not implement')
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(alphaS1), xs), betaS1).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(alphaS2), xt), betaS2).squeeze(1)
        qs_a = torch.add(torch.mul(torch.abs(alphaS3), xa), betaS3).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal_s = self.temporalRec1(tem_feat)
            temporal_t = self.temporalRec2(tem_feat)
            temporal_a = self.temporalRec3(tem_feat)

            # ax+b
            alphaT1= torch.chunk(temporal_s, 2, dim=1)[0]
            alphaT1 = torch.add(alphaT1, ones)
            betaT1 = torch.chunk(temporal_s, 2, dim=1)[1]


            alphaT2= torch.chunk(temporal_t, 2, dim=1)[0]
            alphaT2 = torch.add(alphaT2, ones)
            betaT2 = torch.chunk(temporal_t, 2, dim=1)[1]

            alphaT3= torch.chunk(temporal_a, 2, dim=1)[0]
            alphaT3 = torch.add(alphaT3, ones)
            betaT3 = torch.chunk(temporal_a, 2, dim=1)[1]

        else:
            raise Exception('not implement')
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(alphaT1), xs), betaT1).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(alphaT2), xt), betaT2).squeeze(1)
        qt_a = torch.add(torch.mul(torch.abs(alphaT3), xa), betaT3).squeeze(1)


        if self.sr and self.tr:
            st_a1 = torch.sqrt(torch.abs(torch.mul(alphaS1, alphaT1)))
            st_b1 = torch.div(torch.add(betaS1, betaT1), 2)

            st_a2 = torch.sqrt(torch.abs(torch.mul(alphaS2, alphaT2)))
            st_b2 = torch.div(torch.add(betaS2, betaT2), 2)

            st_a3 = torch.sqrt(torch.abs(torch.mul(alphaS3, alphaT3)))
            st_b3 = torch.div(torch.add(betaS3, betaT3), 2)
            # modular_a_a = torch.sqrt(torch.abs(torch.mul(sa_t, ta_t)))


            qst_s = torch.add(torch.mul(st_a1, xs), st_b1).squeeze(1)
            qst_t = torch.add(torch.mul(st_a2, xt), st_b2).squeeze(1)
            qst_a = torch.add(torch.mul(st_a3, xa), st_b3).squeeze(1)
        # elif self.sr:
            # qst = qs
        # elif self.tr:
            # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)


        # 4 * batch_size
        t=torch.stack((xt.squeeze(1),qt_t,qs_t,qst_t))
        s=torch.stack((xs.squeeze(1),qt_s,qs_s,qst_s))
        a=torch.stack((xa.squeeze(1),qt_a,qs_a,qst_a))

        # if batch_size == 1, then return shape[4] directly
        if(x_size[0]==1):
            t,s,a = t.squeeze(1).to('cpu'),s.squeeze(1).to('cpu'),a.squeeze(1).to('cpu')

        return t, s, a
