import torch
import torch.nn as nn
import torchvision.models as models
import clip
from itertools import product

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

        # self.spa_texts = torch.cat([clip.tokenize(f'a photo with {q} spatial quality' for q in qualitys)])
        # self.tem_texts = torch.cat([clip.tokenize(f'a photo with {q} temporal quality' for q in qualitys)])

        # self.joint_texts = torch.cat(
        #     [f"a photo with {s} spatial quality and {t} temporal quality" for s, t
        #      in product(qualitys, qualitys)])

        # self.joint_texts = [f"a photo that {c} matches '{p}'" for p, c in product(prompt, qualitys_p)]

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

        text_features =  self.clip.encode_text(input_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        image_features = image_features.view(x_size[0], x_size[1], -1)
        text_features = text_features.view(x_size[0],-1, text_features.size(1))

        x_all = []
        x_all_presoftmax = []
        for i in range(x_size[0]):
            visual_feat = image_features[i, ...]
            text_feat = text_features[i, ...]
            cos_sim = logit_scale * visual_feat @ text_feat.t()
            cos_sim_pre = cos_sim
            cos_sim = torch.nn.functional.softmax(cos_sim, dim=1)
            x_all.append(cos_sim.unsqueeze(0))
            x_all_presoftmax.append(cos_sim_pre.unsqueeze(0))

        x_all = torch.cat(x_all, 0)
        x_all_presoftmax = torch.cat(x_all_presoftmax, 0)
        x_all = x_all.view(-1, x_all.size(2))
        x_all_presoftmax = x_all_presoftmax.view(-1, x_all_presoftmax.size(2))

        # x_all = logit_scale * image_features @ text_features.t()
        # x_all_presoftmax = x_all
        # x_all = torch.nn.functional.softmax(x_all, dim=1)

        logits_all = x_all.view(-1, len(qualitys), len(qualitys))

        xs = logits_all.sum(2)
        xt = logits_all.sum(1)
        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]

        xs = xs.view(x_size[0],-1)
        xt = xt.view(x_size[0],-1)
        xa = xa.view(x_size[0], -1)
        # batch*frame_num
        xs = torch.mean(xs, dim=1).unsqueeze(1)
        xt = torch.mean(xt, dim=1).unsqueeze(1)
        xa = torch.mean(xa, dim=1).unsqueeze(1)

        # # Encode text features
        # s_text_features = self.clip.encode_text(self.spa_texts.to(x.device))
        # t_text_features = self.clip.encode_text(self.tem_texts.to(x.device))
        # # 5*512
        #
        # # Normalize text features
        # s_text_features = s_text_features / s_text_features.norm(dim=1, keepdim=True)
        # t_text_features = t_text_features / t_text_features.norm(dim=1, keepdim=True)
        #
        # # Compute cosine similarity as logits
        # # (b_s*frame_num)*5
        # x_s = logit_scale * image_features @ s_text_features.t()
        # x_t = logit_scale * image_features @ t_text_features.t()
        # # print(x_t.shape)
        #
        # # Apply softmax to logits
        # # (batch * frame_num) * 5
        # xs = torch.nn.functional.softmax(x_s, dim=1)
        # xt = torch.nn.functional.softmax(x_t, dim=1)
        #
        # # Weighted sum of outputs
        # # Assuming you want to apply specific weights to the classes
        # xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        # xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]
        # # 1-dim (batch*frame_num)

        # xs = xs.view(x_size[0],-1)
        # xt = xt.view(x_size[0],-1)
        # # batch*frame_num
        # xs = torch.mean(xs, dim=1).unsqueeze(1)
        # xt = torch.mean(xt, dim=1).unsqueeze(1)
        # # batch*1
        #
        #
        # alignment = clip.tokenize(prmt).to(x.device)
        # alignment = self.clip.encode_text(alignment)
        # ali_feat = alignment / alignment.norm(dim=1,keepdim=True)
        # ali_feat =logit_scale * image_features @ ali_feat.t()
        # ali_feat = ali_feat.view(x_size[0],-1)
        # # batch * 128
        # xa = torch.mean(ali_feat, dim=1).unsqueeze(1)
        # # batch*1
        
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


class ViTbCLIP_SpatialTemporal_dropout_meanpool(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_dropout_meanpool, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        ckpt = torch.load('pickscore_vitb16.pt')
        ViT_B_16.load_state_dict(ckpt['model_state_dict'])

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

        # self.spa_texts = torch.cat([clip.tokenize(f'a photo with {q} spatial quality' for q in qualitys)])
        # self.tem_texts = torch.cat([clip.tokenize(f'a photo with {q} temporal quality' for q in qualitys)])

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
        # print(text_features.shape)

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
            cos_sim = torch.nn.functional.softmax(cos_sim)
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

        # xs = xs.view(x_size[0],-1)
        # xt = xt.view(x_size[0],-1)
        # xa = xa.view(x_size[0], -1)
        # # batch*frame_num
        # xs = torch.mean(xs, dim=1).unsqueeze(1)
        # xt = torch.mean(xt, dim=1).unsqueeze(1)
        # xa = torch.mean(xa, dim=1).unsqueeze(1)


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