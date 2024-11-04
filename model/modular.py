import torch
import torch.nn as nn
import torchvision.models as models
import clip

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

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.spatial_rec = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)
        self.temporal_rec = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

        self.sr = sr
        self.tr = tr

        # 5*77
        self.joint_texts = torch.cat(
            [clip.tokenize(f'a photo with {q} quality' for q in qualitys)])
        
        self.spa_texts = torch.cat([clip.tokenize(f'a photo with {q} spatial quality' for q in qualitys)])
        self.tem_texts = torch.cat([clip.tokenize(f'a photo with {q} tempotal quality' for q in qualitys)])

    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        '''
        linear -> relu ->linear
        512-1
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

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

    def forward(self, x, tem_feat, spa_feat):
        x_size = x.shape
        # x: (batch * frames) x 3-channel x height x width
        # eg. 128*3*224*224
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        
        # Using clip.visual
        # # input dimension: batch x frames x 3 x height x width
        # # ViT.visual accepts a 4-dim input format
        # x = self.feature_extraction(x)
        # # 2-dim output
        # # eg. 128*512

        # # convert to 2-dim
        # x = self.base_quality(x)
        # # eg. 128*1

        # x = x.view(x_size[0],-1)
        # # eg. 16*8

        # x = torch.mean(x, dim=1).unsqueeze(1)  
        # # eg. 16*1

        # Using clip.text
        # input shape (batch_size*frame_num)*c*h*w, which h and w must be 224
        image_features = self.clip.encode_image(x)
        # (batch_size*frame_num) * 512

        # Normalize image features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        # Encode text features
        s_text_features = self.clip.encode_text(self.spa_texts.to(x.device))
        t_text_features = self.clip.encode_text(self.tem_texts.to(x.device))
        # 5*512

        # Normalize text features
        s_text_features = s_text_features / s_text_features.norm(dim=1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity as logits
        # (b_s*frame_num)*5
        x_s = logit_scale * image_features @ s_text_features.t()
        x_t = logit_scale * image_features @ t_text_features.t()

        # Apply softmax to logits
        # (batch * frame_num) * 5
        xs = torch.nn.functional.softmax(x_s, dim=1)
        xt = torch.nn.functional.softmax(x_t, dim=1)

        # Weighted sum of outputs
        # Assuming you want to apply specific weights to the classes
        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]
        # 1-dim (batch*frame_num)

        xs = xs.view(x_size[0],-1)
        xt = xt.view(x_size[0],-1)
        # batch*frame_num
        xs = torch.mean(xs, dim=1).unsqueeze(1)  
        xt = torch.mean(xt, dim=1).unsqueeze(1)  
        # batch*1
        

        # spatial rectifier 
        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial = self.spatial_rec(spa_feat)
            s_ones = torch.ones_like(xs)  
            
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]
            sa = torch.add(sa, s_ones)
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(xs)
            sb = torch.zeros_like(xs)
        qs_s = torch.add(torch.mul(torch.abs(sa), xs), sb).squeeze(1)
        qs_t = torch.add(torch.mul(torch.abs(sa), xt), sb).squeeze(1)
        # shape batch*(batch*frame_num)

        # tempotal rectifier 
        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(tem_feat)
            t_ones = torch.ones_like(xs)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones)
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(xs)
            tb = torch.zeros_like(xs)
        qt_s = torch.add(torch.mul(torch.abs(ta), xs), tb).squeeze(1)
        qt_t = torch.add(torch.mul(torch.abs(ta), xt), tb).squeeze(1)

        if self.sr and self.tr:
            modular_a = torch.sqrt(torch.abs(torch.mul(sa, ta)))
            modular_b = torch.div(torch.add(sb, tb), 2)
            qst_s = torch.add(torch.mul(modular_a, xs), modular_b).squeeze(1)
            qst_t = torch.add(torch.mul(modular_a, xt), modular_b).squeeze(1)
        # elif self.sr:
            # qst = qs
        # elif self.tr:
            # qst = qt
        else:
            raise Exception('haven\'t implement yet')
            qst = x.squeeze(1)

        # x 2 qs 2 qt 2 qst 2
        # print(x.shape)
        # print(qs.shape)
        # print(qst.shape)
        x=torch.concat((xs.unsqueeze(0),xt.unsqueeze(0)),dim=0)
        qs=torch.concat((qs_s.unsqueeze(0),qs_t.unsqueeze(0)),dim=0)
        qt=torch.concat((qt_s.unsqueeze(0),qt_t.unsqueeze(0)),dim=0)
        qst=torch.concat((qst_s.unsqueeze(0),qst_t.unsqueeze(0)),dim=0)
        
        
        return x, qs, qt, qst
