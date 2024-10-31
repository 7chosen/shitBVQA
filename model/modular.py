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
        text_features = self.clip.encode_text(self.joint_texts.to(x.device))
        # 5*512

        # Normalize text features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity as logits
        # (b_s*frame_num)*5
        x = logit_scale * image_features @ text_features.t()

        # Apply softmax to logits
        # (batch * frame_num) * 5
        x = torch.nn.functional.softmax(x, dim=1)

        # Weighted sum of outputs
        # Assuming you want to apply specific weights to the classes
        x = 1 * x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + 4 * x[:, 3] + 5 * x[:, 4]
        # 1-dim (batch*frame_num)

        x = x.view(x_size[0],-1)
        # batch*frame_num
        x = torch.mean(x, dim=1).unsqueeze(1)  
        # batch*1
        

        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial = self.spatial_rec(spa_feat)
            s_ones = torch.ones_like(x)  
            
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]
            sa = torch.add(sa, s_ones)
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(x)
            sb = torch.zeros_like(x)
        qs = torch.add(torch.mul(torch.abs(sa), x), sb).squeeze(1)
        # shape batch*(batch*frame_num)

        if self.tr:
            x_3D_features_size = tem_feat.shape
            tem_feat = tem_feat.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(tem_feat)
            t_ones = torch.ones_like(x)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones)
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(x)
            tb = torch.zeros_like(x)
        qt = torch.add(torch.mul(torch.abs(ta), x), tb).squeeze(1)

        if self.sr and self.tr:
            modular_a = torch.sqrt(torch.abs(torch.mul(sa, ta)))
            modular_b = torch.div(torch.add(sb, tb), 2)
            qst = torch.add(torch.mul(modular_a, x), modular_b).squeeze(1)
        elif self.sr:
            qst = qs
        elif self.tr:
            qst = qt
        else:
            qst = x.squeeze(1)


        # print(x.shape)
        # print(qs.shape)
        # print(qst.shape)
        return x, qs, qt, qst
