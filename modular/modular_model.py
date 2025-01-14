from q_align.mm_utils import  tokenizer_image_token, get_model_name_from_path
from q_align.model.builder import load_pretrained_model
from q_align.conversation import conv_templates
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import PIL.Image as Image
from itertools import product
import clip
# import torchvision.models as models
import torch.nn as nn
import torch
import random
import numpy as np
from collections import defaultdict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# qualitys = ['bad', 'fair', 'good']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']


class ViTbCLIP(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP, self).__init__()
        # self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        ViT_B_16, _ = clip.load("ViT-B/16")
        self.clip = ViT_B_16
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

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

    def forward(self, x, tem_feat, spa_feat, prmt, num):
        x_size = x.shape
        # x: (batch * frames) x 3-channel x height x width
        # eg. 128*3*224*224
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # input shape (batch_size*frame_num)*c*h*w, which h and w must be 224
        image_features = self.clip.encode_image(x)

        # print(image_features[0][0])
        # (batch_size*frame_num) * 512

        # Normalize image features
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        # bs * frames * 512
        image_features = image_features.view(x_size[0], x_size[1], -1)

        # Get logit scale
        logit_scale = self.clip.logit_scale.exp()

        input_texts = []
        for i in range(x_size[0]):
            prompt = [prmt[i]]
            texts = [f"a photo with {s} spatial quality, {t} temporal quality, which matches {p}"
                     for s, t, p in product(qualitys, qualitys, prompt)]
            input_texts.append(
                torch.cat([clip.tokenize(texts, context_length=77, truncate=True)]))

        input_texts = torch.cat(input_texts, dim=0)
        # print(input_texts.shape)

        text_features = self.clip.encode_text(input_texts.to(x.device))
        # print(text_features[0][0])
        # 200 * 512
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # bs * 15 * 512
        text_features = text_features.view(
            x_size[0], -1, text_features.size(1))
        # print(text_features[0][0][0])
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
        # bs * 8 * 125
        # 128*125

        x_all_presoftmax = x_all_presoftmax.view(-1, x_all_presoftmax.size(2))

        logits_all = x_all.view(-1, len(qualitys), len(qualitys))

        xs = logits_all.sum(dim=2)
        xt = logits_all.sum(dim=1)

        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * \
            xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * \
            xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]

        xs = xs.view(x_size[0], -1)
        xt = xt.view(x_size[0], -1)
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

        # if batch_size == 1, then return shape[4] directly
        if (x_size[0] == 1):
            t, s, a = t.squeeze(1).to('cpu'), s.squeeze(
                1).to('cpu'), a.squeeze(1).to('cpu')

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

    logprobs = np.array([logits["excellent"], logits["good"],
                        logits["fair"], logits["poor"], logits["bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0.]))


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
    pil_img = pil_img.cpu().numpy()
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


class ViTbQalign(torch.nn.Module):
    def __init__(self, model_path, model_base, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbQalign, self).__init__()
        # disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(model_path, model_base, model_name)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = "mplug_owl2"

        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.spatialRec1 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)
        self.spatialRec2 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)
        self.spatialRec3 = self.spatial_rectifier(
            5*256*self.feat_len, self.dropout_sp)

        self.temporalRec1 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec2 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048
        self.temporalRec3 = self.temporal_rectifier(
            (256)*self.feat_len, self.dropout_tp)  # Fast:256  Slow:2048

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

    def forward(self, x, tem_feat, spa_feat, prmt, num, phase="train"):

        x_size = x.shape
        ret = torch.zeros(x_size[0]).to('cuda')
        x_ast = torch.zeros(x_size[0], 3).to('cuda')
        toks = ["good", "poor", "fair", "bad", "excellent"]
        ids_ = [id_[1] for id_ in self.tokenizer(toks)["input_ids"]]

        # if phase == 'train':
        #     for i in range(x_size[0]):
        #         llddata={}
        #         llddata["logits"] = defaultdict(float)

        #         image = [expand2square(img, tuple(int(t*255) for t in self.image_processor.image_mean)) for img in x[i]]
        #         image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to('cuda')

        #         conv = conv_templates[self.conv_mode].copy()
        #         inp = random.choice(strings) + prmt[i] + '\n' + DEFAULT_IMAGE_TOKEN
        #         conv.append_message(conv.roles[0], inp)
        #         conv.append_message(conv.roles[1], None)
        #         if num == 3:
        #             mode=random.choice(["spatial","temporal","alignment"])
        #             prompt = conv.get_prompt() + f" The {mode} quality of the video is"
        #         elif num==1:
        #             prompt = conv.get_prompt() + " The quality of the video is"
        #         input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
        #         with torch.inference_mode():
        #             output_logits = self.model(input_ids,
        #                 images=[image_tensor])["logits"][:,-1]
        #             for tok, id_ in zip(toks, ids_):
        #                 llddata["logits"][tok] += output_logits.mean(0)[id_].item()
        #             llddata["score"] = wa5(llddata["logits"])
        #         ret[i]=llddata["score"]
        #     xa=xs=xt=ret.unsqueeze(1)
        # elif phase == 'eval':
        for i in range(x_size[0]):
            llddata = {}
            llddata["logits"] = defaultdict(float)

            image = [expand2square(img, tuple(
                int(t*255) for t in self.image_processor.image_mean)) for img in x[i]]
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'].half().to('cuda')

            conv = conv_templates[self.conv_mode].copy()
            inp = random.choice(strings) + prmt[i] + '\n' + DEFAULT_IMAGE_TOKEN
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)

            if num == 3:
                prompt_all = [conv.get_prompt(
                ) + f" The {con} quality of the video is" for con in ["spatial", "temporal", "alignment"]]
                for idx, prompt in enumerate(prompt_all):
                    input_ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        output_logits = self.model(input_ids,
                                                   images=[image_tensor])["logits"][:, -1]
                        for tok, id_ in zip(toks, ids_):
                            llddata["logits"][tok] += output_logits.mean(0)[
                                id_].item()
                        llddata["score"] = wa5(llddata["logits"])
                x_ast[i][idx] = llddata["score"]

            if num == 1:
                prompt = conv.get_prompt() + " The quality of the video is"
                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
                with torch.inference_mode():
                    output_logits = self.model(input_ids,
                                               images=[image_tensor])["logits"][:, -1]
                    for tok, id_ in zip(toks, ids_):
                        llddata["logits"][tok] += output_logits.mean(0)[
                            id_].item()
                    llddata["score"] = wa5(llddata["logits"])
                ret[i] = llddata["score"]
        if num == 3:
            xt = x_ast[:, 0].unsqueeze(1)
            xs = x_ast[:, 1].unsqueeze(1)
            xa = x_ast[:, 2].unsqueeze(1)
        else:
            xa = xs = xt = ret.unsqueeze(1)

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

        # if batch_size == 1, then return shape[4] directly
        if (x_size[0] == 1):
            t, s, a = t.squeeze(1).to('cpu'), s.squeeze(
                1).to('cpu'), a.squeeze(1).to('cpu')

        return t, s, a
