import logging
import os

import clip
import torch
from torch import nn

from .backbones.clip.clip_vision import clip_joint_l14, clip_joint_b16
from .backbones.clip.clip_text import clip_text_l14, clip_text_b16
from .criterions import VTC_VTM_Loss

from itertools import product
logger = logging.getLogger(__name__)


qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
class ViCLIP(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self, tokenizer=None, is_pretrain=True,freeze_text=False):
        super(ViCLIP, self).__init__()

        # self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.text_width = 768
        self.embed_dim = 768
        self.masking_prob = 0.9
        
        self.vision_encoder_name = 'vit_b16'
        self.vision_encoder_pretrained = True
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24
        self.vision_width = 1024
        
        
        self.text_encoder_name = 'vit_b16'
        self.text_encoder_pretrained = True#'bert-base-uncased'
        self.text_encoder_d_model = 512
        self.text_encoder_vocab_size = 49408
        self.max_txt_l = 32

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.temp = nn.parameter.Parameter(torch.ones([]) * 1/100.0,requires_grad=False)
        self.temp_min = 1/100.0

        # criterions
        self.clip_loss = VTC_VTM_Loss(False)

        # Freeze weights
        if freeze_text:
            self.freeze_text()

        self.tem_texts = [f'a photo with {q} temporal quality' for q in qualitys]
        self.spa_texts = [f'a photo with {q} spatial quality' for q in qualitys]
        
        # if is_pretrain:
        #     pt='/home/user/Desktop/1/shitBVQA/ckpts_modular/ViCLIP-B_InternVid-FLT-10M.pth'
        #     state_dict = torch.load(pt, map_location='cpu')['model']
        #     self.load_state_dict(state_dict)

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )

        return ret

    def forward(self, image,raw_text, idx, log_generation=None, return_sims=False):
        """Forward pass to calculate the contrastive loss between image and text representations.
        Args:
            image (torch.Tensor): The input images with shape [B,T,C,H,W], where B is batch size, T is the number of tokens, C is the number of channels, H is the height, and W is the width of the images.
            text (dict): A dictionary containing preprocessed text data. This should include tokenized text and potentially other information required for encoding text. The exact structure depends on the text preprocessing and encoding method used.
            raw_text (str or list of str): The raw text input(s) corresponding to the images. This can be either a single string or a list of strings, where each string is the raw text description of the corresponding image in the batch.
            idx (torch.Tensor): A tensor of indices representing the correspondence between images and texts within the batch. It is used to compute the contrastive loss by matching the correct pairs of image and text representations.
            log_generation (bool, optional): If specified, enables logging of additional information during the forward pass. Defaults to None.
            return_sims (bool, optional): If True, the function returns the similarity scores between image and text embeddings instead of the loss. Defaults to False.
        Returns:
            If return_sims is False, returns a dictionary containing the following key-value pairs:
                'loss_vtc' (torch.Tensor): The vision-text contrastive loss computed between the image and text embeddings.
            If return_sims is True, returns a torch.Tensor containing the similarity scores between the normalized vision and text embeddings.
        """
        # self.clip_contrastive_temperature()
        
        input_texts=[]
        for i in range(image.shape[0]):
            prompt=[raw_text[i]]
            texts=[f"a photo with {s} spatial quality and {t} temporal quality, which match {p}"
                   for s,t,p in product(qualitys,qualitys,prompt)]
            texts=self.encode_text(texts)
            texts=texts/texts.norm(dim=1,keepdim=True)
            input_texts.append(texts)
        
        # batch * 25 * 512
        text_embeds = torch.stack(input_texts)
        # batch * 512
        vision_embeds = self.encode_vision(image)
        
        x_all=[]
        x_all_presoftmax=[]
        for i in range(image.shape[0]):
            visual_feat=vision_embeds[i,...]
            text_feat=text_embeds[i,...]
            sim = (1/self.temp) * visual_feat @ text_feat.t()
            sim_pre=sim
            
            sim=nn.functional.softmax(sim)
            
            x_all.append(sim.unsqueeze(0))
            x_all_presoftmax.append(sim_pre.unsqueeze(0))
        
        # batch * 25
        x_all=torch.cat(x_all,0)
        # batch * 25 
        x_all_presoftmax=torch.cat(x_all_presoftmax,0)
        
        logits_all=x_all.view(-1,len(qualitys),len(qualitys))
        xs = logits_all.sum(2)
        xt = logits_all.sum(1)
        xa = x_all_presoftmax.mean(1)

        xs = 1 * xs[:, 0] + 2 * xs[:, 1] + 3 * xs[:, 2] + 4 * xs[:, 3] + 5 * xs[:, 4]
        xt = 1 * xt[:, 0] + 2 * xt[:, 1] + 3 * xt[:, 2] + 4 * xt[:, 3] + 5 * xt[:, 4]        
        

        if return_sims:
            # sims = torch.nn.functional.normalize(vision_embeds, dim=-1) @ \
                #   torch.nn.functional.normalize(text_embeds, dim=-1).transpose(0, 1)
            return [xt,xs,xa]

        # calculate loss

        ## VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if test and self.config.model.vision_encoder.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.config.model.vision_encoder.masking_prob
            )

        return self.vision_encoder(image)

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        device = next(self.text_encoder.parameters()).device
        text = self.text_encoder.tokenize(
            text, context_length=self.max_txt_l
        ).to(device)
        text_embeds = self.text_encoder(text)
        return text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        logger.info(f'vision encoder name: {encoder_name}')
        # if encoder_name != "vit_l14":
        #     raise ValueError(f"Not implemented: {encoder_name}")

        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14(
                pretrained=self.config.model.vision_encoder.pretrained,
                input_resolution=self.config.inputs.image_res,
                kernel_size=self.config.model.vision_encoder.kernel_size,
                center=self.config.model.vision_encoder.center,
                num_frames=self.config.inputs.video_input.num_frames,
                drop_path=self.config.model.vision_encoder.drop_path_rate,
                checkpoint_num=self.config.model.vision_encoder.checkpoint_num,
            )
        elif encoder_name == "vit_b16":
                vision_encoder = clip_joint_b16(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        logger.info(f'text encoder name: {encoder_name}')
        # if encoder_name != "vit_l14":
        #     raise ValueError(f"Not implemented: {encoder_name}")
        if encoder_name == "vit_l14":
            text_encoder = clip_text_l14(
                pretrained=self.config.model.text_encoder.get("pretrained", True),
                embed_dim=self.config.model.text_encoder.d_model,
                context_length=self.config.max_txt_l,
                vocab_size=self.config.model.text_encoder.vocab_size,
                checkpoint_num=self.config.model.text_encoder.get("checkpoint_num", 0),
            )
        elif encoder_name == "vit_b16":
            text_encoder = clip_text_b16(
                pretrained=self.text_encoder_pretrained,
                embed_dim=self.text_encoder_d_model,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
