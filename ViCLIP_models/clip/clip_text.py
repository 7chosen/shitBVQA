import os
import logging
from collections import OrderedDict
from pkg_resources import packaging
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as checkpoint
import functools


logger = logging.getLogger(__name__)


OPENCLIP_MODEL_PATH = 'https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
_MODELS = {
    "ViT-L/14": "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/vit_l14_text.pth",
    "CLIP-ViT-L/14": "https://huggingface.co/openai/clip-vit-large-patch14-336/vit_l14_text.pth",
    "CLIP-ViT-B/16": "https://huggingface.co/openai/clip-vit-base-patch16/vit_b16_text.pth",
}


# MODEL_PATH = 'https://huggingface.co/laion'
# _MODELS = {
#     "ViT-L/14": os.path.join(MODEL_PATH, "CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "vit_l14_text.pth"),
#     "ViT-B/16": os.path.join(MODEL_PATH, "CLIP-ViT-B-16-DataComp.XL-s13B-b90K", "vit_b16_text.pth"),
# }


from collections import OrderedDict
from typing import Union, Dict

def inspect_pth_file(pth_path: str, verbose: bool = True) -> Dict:
    """
    Inspect the shapes of all parameters in a .pth file
    
    Args:
        pth_path: Path to the .pth file
        verbose: If True, print shapes while analyzing
        
    Returns:
        Dictionary containing parameter shapes
    """
    # Load the state dict
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"File not found: {pth_path}")
    
    try:
        state_dict = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}

    # Handle different state dict formats
    if isinstance(state_dict, OrderedDict):
        params = state_dict
    elif hasattr(state_dict, 'state_dict'):
        params = state_dict.state_dict()
    else:
        try:
            params = state_dict['state_dict']
        except:
            params = state_dict

    # Store shapes in a dictionary
    shapes = {}
    total_params = 0
    
    # Get maximum key length for pretty printing
    max_key_length = max(len(key) for key in params.keys())
    
    # Analyze each parameter
    for key, tensor in params.items():
        shape = tuple(tensor.shape)
        shapes[key] = shape
        num_params = torch.prod(torch.tensor(shape)).item()
        total_params += num_params
        
        if verbose:
            print(f"{key:<{max_key_length}} | Shape: {str(shape):<20} | Parameters: {num_params:,}")
    
    if verbose:
        print("\nTotal Parameters:", f"{total_params:,}")
        print("File size:", f"{os.path.getsize(pth_path)/1024/1024:.2f} MB")
    
    return shapes

def inspect_nested_shapes(state_dict: Union[Dict, str], prefix: str = '') -> None:
    """
    Recursively inspect nested state dict shapes
    
    Args:
        state_dict: Either a state dict or path to .pth file
        prefix: Prefix for nested keys
    """
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    
    for key, value in state_dict.items():
        current_prefix = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            inspect_nested_shapes(value, current_prefix)
        elif isinstance(value, torch.Tensor):
            print(f"{current_prefix}: {tuple(value.shape)}")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 checkpoint_num: int = 0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.checkpoint_num = checkpoint_num

    def forward(self, x: torch.Tensor):
        if self.checkpoint_num > 0:
            segments = min(self.checkpoint_num, len(self.resblocks))
            return checkpoint.checkpoint_sequential(self.resblocks, segments, x,use_reentrant=False)
        else:
            return self.resblocks(x)


class CLIP_TEXT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            checkpoint_num: int,
        ):
        super().__init__()

        self.context_length = context_length
        self._tokenizer = _Tokenizer()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            checkpoint_num=checkpoint_num,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def no_weight_decay(self):
        return {'token_embedding', 'positional_embedding'}

    @functools.lru_cache(maxsize=None)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def tokenize(self, texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, text, return_embed=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_embed:
            return x @ self.text_projection, x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

def clip_text_b16(
    embed_dim=512,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
):

    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num
    )
    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            # pretrained = _MODELS["ViT-B/16"]
            pretrained = "./ckpts/ViCLIP-B_InternVid-FLT-10M.pth"
        logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location='cpu',weights_only=False)
        mid=state_dict['model']
        prefix = 'text_encoder.'
        mid2 = {k[len(prefix):]: v for k, v in mid.items() if k.startswith(prefix)}
        state_dict=mid2        
        # print(state_dict.keys())
        # inspect_nested_shapes(model.state_dict())
        # inspect_nested_shapes(state_dict)

        if context_length != state_dict["positional_embedding"].size(0):
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            print(f"Resize positional embedding from {state_dict['positional_embedding'].size(0)} to {context_length}")
            if context_length < state_dict["positional_embedding"].size(0):
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = F.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                    value=0,
                )

        message = model.load_state_dict(state_dict, strict=False)
        # print(f"Load pretrained weights from {pretrained}: {message}")
    return model.eval()


def clip_text_l14(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
    )
    if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
        pretrained = _MODELS[pretrained]
    else:
        pretrained = _MODELS["CLIP-ViT-L/14"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    if context_length != state_dict["positional_embedding"].size(0):
        # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
        print(f"Resize positional embedding from {state_dict['positional_embedding'].size(0)} to {context_length}")
        if context_length < state_dict["positional_embedding"].size(0):
            state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
        else:
            state_dict["positional_embedding"] = F.pad(
                state_dict["positional_embedding"],
                (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                value=0,
            )

    message = model.load_state_dict(state_dict, strict=False)
    print(f"Load pretrained weights from {pretrained}: {message}")
    return model.eval()


def clip_text_l14_336(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
):
    raise NotImplementedError
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )
    pretrained = _MODELS["ViT-L/14_336"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def build_clip(config):
    model_cls = config.text_encoder.clip_teacher
    model = eval(model_cls)()
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 4
    
    config = {
        'text_encoder':
            {
            'clip_teacher': 'clip_text_b16',
        }
    }
    from easydict import EasyDict
    model = build_clip(EasyDict(config))
    
    output = model('This is a dog')
    print(output.shape)