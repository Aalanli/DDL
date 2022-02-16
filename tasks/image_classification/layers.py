# %%
import math
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

#from continousSum.continous_sum import ContinousPool1D


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class AlibiMask(nn.Module):
    def __init__(self, start_ratio=1/2, heads=8, apply=True):
        super().__init__()
        self.use = apply
        self.heads = heads
        self.start_ratio = start_ratio
        self.mask = None
    
    def alibi_mask(self, n, heads):
        a = torch.arange(0.0, n, 1.0).unsqueeze(0).repeat(n, 1)
        b = torch.arange(0.0, n, 1.0).unsqueeze(1)
        a = a - b
        a = a.unsqueeze(0).repeat(heads, 1, 1)
        c = torch.tensor([[[self.start_ratio / 2 ** i]] for i in range(heads)])
        return c * a
    
    def forward(self, x):
        if self.use is False:
            return None
        b, seq_len, dim = x.shape
        if self.mask is None or self.mask.shape[-2] < seq_len:
            self.mask = self.alibi_mask(seq_len, self.heads).to(x)
        if self.mask.device != x.device or self.mask.dtype != x.dtype:
            self.mask = self.mask.to(x)
        return self.mask[:, :seq_len, :seq_len]


class Attention(nn.Module):
    def __init__(self, d_model, heads, bias=False, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.b = bias

        self.qw = nn.Linear(d_model, d_model, bias=bias)
        self.kw = nn.Linear(d_model, d_model, bias=bias)
        self.vw = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor):
        # shape = [batch, sequence, features]
        # split features into heads; size = [batch, heads, sequence, depth]
        batch, seq_len, features = x.shape
        x = x.reshape((batch, seq_len, self.heads, features // self.heads))
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor):
        # inverse operation of split heads
        batch, heads, seq_len, depth = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape((batch, seq_len, heads * depth))
    
    def unsqueeze_mask(self, mask: torch.Tensor):
        if mask.dim() == 2:
            return mask.unsqueeze(1).unsqueeze(2)
        raise NotImplementedError(f"mask dim {mask.dim()} not supported")

    def multihead_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, alibi_mask=None):
        """
        Most naive implementation
        mask.shape = [bs, k.shape[-2]]; k.shape[-2] = k_seq_len
        """
        depth = q.shape[-1]
        w = q @ k.transpose(-1, -2)
        w = w / math.sqrt(depth)

        if mask is not None:
            w = w.masked_fill(mask, float('-inf'))
        
        if alibi_mask is not None:
            w = w + alibi_mask
        
        a = w.softmax(-1)
        a = self.dropout(a)
        out = a @ v
        return out, a
        
    def attn_proj(self, q, k, v):
        """Transforms the inputs"""
        q = self.qw(q)
        k = self.kw(k)
        v = self.vw(v)
        return q, k, v
    
    def forward(self, q, k, v, mask=None, alibi_mask=None):
        q, k, v = self.attn_proj(q, k, v)
        q, k, v = map(self.split_heads, (q, k, v))
        out, a = self.multihead_attn(q, k, v, mask, alibi_mask)
        return self.combine_heads(out), a
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, proj_forward, activation=F.relu, dropout=0.1, attn_dropout=0.1, bias=None):
        super().__init__()
        self.attn = Attention(d_model, heads, bias, dropout=attn_dropout)        

        self.linear1 = nn.Linear(d_model, proj_forward)
        self.linear2 = nn.Linear(proj_forward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, mask=None, alibi_mask=None, pos=None):
        if pos is not None:
            q = k = x + pos
        else:
            q = k = x
        x1 = self.attn(q, k, x, mask, alibi_mask=alibi_mask)[0]
        x = self.drop(x1) + x  # save, non-deterministic
        x = self.norm1(x)
        x1 = self.activation(self.linear1(x))
        x1 = self.drop(x1)  # save, non-deterministic
        x1 = self.linear2(x1)
        x = x + x1
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, layers, hidden_dim, proj_dim, heads, dropout, bias, alibi, alibi_start):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, heads, proj_dim, dropout=dropout, attn_dropout=dropout, bias=bias) for _ in range(layers)])
        self.alibi_mask = AlibiMask(apply=alibi, start_ratio=alibi_start)
    
    def forward(self, x, mask=None, pos=None):
        alibi = self.alibi_mask(x)
        for layer in self.layers:
            x = layer(x, mask, alibi, pos)
        return x


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input, mask=None):
        xs = self.body(input)
        if mask is None:
            return xs
        out = {}
        for name, x in xs.items():
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = (x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    return tensor, mask


class Model(nn.Module):
    def __init__(self, resnet_type, transformer_layers, heads, num_classes, hidden_dim, proj_dim, bias, alibi, alibi_start):
        super().__init__()
        #self.transformer = Transformer(transformer_layers, hidden_dim, proj_dim, heads, 0.1, bias, alibi, alibi_start)
        self.backbone = Backbone(resnet_type, True, False, False)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.in_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, 3, 1, 1)
        #self.pos_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    def forward(self, samples):
        if type(samples) == list:
            samples, mask = nested_tensor_from_tensor_list(samples)
            features, mask = self.backbone(samples, mask)['0']
            #pos = self.pos_emb(mask)
            #mask = mask.flatten(1)[:, None, None, :]
        else:
            features = self.backbone(samples)['0']
            #mask = None
            #pos = self.pos_emb(torch.zeros((features.shape[0], features.shape[2], features.shape[3]), 
            #                   device=samples.device, dtype=torch.bool))

        features = self.in_proj(features)
        hs = features.flatten(2).transpose(-1, -2)
        #pos = pos.flatten(1, 2)
        #hs = self.transformer(hs, mask, None)
        outputs_class = self.class_embed(hs).sum(1)
        return outputs_class

