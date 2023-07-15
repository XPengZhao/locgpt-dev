# -*- coding: utf-8 -*-
"""Implementation of Models
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

from loc_comdel import area


class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # input dimension of gamma
        out_dim = 0

        # why include input?
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']  # L

        # why not log sampling?
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_ch, enable=True):
    """get embedding function and output_dims

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 0 for default positional encoding, -1 for none

    Returns
    -------
        embedding function; output_dims
    """
    if enable == False:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': input_ch,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Return
        ----------
        x: tensor, [B, n_seq, feature_dim]
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        def pair(t):
            return t if isinstance(t, tuple) else (t, t)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)
        feature = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)

        return self.mlp_head(x), feature                                        #  (b, num_classes)



class TNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=3, skips=[4]):
        """input + 8 hidden layer + output

        Parameters
        ----------
        D : hidden layer number
        W : Dimension per hidden layer
        input_ch : int, input channel, by default 256
        output_ch : int, out channel, by default 3
        skip : list, residual layer, optional
        """
        super().__init__()
        self.D, self.W = D, W
        self.input_ch = input_ch
        self.output_ch = output_ch

        self.skips = skips
        self.num_layers = D + 2    # hidden + input + output

        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )
        # self.pos_bn = nn.ModuleList([nn.BatchNorm1d(W) for i in range(D)])
        self.output_layer = nn.Linear(W, output_ch)

        self.activation = nn.Softplus(beta=100)



    def forward(self, inputs):
        """forward function of the model

        Parameters
        ----------
        x : [batchsize ,input]

        Returns
        ----------
        outputs: [batchsize, 3].   position
        """
        inputs = inputs.to(torch.float32)
        h = inputs

        for i, layer in enumerate(self.linears):
            h = layer(h)
            # h = F.relu(h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        pos = self.output_layer(h)    # (batch_size, 3)
        return pos



def vit_worker():
    model_vit = ViT(
            image_size = (16, 20),
            patch_size = (1, 20),
            channels = 1,
            num_classes = 2,  # alpha, beta
            dim = 20,
            depth = 2,
            heads = 8,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    return model_vit

model_ch1 = vit_worker()
model_ch2 = vit_worker()
model_ch3 = vit_worker()


class LocGPT(nn.Module):
    def __init__(self):
        super(LocGPT, self).__init__()
        self.vit1 = model_ch1
        self.vit2 = model_ch2
        self.vit3 = model_ch3

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mlp = TNetwork(input_ch=60)


    def forward(self, x, y):
        B, C, H, W = x.shape

        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x3 = x[:, 2, :, :].unsqueeze(1)

        omega1, encoder_output1 = self.vit1(x1)
        omega2, encoder_output2 = self.vit2(x2)
        omega3, encoder_output3 = self.vit3(x3)

        l1 = encoder_output1[:, 0]
        l2 = encoder_output2[:, 0]
        l3 = encoder_output3[:, 0]

        s = area(omega1, omega2, omega3)

        f = torch.cat((l1, l2, l3), dim=1)
        p = self.mlp(f)
        return torch.cat((s, p), dim=1)





if __name__ == '__main__':

    model_vit = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    img = torch.randn(16, 3, 256, 256)

    preds = model_vit(img)

    print(preds.shape)  # (16, 1000)

