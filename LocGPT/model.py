# -*- coding: utf-8 -*-
"""Implementation of Models
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
import numpy as np
from loc_comdel import area



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

        self.encoder = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

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

        x = self.encoder(x)                                                 # (b, 65, dim)
        feature = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)

        return self.mlp_head(x), feature                                        #  (b, num_classes)



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


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=.1, max_len=1024):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=p_drop)

    positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
    position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

    positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
    positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

    # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
    positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

    # register pe to buffer and require no grads
    self.register_buffer('pe', positional_encoding)

  def forward(self, x):
    # x: [seq_len, batch, d_model]
    # we can add positional encoding to x directly, and ignore other dimension
    x = x + self.pe[:x.size(0), ...]

    return self.dropout(x)



def get_padding_mask(seq_q, seq_k):
    '''
    Create a mask for keys that correspond to padding tokens.

    parameters:
    -------------
    seq_q: [batch, seq_len]
    seq_k: [batch, seq_len]

    return:
    -------------
    mask: [batch, len_q, len_k]
    '''
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # we define index of PAD is 0, if tensor equals (zero) PAD tokens
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch, 1, len_k]

    return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]


def get_lookahead_mask(seq):
    '''
    Build attention mask matrix for decoder when it autoregressing.

    parameters:
    seq: [batch, target_len]

    return:
    lookahead_mask: [batch, target_len, target_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, target_len, target_len]
    lookahead_mask = np.triu(np.ones(attn_shape), k=1) # [batch, target_len, target_len]
    lookahead_mask = torch.from_numpy(lookahead_mask)

    return lookahead_mask


def get_lookself_mask(seq):
    '''
    Build attention mask matrix for look only self

    parameters:
    seq: [batch, target_len]

    return:
    lookahead_mask: [batch, target_len, target_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, n_seq, n_seq]
    eye = torch.eye(seq.size(1))
    lookahead_mask = torch.ones(attn_shape) - eye # [batch, n_seq, n_seq]

    return lookahead_mask




class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(**kwargs) + x)


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
    def forward(self, inputs):
        return self.net(inputs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads    # dim_head concat
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.w_q = nn.Linear(dim, inner_dim, bias=False)
        self.w_k = nn.Linear(dim, inner_dim, bias=False)
        self.w_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()


    def forward(self, q_input, k_input, v_input, attn_mask=None):

        q, k, v = self.w_q(q_input), self.w_k(k_input), self.w_v(v_input)  # (b, n, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q,k,v])   # (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1) # [batch, n_heads, seq_len, seq_len]
            dots.masked_fill_(attn_mask, -1e9)
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
                PostNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PostNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.mlp_head = nn.Linear(dim, 2)


    def forward(self, enc_token, enc_input):
        """
        Return
        ----------
        enc_input: tensor, [B, n_seq, feature_dim]
        """
        attn_padding_mask = get_padding_mask(enc_token, enc_token) # [batch, target_len, target_len]
        for attn, ff in self.layers:
            x = attn(enc_input, q_input=enc_input, k_input=enc_input, v_input=enc_input, attn_mask=attn_padding_mask)
            x = ff(x, inputs=x)
        omega = self.mlp_head(x)  # [B, n_seq, 2]

        return omega, x



class Decoder(nn.Module):

  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
    super().__init__()

    self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
    self.pe = nn.Linear(3, dim)

    self.layers = nn.ModuleList([])
    for _ in range(depth):
        self.layers.append(nn.ModuleList([
            PostNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # decoder
            PostNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # decoder-encoder
            PostNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

  def forward(self, enc_token, enc_output, dec_token, dec_input):
    '''
    Params:
    ------------------
    enc_token: [B, n_seq]
    dec_token: [B, n_seq]
    encoder_output: [B, n_phase, dim_pst]
    decoder_input: [B, n_pos, dim_pos]

    Returns
    x: [B, n_pos, feature_dim]
    ------------------
    '''

    attn1_padding_mask = get_padding_mask(dec_token, dec_token) # [batch, target_len, target_len]
    attn1_lookahead_mask = get_lookahead_mask(dec_token).cuda() # [batch, target_len, target_len]
    attn1_mask = torch.gt(attn1_padding_mask + attn1_lookahead_mask, 0)

    attn2_padding_mask = get_padding_mask(dec_token, enc_token) # [batch, target_len, source_len]
    attn2_lookself_mask = get_lookself_mask(dec_token)
    attn2_mask = torch.gt(attn2_padding_mask + attn2_lookself_mask, 0)

    dec_input = self.pe(dec_input)
    dec_input += self.pos_embedding                  # 加位置嵌入（直接加）      (b, 1, dim)

    # masked mutlihead attention
    for attn1, attn2, ff in self.layers:
        x = attn1(dec_input,  q_input=dec_input, k_input=dec_input, v_input=dec_input, attn_mask=attn1_mask)
        x = attn2(x,  q_input=x, k_input=enc_output, v_input=enc_output, attn_mask=attn2_mask)
        x = ff(x, inputs=x)
    return x




class LocGPT(nn.Module):
    def __init__(self, **kwargs):
        super(LocGPT, self).__init__()
        self.encoder1, self.encoder2, self.encoder3 = [Encoder(**kwargs) for _ in range(3)]
        self.decoder = Decoder(**kwargs)
        self.pos_linear = nn.Linear(kwargs['dim'], 3, bias=False)
        self.pe, _ = get_embedder(multires=10, input_ch=1)  # 1 -> 1x2x10


    def forward(self, enc_token, enc_input, dec_token, dec_input, gateway_pos):
        """
        Params
        --------------
        enc_token: tensor, [B, n_seq]. The index of spectrum as the input of encoder
        enc_input: tensor, [B, n_seq, n_gateway, spt_dim]
        dec_token: [B, n_seq]
        dec_input: [B, n_seq, 3]
        """
        B, n_seq = enc_token.shape

        omega1, enc_output1 = self.encoder1(enc_input[..., 0, :])
        omega2, enc_output2 = self.encoder2(enc_input[..., 1, :])
        omega3, enc_output3 = self.encoder3(enc_input[..., 2, :])

        enc_output = enc_output1 + enc_output2 + enc_output3

        s = torch.zeros((B, n_seq, 1), dtype=torch.float32).to(omega1.device)
        for i in range(n_seq): # [B, n_seq, 2]
            s[:,i] = area(omega1[:,i], omega2[:,i], omega3[:,i])

        #NOTE
        dec_output = self.decoder(enc_token, enc_output, dec_token, dec_input)
        pos = self.pos_linear(dec_output)

        return torch.cat((s, pos), dim=-1)

