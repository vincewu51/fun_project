# mmdit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size  # Fixed typo

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps):
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0 and train
        if force_drop_ids is not None:
            drop_ids = force_drop_ids
        elif use_dropout:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.zeros_like(labels).bool()
        
        labels = labels.clone()
        labels[drop_ids] = self.num_classes  # Use class as a null token for CFG
        return self.embedding_table(labels)

class CaptionEmbedder(nn.Module):
    """
    Embeds text captions into vector representations.
    """
    def __init__(self, in_channels, hidden_size, tokenizer_max_length=77, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.tokenizer_max_length = tokenizer_max_length
        
        # Simple token embedding layer (in practice, you'd use a pre-trained text encoder)
        self.token_embedding = nn.Embedding(in_channels, hidden_size)
        self.positional_embedding = nn.Parameter(torch.empty(tokenizer_max_length, hidden_size))
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, caption, train=True):
        # caption shape: [batch_size, seq_len]
        x = self.token_embedding(caption) + self.positional_embedding[:caption.shape[1]]
        x = self.linear(x)
        
        # Dropout during training for classifier-free guidance
        if train and self.dropout_prob > 0:
            x = self.dropout(x)
            
        return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        # Output shape after proj: [B, embed_dim, grid_size, grid_size]
        # Rearrange to: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class Attention(nn.Module):
    """
    Multi-head attention module with optional context conditioning.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape
        
        # For self-attention
        if context is None:
            context = x
        
        # Compute Q from x and K,V from context
        q = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k, v = self.qkv(context).reshape(B, context.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[1:3]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer, etc.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        
        # Condition processing layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Initialize adaLN_modulation to output zeros for adaLN-Zero
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c, context=None):
        # Process conditioning signal
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with adaLN-Zero
        x = x + gate_msa.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1), context)
        
        # MLP with adaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class MMDiT(nn.Module):
    """
    Multi-Modal Diffusion Transformer
    """
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        caption_channels=768,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Patch embedding
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size, bias=True)
        
        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Class label embedding (for unconditional/class-conditional generation)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.1)
        
        # Caption embedding
        self.context_embedder = CaptionEmbedder(caption_channels, caption_channels)  # Fixed initialization
        self.caption_projection = nn.Linear(caption_channels, hidden_size)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding layer
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, context=None, train=True):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (unconditional/class-conditional generation)
        context: (N, L, C) tensor of text embeddings (multi-modal generation)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        
        t = self.t_embedder(t)                   # (N, D)
        
        if y is not None:
            y = self.y_embedder(y, train)        # (N, D)
            
        if context is not None:
            context = self.context_embedder(context, train)
            context = self.caption_projection(context)  # (N, L, D)
            
        # Combine time and class embeddings
        if y is not None:
            c = t + y
        else:
            c = t
            
        # Apply blocks
        for block in self.blocks:
            x = block(x, c, context)
            
        # Output layer
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y=None, context=None, cfg_scale=4.0):
        """
        Forward pass with classifier-free guidance.
        """
        # Double the batch for CFG
        x_in = torch.cat([x] * 2, dim=0)
        t_in = torch.cat([t] * 2, dim=0)
        
        # Process context and labels for CFG
        if context is not None:
            # Create null context by zeroing embeddings
            null_context = torch.zeros_like(context)
            context_in = torch.cat([context, null_context], dim=0)
        else:
            context_in = None
            
        if y is not None:
            # Create null labels
            null_y = torch.full_like(y, self.y_embedder.num_classes)  # Use the null token
            y_in = torch.cat([y, null_y], dim=0)
        else:
            y_in = None
            
        # Forward pass
        out = self.forward(x_in, t_in, y=y_in, context=context_in, train=False)
        
        # Split output and apply CFG
        cond_out, uncond_out = torch.chunk(out, 2, dim=0)
        cfg_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return cfg_out

# mmdit_small.py
def MMDiT_S_2(**kwargs):
    return MMDiT(depth=2, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def MMDiT_S_4(**kwargs):
    return MMDiT(depth=2, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def MMDiT_B_2(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MMDiT_B_4(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def MMDiT_L_2(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MMDiT_L_4(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)