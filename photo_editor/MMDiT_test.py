import torch
import pytest
from MMDiT import (
    TimestepEmbedder, LabelEmbedder, CaptionEmbedder, 
    PatchEmbed, Attention, Mlp, DiTBlock, FinalLayer, MMDiT,
    MMDiT_S_2, MMDiT_S_4, MMDiT_B_2, MMDiT_B_4, MMDiT_L_2, MMDiT_L_4
)

def test_timestep_embedder():
    """Test the TimestepEmbedder module"""
    embedder = TimestepEmbedder(hidden_size=384)
    timesteps = torch.randint(0, 1000, (4,))
    embedding = embedder(timesteps)
    assert embedding.shape == (4, 384)

def test_label_embedder():
    """Test the LabelEmbedder module"""
    embedder = LabelEmbedder(num_classes=1000, hidden_size=384, dropout_prob=0.1)
    labels = torch.randint(0, 1000, (4,))
    
    # Test training mode
    embedding_train = embedder(labels, train=True)
    assert embedding_train.shape == (4, 384)
    
    # Test inference mode
    embedding_infer = embedder(labels, train=False)
    assert embedding_infer.shape == (4, 384)

def test_caption_embedder():
    """Test the CaptionEmbedder module"""
    embedder = CaptionEmbedder(in_channels=1000, hidden_size=384)
    captions = torch.randint(0, 1000, (4, 77))  # batch_size=4, seq_len=77
    
    # Test training mode
    embedding_train = embedder(captions, train=True)
    assert embedding_train.shape == (4, 77, 384)
    
    # Test inference mode
    embedding_infer = embedder(captions, train=False)
    assert embedding_infer.shape == (4, 77, 384)

def test_patch_embed():
    """Test the PatchEmbed module"""
    patch_embed = PatchEmbed(img_size=32, patch_size=2, in_chans=4, embed_dim=384)
    x = torch.randn(4, 4, 32, 32)  # batch_size=4, channels=4, height=32, width=32
    embedding = patch_embed(x)
    assert embedding.shape == (4, 256, 384)  # 256 patches (16x16 grid)

def test_attention():
    """Test the Attention module"""
    attn = Attention(dim=384, num_heads=6)
    x = torch.randn(4, 256, 384)  # batch_size=4, seq_len=256, dim=384
    
    # Test self-attention
    output = attn(x)
    assert output.shape == (4, 256, 384)
    
    # Test cross-attention
    context = torch.randn(4, 77, 384)  # context from text encoder
    cross_output = attn(x, context=context)
    assert cross_output.shape == (4, 256, 384)

def test_mlp():
    """Test the Mlp module"""
    mlp = Mlp(in_features=384, hidden_features=1536, out_features=384)
    x = torch.randn(4, 256, 384)
    output = mlp(x)
    assert output.shape == (4, 256, 384)

def test_dit_block():
    """Test the DiTBlock module"""
    block = DiTBlock(hidden_size=384, num_heads=6)
    x = torch.randn(4, 256, 384)  # input tensor
    c = torch.randn(4, 384)       # conditioning signal
    
    # Test without context
    output = block(x, c)
    assert output.shape == (4, 256, 384)
    
    # Test with context
    context = torch.randn(4, 77, 384)
    output_with_context = block(x, c, context)
    assert output_with_context.shape == (4, 256, 384)

def test_final_layer():
    """Test the FinalLayer module"""
    final_layer = FinalLayer(hidden_size=384, patch_size=2, out_channels=8)
    x = torch.randn(4, 256, 384)
    c = torch.randn(4, 384)
    output = final_layer(x, c)
    assert output.shape == (4, 256, 32)  # 2*2*8 = 32

def test_mmdit_forward():
    """Test the main MMDiT model forward pass"""
    model = MMDiT(
        img_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=2,
        num_heads=6,
        num_classes=1000,
        caption_channels=768,
        learn_sigma=True
    )
    
    # Test inputs
    x = torch.randn(4, 4, 32, 32)     # image/latent tensor
    t = torch.randint(0, 1000, (4,))  # timesteps
    
    # Test unconditional generation
    output_uncond = model(x, t)
    assert output_uncond.shape == (4, 8, 32, 32)  # 8 channels due to learn_sigma
    
    # Test class-conditional generation
    y = torch.randint(0, 1000, (4,))
    output_class_cond = model(x, t, y=y)
    assert output_class_cond.shape == (4, 8, 32, 32)
    
    # Test text-conditional generation
    context = torch.randint(0, 1000, (4, 77))  # text tokens
    output_text_cond = model(x, t, context=context)
    assert output_text_cond.shape == (4, 8, 32, 32)
    
    # Test combined conditioning
    output_combined = model(x, t, y=y, context=context)
    assert output_combined.shape == (4, 8, 32, 32)

def test_mmdit_model_variants():
    """Test all MMDiT model variants"""
    # Test small models
    model_s2 = MMDiT_S_2()
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    output_s2 = model_s2(x, t)
    assert output_s2.shape == (2, 8, 32, 32)
    
    model_s4 = MMDiT_S_4()
    output_s4 = model_s4(x, t)
    assert output_s4.shape == (2, 8, 32, 32)
    
    # Test base models
    model_b2 = MMDiT_B_2()
    output_b2 = model_b2(x, t)
    assert output_b2.shape == (2, 8, 32, 32)
    
    model_b4 = MMDiT_B_4()
    output_b4 = model_b4(x, t)
    assert output_b4.shape == (2, 8, 32, 32)
    
    # Test large models
    model_l2 = MMDiT_L_2()
    output_l2 = model_l2(x, t)
    assert output_l2.shape == (2, 8, 32, 32)
    
    model_l4 = MMDiT_L_4()
    output_l4 = model_l4(x, t)
    assert output_l4.shape == (2, 8, 32, 32)

def test_mmdit_unpatchify():
    """Test the unpatchify function"""
    model = MMDiT_S_2()
    x = torch.randn(4, 256, 32)  # patchified tensor
    imgs = model.unpatchify(x)
    assert imgs.shape == (4, 8, 32, 32)

if __name__ == "__main__":
    pytest.main([__file__])