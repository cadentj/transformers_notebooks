import torch
from nnsight import LanguageModel
import einops
import matplotlib.pyplot as plt
import torch.nn as nn

MODEL = LanguageModel("gpt2", dispatch=True, device_map="auto")

def get_split_l0_heads(head_idx=0):
    W_Q, W_K, _ = torch.split(MODEL.transformer.h[0].attn.c_attn.weight, 768, dim=1)
    b_Q, b_K, _ = torch.split(MODEL.transformer.h[0].attn.c_attn.bias, 768)

    W_Q = einops.rearrange(
        W_Q, 
        "d_model (n_heads d_head) -> n_heads d_model d_head", 
        n_heads=12
    )[head_idx]
    b_Q = einops.rearrange(b_Q, "(n_heads d_head) -> n_heads d_head", n_heads=12)
    W_K = einops.rearrange(
        W_K, 
        "d_model (n_heads d_head) -> n_heads d_model d_head", 
        n_heads=12
    )[head_idx]
    b_K = einops.rearrange(b_K, "(n_heads d_head) -> n_heads d_head", n_heads=12)

    W_Q.bias = b_Q[head_idx]
    W_K.bias = b_K[head_idx]

    return W_Q, W_K

def get_pre_attn(
    prompt = "When Mary and John went to the store, John gave a drink to"
):
    l0_attn = MODEL.transformer.h[0].attn
    str_tokens = [MODEL.tokenizer.decode(t) for t in MODEL.tokenizer.encode(prompt)]

    with MODEL.trace(prompt):
        pre_attn = l0_attn.input[0][0].save()

    return str_tokens, pre_attn[0]
