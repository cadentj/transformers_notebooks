import torch
import einops
from transformers import AutoModel, AutoConfig, AutoTokenizer
from nnsight import LanguageModel

from cogmodel import CogConfig, CogModelForHF

MODEL = LanguageModel("openai-community/gpt2", dispatch=True, device_map="auto")

def get_split_l0_heads(head_idx: int):

    """
    Load the weights of the Q and K matrices for a specified head in the 0th layer of the GPT2.
    Automatically splits the concatenated weights into per head Q and K matrices.

    Args:
        head_idx (int): The index of the head to extract the weights for.

    Returns:
        W_Q (Tensor): The weights of the Q matrix for the specified head.
        W_K (Tensor): The weights of the K matrix for the specified head.
    """



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
    prompt: str = "When Mary and John went to the store, John gave a drink to"
):
    """
    This function accepts a prompt and returns the stringified tokens and activations before the 0th layer. 
    Recall that the activations before the 0th layer are the input embeddings.

    Args:
        prompt (str): The prompt to generate the tokens and activations for. 

    Returns:
        str_tokens (List[str]): The stringified tokens of the prompt. 
        pre_attn (Tensor): The activations before the 0th layer.
    """
    l0_attn = MODEL.transformer.h[0].attn
    str_tokens = [MODEL.tokenizer.decode(t) for t in MODEL.tokenizer.encode(prompt)]

    with MODEL.trace(prompt):
        pre_attn = l0_attn.input[0][0].save()

    return str_tokens, pre_attn[0]

def load_cogmodel():
    """
    Args:
        None

    Returns:
        model (CogModelForHF): The CogModelForHF model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
    """
    AutoConfig.register("custom_transformer", CogConfig)
    AutoModel.register(CogConfig, CogModelForHF)
    
    config = AutoConfig.from_pretrained("kh4dien/cogmodel")
    model = AutoModel.from_pretrained("kh4dien/cogmodel", config=config)
    tokenizer = AutoTokenizer.from_pretrained("kh4dien/cogmodel")
    return model, tokenizer
