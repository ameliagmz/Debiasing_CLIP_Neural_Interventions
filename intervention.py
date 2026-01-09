import os
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
from types import MethodType
import open_clip
from PIL import Image
from sklearn.metrics import ndcg_score, average_precision_score
from open_clip import tokenizer
from tqdm import tqdm


def replace_activations(attentions, head, new_attention):
    # Replace the attention for the specified head
    attentions[head-1][0] = new_attention
    
    # attentions[head-1][0] *= 0.1 # Scaling

    return attentions

# Custom multi_head_attention_forward function
def custom_multi_head_attention_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:

    # The actual implementation from the original `multi_head_attention_forward`
    is_batched = query.dim() == 3

    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # Set up shape variables
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if not use_separate_proj_weight:
        q_proj_weight_non_opt = in_proj_weight[:embed_dim, :]
        k_proj_weight_non_opt = in_proj_weight[embed_dim:2 * embed_dim, :]
        v_proj_weight_non_opt = in_proj_weight[2 * embed_dim:, :]
    else:
        q_proj_weight_non_opt = q_proj_weight
        k_proj_weight_non_opt = k_proj_weight
        v_proj_weight_non_opt = v_proj_weight

    if in_proj_bias is not None:
        b_q, b_k, b_v = in_proj_bias.chunk(3)
    else:
        b_q = b_k = b_v = None

    # In-projection
    q = F.linear(query, q_proj_weight_non_opt, b_q)
    k = F.linear(key, k_proj_weight_non_opt, b_k)
    v = F.linear(value, v_proj_weight_non_opt, b_v)

    # Reshape and permute for multi-head attention
    head_dim = embed_dim // num_heads
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

    # Apply the attention mask
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            assert attn_mask.size(0) == bsz * num_heads
        attn_mask = attn_mask.to(q.device)

    # Apply key padding mask
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1)
        key_padding_mask = key_padding_mask.reshape(bsz * num_heads, 1, src_len)

    # Compute scaled dot-product attention
    attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
    attn_output_weights = attn_output_weights / (head_dim ** 0.5)

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)

    # Alternative formulation of multi-head attention
    split_out_proj_weights = torch.chunk(out_proj_weight, num_heads, dim=1)  # Split into (12, 64, 768)
    projected_heads = []
    for i in range(num_heads):
        # attn_output[i] has shape (tgt_len, head_dim) -> (50, 64)
        # split_out_proj_weights[i] has shape (64, 768)
        # We want to project each head's output from (50, 64) to (50, 768)
        projected_head = torch.matmul(attn_output[i], split_out_proj_weights[i].T)  # Shape: (50, 768)
        projected_heads.append(projected_head)

    # INTERVENTION
    for head, new_attn in zip(self.HEADS, self.NEW_ATTENTIONS):
        projected_heads = replace_activations(projected_heads, head, new_attn)

    projected_heads = torch.stack(projected_heads)  # Shape: (12, 50, 768)

    projected_heads_sum = torch.sum(projected_heads, dim=0)  # Shape: (50, 768)
    projected_heads_sum += out_proj_bias  # Shape: (50, 768)
    projected_heads_sum = projected_heads_sum.view(tgt_len, bsz, embed_dim)  # Shape: (50, 1, 768)

    # Original formulation
    # Combine heads
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # Output projection
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    # Check if the two formulations give the same output
    #print(torch.allclose(projected_heads_sum, attn_output, atol=1e-6))
    attn_output = projected_heads_sum

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    else:
        attn_output_weights = None

    if not is_batched:
        attn_output = attn_output.squeeze(1)
        if need_weights:
            attn_output_weights = attn_output_weights.squeeze(0)

    return attn_output, attn_output_weights, projected_heads

# Define a custom forward function that mimics the original behavior, including manual qkv projection
def custom_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:


    why_not_fast_path = ''
    if ((attn_mask is not None and torch.is_floating_point(attn_mask))
       or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
        why_not_fast_path = "floating-point masks are not supported for fast path."

    is_batched = query.dim() == 3

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )


    if not is_batched:
        why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    elif query is not key or key is not value:
        # When lifting this restriction, don't forget to either
        # enforce that the dtypes all match or test cases where
        # they don't!
        why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
        why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
    elif self.in_proj_weight is None:
        why_not_fast_path = "in_proj_weight was None"
    elif query.dtype != self.in_proj_weight.dtype:
        # this case will fail anyway, but at least they'll get a useful error message.
        why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
    elif self.training:
        why_not_fast_path = "training is enabled"
    elif (self.num_heads % 2) != 0:
        why_not_fast_path = "self.num_heads is not even"
    elif not self.batch_first:
        why_not_fast_path = "batch_first was not True"
    elif self.bias_k is not None:
        why_not_fast_path = "self.bias_k was not None"
    elif self.bias_v is not None:
        why_not_fast_path = "self.bias_v was not None"
    elif self.add_zero_attn:
        why_not_fast_path = "add_zero_attn was enabled"
    elif not self._qkv_same_embed_dim:
        why_not_fast_path = "_qkv_same_embed_dim was not True"
    elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
        why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                             is not supported with NestedTensor input"
    elif torch.is_autocast_enabled():
        why_not_fast_path = "autocast is enabled"

    if not why_not_fast_path:
        tensor_args = (
            query,
            key,
            value,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj.weight,
            self.out_proj.bias,
        )


    if self.batch_first and is_batched:
        # make sure that the transpose op does not affect the "is" property
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = (x.transpose(1, 0) for x in (query, key))
                value = key
        else:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

    attn_output, attn_output_weights, projected_heads = custom_multi_head_attention_forward(
            self,
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal)
    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights, projected_heads
    else:
        return attn_output, attn_output_weights, projected_heads

def replace_forward_only(model, layer_index, heads, new_attentions):
    # Enumerate through all modules to find MultiheadAttention layers
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.MultiheadAttention):
            # Replace the forward function only if the index matches the target
            target_name = f'visual.transformer.resblocks.{layer_index-1}.attn'

            if name == target_name:
                module.HEADS = heads
                module.NEW_ATTENTIONS = new_attentions

                module.forward = MethodType(custom_forward, module)


def generate_attention_experiments(layer, heads, avg_attn_folder):
    """
    Automatically generates intervention experiment configs for a given layer and set of heads.
    
    Returns:
        new_attn_exp: list of lists of tensors
        heads_exp: list of lists of head indices
        layers_exp: list of layer indices
        interventions_exp: list of intervention names
    """

    categories = ["person", "male", "female"]
    # categories = ["male_White"]
    zero_ablation = torch.zeros(768)

    new_attn_exp = [[]]          # first entry: baseline (original)
    heads_exp = [[]]             # empty heads for baseline
    layers_exp = [-1]            # -1 for original model
    interventions_exp = ["original"]

    # --- Generate experiments for each category ---
    for category in categories:
        # Load tensors for all heads of this category
        tensors = []
        for h in heads:
            path = os.path.join(avg_attn_folder, f"{category}_l{layer}_h{h}.pt")
            tensor = torch.load(path, weights_only=True)
            tensors.append(tensor)

        # 1. Add experiment with all heads combined
        new_attn_exp.append(tensors)
        heads_exp.append(heads)
        layers_exp.append(layer)
        interventions_exp.append(category)

        # 2. Add one experiment per head
        for i, h in enumerate(heads):
            new_attn_exp.append([tensors[i]])
            heads_exp.append([h])
            layers_exp.append(layer)
            interventions_exp.append(category)

    # --- Add zero ablation (all heads zeroed out) ---
    new_attn_exp.append([zero_ablation for _ in heads])
    heads_exp.append(heads)
    layers_exp.append(layer)
    interventions_exp.append("zero")

    return new_attn_exp, heads_exp, layers_exp, interventions_exp