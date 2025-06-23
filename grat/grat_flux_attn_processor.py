import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
create_block_mask = torch.compile(create_block_mask)
from typing import Optional
from functools import partial, lru_cache

@lru_cache
def init_local_mask_flex(height, width, text_length, attenable_text, group_h, group_w, device):
    total_length = height * width
    cell_size = group_h * group_w
    h, w = height // group_h, width // group_w
    def local_mask(b, h_, q_idx, kv_idx):
        #q_y = q_idx // cell_size
        #kv_y = kv_idx // cell_size
        #q_h = q_y // w
        #q_w = q_y % w
        #kv_h = kv_y // w
        #kv_w = kv_y % w
        q_y = q_idx // cell_size
        kv_y = kv_idx // cell_size
        q_h = (q_y%(h*w))//w
        q_w = (q_y%(h*w))%w
        kv_h = (kv_y%(h*w))//w
        kv_w = (kv_y%(h*w))%w

        text = kv_idx < total_length + attenable_text
        text2 = torch.logical_or(
            q_idx >= total_length,
            torch.logical_and(kv_idx < total_length + attenable_text, kv_idx >= total_length)
        )
        ## GRAT-X 
        image = torch.logical_and(
            torch.logical_or(q_h == kv_h, q_w == kv_w),
            q_idx < total_length
        )
        return torch.logical_and(image | text2, text)

    BLOCK_MASK = create_block_mask(
        local_mask,
        B=None,
        H=None,
        device=device,
        Q_LEN=text_length + height * width,
        KV_LEN=text_length + height * width,
        _compile=True
    )
    return BLOCK_MASK

class GratFluxAttnProcessor:
    def __init__(self, mask, height, width, group_h, group_w, text_length):
        self.flex_attn = partial(flex_attention, block_mask=mask)
        self.mask=mask
        self.flex_attn = torch.compile(self.flex_attn, dynamic=False)
        self.height = height
        self.width = width
        self.group_h = group_h
        self.group_w = group_w
        self.text_length = text_length

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "GratFluxAttnProcessor requires PyTorch 2.0"
            )

    def clusterify(self, row):
        bsz, head, n, c = row.shape
        p_h, p_w = self.group_h, self.group_w
        h, w = self.height, self.width
        h_, w_ = h // p_h, w // p_w
        #[1, 24, 3072, 128]
        row = row.reshape(bsz, head, h_, p_h, w_, p_w, c)
        row = torch.einsum('nxhpwqc->nxhwpqc', row)
        row = row.reshape(bsz, head, -1, c)
        return row

    def unclusterify(self, row):
        bsz, head, n, c = row.shape
        p_h, p_w = self.group_h, self.group_w
        h, w = self.height, self.width
        h_, w_ = h // p_h, w // p_w
        row = row.reshape(bsz, head, h_, w_, p_h, p_w, c)
        row = torch.einsum('nxhwpqc->nxhpwqc', row)
        row = row.reshape(bsz, head, -1, c)
        return row

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query_image, query_text = query[:, :, :-self.text_length], query[:, :, -self.text_length:]
        key_image, key_text = key[:, :, :-self.text_length], key[:, :, -self.text_length:]
        value_image, value_text = value[:, :, :-self.text_length], value[:, :, -self.text_length:]

        query_image = self.clusterify(query_image)
        key_image = self.clusterify(key_image)
        value_image = self.clusterify(value_image)

        query = torch.cat([query_image, query_text], dim=2)
        key = torch.cat([key_image, key_text], dim=2)
        value = torch.cat([value_image, value_text], dim=2)

        hidden_states = self.flex_attn(query, key, value)

        hidden_states_image, hidden_states_text = hidden_states[:, :, :-self.text_length], hidden_states[:, :, -self.text_length:]
        hidden_states_image = self.unclusterify(hidden_states_image)
        hidden_states = torch.cat([hidden_states_image, hidden_states_text], dim=2)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        #hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states 
