from typing import cast, Tuple, Unpack
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.generic import TransformersKwargs
import torch
from .config_SLM import SLMConfig


def precompute_rope_freqs(dim: int, max_seq_length: int = 32*1024, rope_base: float = 1e6) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs, atten_factor = 1.0/ (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), 1.0
    t = torch.arange(max_seq_length, device=freqs.device)

    # Length * dim
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cos(freqs.repeat(1, 2)) * atten_factor
    freqs_sin = torch.sin(freqs.repeat(1, 2)) * atten_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_weight: torch.Tensor, sin_weight: torch.Tensor, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin_weight.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin_weight.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))


    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(input_seq_emb.float())).type_as(input_seq_emb)


class FFN(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = int(8*hidden_size/3)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


    def forward(self, input_embs: torch.Tensor) -> torch.Tensor:
        output = F.silu(self.gate(input_embs) * self.up_proj(input_embs))
        output = self.down_proj(output)

        return output


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    [B, kv_head_num, L, dims] -> [B, attention_head_num, L, dims]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    droput_p: float = 0.0,
    **kwargs: Unpack[TransformersKwargs]
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(keys, module.num_key_value_groups)
    value_states = repeat_kv(values, module.num_key_value_groups)

    attention_weight = torch.matmul(queries, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_weight = attention_weight + attention_mask

    attention_weight = F.softmax(attention_weight, dim=-1).to(queries.dtype)
    attention_weight = F.dropout(attention_weight, droput_p, training=module.training)
    attention_output = torch.matmul(attention_weight, value_states)
    attention_output = attention_output

    return attention_output, attention_weight


class Attention(nn.Module):
    def __init__(self, config: SLMConfig, layer_id: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        if config.num_key_value_heads is None:
            self.num_key_value_heads = config.num_attention_heads
            self.num_key_value_groups = 1
        else:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_casual = config.is_casual
        self.p = config.p
        self.layer_id = layer_id

        self.WQ = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=False)
        self.WV = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.WK = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.score_dropout = nn.Dropout(p=self.p)
        self.output_project = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(self,
        input_embs: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        B, L, d = input_embs.shape
        queries = cast(torch.Tensor, self.WQ(input_embs)).reshape(B, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
        keys = cast(torch.Tensor, self.WK(input_embs)).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = cast(torch.Tensor, self.WV(input_embs)).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        queries, keys = cast(torch.Tensor, self.q_norm(queries)), cast(torch.Tensor, self.k_norm(keys))

        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, unsqueeze_dim=0)

        if past_key_values is not None:
            keys, values = past_key_values.update(keys, values, self.layer_id)

        # print(queries.shape, keys.shape, values.shape)
        # key_states = repeat_kv(keys, self.num_key_value_groups)
        # value_states = repeat_kv(values, self.num_key_value_groups)
        # p = self.p if self.training else 0.0
        # attn_output = F.scaled_dot_product_attention(
        #     query=queries,
        #     key=key_states,
        #     value=value_states,
        #     attn_mask=attention_mask,
        #     dropout_p=p,
        #     is_causal=False,
        #     scale=self.scaling
        # )

        attn_output, attn_weight = eager_attention_forward(
            module=self,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=self.scaling,
            droput_p=self.p,
            **kwargs
        )

        output = attn_output.transpose(1, 2).contiguous().view(B, L, d)
        output = self.output_project(output)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, config: SLMConfig, layer_id: int) -> None:
        super().__init__()
        self.attention = Attention(config, layer_id)
        self.LN1 = RMSNorm(config.hidden_size)
        self.FFN = FFN(config)
        self.LN2 = RMSNorm(config.hidden_size)


    def forward(self,
        input_ids_embs: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Attention
        normed = self.LN1(input_ids_embs)
        embs = self.attention(
            normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        x = input_ids_embs + embs

        # FFN
        normed = self.LN2(x)
        embs = self.FFN(normed)
        output_embs = x + embs

        return output_embs


class SLMModel(PreTrainedModel):
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        head_dims = config.hidden_size // config.num_attention_heads
        freqs_cos, freqs_sin = precompute_rope_freqs(dim=head_dims)  # TODO: in the future, it will change to dynamic calculate.
        self.freqs_cos = nn.Buffer(freqs_cos, persistent=False)
        self.freqs_sin = nn.Buffer(freqs_sin, persistent=False)
        self.output_norm = RMSNorm(config.hidden_size)

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = AttentionBlock(config=config, layer_id=i)
            self.layers.append(layer)

        self.post_init()


    def forward(self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length()if past_key_values is not None else 0
        L = input_ids.shape[1]

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = (self.freqs_cos[past_seen_tokens:L+past_seen_tokens], self.freqs_sin[past_seen_tokens:L+past_seen_tokens])

        attention_mask = torch.log(torch.tril(torch.ones(L, L+past_seen_tokens), diagonal=past_seen_tokens)).to(hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
                )

        hidden_states = self.output_norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class SLMforCasualLM(PreTrainedModel, GenerationMixin):
    config_class = SLMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        return_dict: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        output: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
            )

        hidden_states = output.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if hidden_states is None: raise ValueError("The model should output hidden_states.")
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # CrossEntropy
            y_pred, y = logits[:, :-1].contiguous(), labels[:, 1:]
            loss = F.cross_entropy(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=(hidden_states, )
        )

    @torch.inference_mode()
    def naive_generate(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        length = input_ids.shape[-1]
        max_length = max(length, max_length)


        response = torch.zeros((1, max_length), dtype=torch.int32).to(input_ids.device)
        response[:, :length] = input_ids[:1, :length]

        for i in range(length, max_length):
            output_pred = self.lm_head(self.model(response[:, :i]).last_hidden_state)
            response[:, i] = torch.argmax(output_pred[:, -1], dim=-1)
            del output_pred

        return response


if __name__ == "__main__":
    '''Module Test'''
    vocab_size = 256
    hidden_size = 64
    num_hidden_layers = 6
    config = SLMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=1)
    model = SLMforCasualLM(config=config)

    # input
    input_ids = torch.randint(0, vocab_size, (1, 4))
    attention_mask = torch.randint(0, vocab_size, (1, 4))
    output = model(input_ids, attention_mask)
    print(output)

    gen_res = model.generate(input_ids, max_length=8, use_cache=True)
    print(gen_res)
    gen_resv2 = model.generate(input_ids, max_length=8, use_cache=False)
    print(gen_resv2)
    gen_na_res = model.naive_generate(input_ids, max_length=8)
    print(gen_na_res)