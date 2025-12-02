import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# =======================
# 1. Config
# =======================

class LLMConfig(PretrainedConfig):
    model_type = "llm_dense"

    def __init__(
        self,
        vocab_size: int = 6400,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = 2,   # GQA: if None, = num_attention_heads
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_swiglu=True,   # Use SwiGLU FFN instead of a standard MLP
        rope_scaling: dict = None,
        flash_attn: bool = False,  # Enable flash attention when available
        # flash_attn_impl: str = "auto",  # "auto" | "sdpa" | "xformers" | "none" (kept for future finer control)
        # (custom flash_attn_impl left commented for future use)
        # flash_attn_impl: str = "auto",  # "auto" | "sdpa" | "xformers" | "none"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.use_swiglu = use_swiglu
        self.flash_attn = flash_attn
        self.rope_scaling = rope_scaling or {"rope_type": "default"}


# =======================
# 2. Core blocks: RMSNorm + RoPE + FFN
# =======================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding with optional YaRN frequency-domain scaling.

    Args:
        dim: per-head hidden size.
        max_position_embeddings: maximum sequence length supported.
        base: RoPE base frequency constant.
        rope_scaling: optional dict for YaRN scaling (rope_type="yarn", original_max_position_embeddings, factor, beta_fast, beta_slow).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dim must be even"
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self.rope_scaling = rope_scaling or {}

        # ===== 1. Base inverse frequencies: inv_freq: [dim/2] =====
        # inv_freq[i] = 1 / base^((2i)/dim)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )  # [dim/2]

        # ===== 2. Optional YaRN scaling of inv_freq =====
        rope_type = self.rope_scaling.get("rope_type", None)

        if rope_type == "yarn":
            # Original maximum context length before scaling
            orig_max = int(self.rope_scaling.get("original_max_position_embeddings", 2048))
            # Scaling factor (e.g., 16 to scale 2k to ~32k)
            factor = float(self.rope_scaling.get("factor", 4.0))
            # YaRN beta parameters
            beta_fast = float(self.rope_scaling.get("beta_fast", 4.0))
            beta_slow = float(self.rope_scaling.get("beta_slow", 1.0))

            # Apply scaling only when target context exceeds original length
            if self.max_position_embeddings > orig_max:
                # Find cutoff dimension where wavelength exceeds original context
                #           From here onward the frequencies correspond to the "low-frequency band"
                half_dim = dim // 2
                freqs = inv_freq.clone()

                # Compute power schedule for beta interpolation
                # Keep this consistent with the earlier code:
                corr_dim = next(
                    (
                        i
                        for i in range(half_dim)
                        if 2 * math.pi / freqs[i].item() > orig_max
                    ),
                    half_dim,
                )

                # power: [0, 1] used for beta interpolation
                power = torch.arange(half_dim, device=freqs.device, dtype=torch.float32)
                power = power / max(half_dim - 1, 1)

                # Interpolate beta between beta_slow and beta_fast
                beta = beta_slow + (beta_fast - beta_slow) * power  # [half_dim]

                # YaRN scaling factors per dimension
                # Low-frequency band (< corr_dim): scale with lambda(factor, beta)
                # High-frequency band (>= corr_dim): uniformly scale to 1/factor
                idx = torch.arange(half_dim, device=freqs.device)
                scale_low = (beta * factor - beta + 1.0) / (beta * factor)  # [half_dim]
                scale_high = torch.full_like(scale_low, 1.0 / factor)

                scale = torch.where(idx < corr_dim, scale_low, scale_high)  # [half_dim]

                # Apply scaling to frequencies
                freqs = freqs * scale

                inv_freq = freqs

        # ===== 3. Precompute angles for all positions =====
        # t: [0, ..., max_position_embeddings - 1]
        t = torch.arange(
            self.max_position_embeddings, device=inv_freq.device, dtype=torch.float32
        )  # [max_seq]

        # outer(t, inv_freq): [max_seq, dim/2]
        # each element (pos, i) = pos * inv_freq[i]
        freqs = torch.outer(t, inv_freq).float()  # [max_seq, dim/2]

        # ===== 4. Cache cos/sin for fast lookup =====
        #
        emb_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        emb_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        self.register_buffer("cos_cached", emb_cos, persistent=False)
        self.register_buffer("sin_cached", emb_sin, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        """
        Apply rotary embedding to x.

        Args:
            x: [batch, n_heads, seq_len, head_dim]
            position_ids: [batch, seq_len] positions for each token

        Returns:
            x_rotated: [batch, n_heads, seq_len, head_dim]
        """
        bsz, n_heads, seq_len, dim = x.shape
        assert dim == self.dim, f"head_dim mismatch: got {dim}, expected {self.dim}"

        # Validate position range
        if position_ids.max() >= self.max_position_embeddings:
            raise ValueError(
                f"position_ids max {position_ids.max().item()} >= "
                f"max_position_embeddings {self.max_position_embeddings}"
            )

        # Gather cos/sin for the given positions
        cos = self.cos_cached[position_ids]  # [B, S, D]
        sin = self.sin_cached[position_ids]  # [B, S, D]

        # Broadcast to heads dimension
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)  # [B, 1, S, D]

        # Split even/odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Combine rotated components
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
        x_rotated[..., 1::2] = x_even * sin[..., ::2] + x_odd * cos[..., ::2]

        return x_rotated

class FeedForward(nn.Module):
    """
    Feed-forward network with optional SwiGLU variant.
    - use_swiglu=False: RMSNorm -> Linear -> SiLU -> Linear -> Dropout
    - use_swiglu=True: RMSNorm -> gate/up -> SiLU(gate)*up -> down -> Dropout
    """






    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.hidden_size = config.hidden_size
        self.use_swiglu = config.use_swiglu

        # Infer intermediate_size if not provided
        if config.intermediate_size is None:
            if self.use_swiglu:
                # Default SwiGLU width: ~8/3 * hidden, rounded to 64
                inter = int(config.hidden_size * 8 / 3)
                config.intermediate_size = 64 * ((inter + 63) // 64)
            else:
                # Default MLP width: 4 * hidden
                config.intermediate_size = 4 * config.hidden_size

        self.intermediate_size = config.intermediate_size

        hidden = self.hidden_size
        inter = self.intermediate_size

        # Standard 2-layer MLP

        # =========================
        if not self.use_swiglu:
            self.fc1 = nn.Linear(hidden, inter, bias=False)
            self.fc2 = nn.Linear(inter, hidden, bias=False)
            self.act = nn.SiLU()

        # =========================
        # SwiGLU variant
        # =========================
        else:
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj   = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)
            self.act = nn.SiLU()

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)

        # Standard MLP path
        if not self.use_swiglu:
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)

        # SwiGLU path
        else:
            gate = self.gate_proj(x)
            up   = self.up_proj(x)
            x = self.act(gate) * up
            x = self.down_proj(x)

        x = self.dropout(x)
        x = self.dropout(x)
        return x
# =======================
# 3. Attention (GQA + RoPE)
class Attention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.n_rep = self.num_heads // self.num_kv_heads  # GQA: KV head repetition factor

        self.c_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling={
                 "rope_type": "yarn",
                 "factor": 16.0,                       # example: scale 2k -> 32k
                 "original_max_position_embeddings": 2048,
                 "beta_fast": 4.0,
                 "beta_slow": 1.0,
            },
        )
        # FlashAttention toggle
        self.flash = getattr(config, "flash_attn", True)
        self.dropout = getattr(config, "dropout", 0.0)
        self.attn_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else None
        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else None

    def _shape(self, x: torch.Tensor, num_heads: int, bsz: int, seq_len: int) -> torch.Tensor:
        # [bsz, seq, num_heads*head_dim] -> [bsz, num_heads, seq, head_dim]
        return x.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def repeat_kv_bhsd(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads for grouped-query attention.
        Input:  [batch, kv_heads, seq_len, head_dim]
        Output: [batch, kv_heads * n_rep, seq_len, head_dim]

        """
        bsz, kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        # [B, kv_heads, 1, S, D]
        x = x.unsqueeze(2)
        # Expand KV heads along repetition dimension
        x = x.expand(bsz, kv_heads, n_rep, seq_len, head_dim)
        # Merge back into [B, kv_heads * n_rep, S, D]
        return x.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)
    # ========================================================
    # FlashAttention / fallback computation
    # ========================================================
    def _scaled_dot_product(self, q, k, v, attention_mask):

        bsz, n_heads, q_len, _ = q.shape
        _, _, k_len, _ = k.shape

        # ------ FlashAttention fast path ------
        can_flash = (
            self.flash
            and q_len > 1
            and k_len > 1
            and (attention_mask is None or torch.all(attention_mask == 0))
        )

        if can_flash:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # ------ Fallback: standard attention ------
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        causal = torch.full((q_len, k_len), float("-inf"), device=q.device)
        causal = torch.triu(causal, diagonal=1)
        attn_scores = attn_scores + causal.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(q)

        if self.attn_dropout:
            attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, v)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, 1, 1, seq_k]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states: [bsz, seq_q, hidden]
        attention_mask: [bsz, 1, seq_q, seq_k] or [bsz, 1, 1, seq_k]
        position_ids: [bsz, seq_q]
        """
        bsz, seq_len, _ = hidden_states.size()

        # 1. pre-RMSNorm
        x = self.norm(hidden_states)

        # 2. Project to Q, K, V
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        # 3. Reshape to [bsz, heads, seq, head_dim]
        q = self._shape(q, self.num_heads, bsz, seq_len)
        k = self._shape(k, self.num_kv_heads, bsz, seq_len)
        v = self._shape(v, self.num_kv_heads, bsz, seq_len)

        # 4. RoPE
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0).expand(bsz, -1)  # [bsz, seq]
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # 5. Append past_key_value
        if past_key_value is not None:
            # past_kv: [bsz, kv_heads, seq_past, dim]
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        # 6. GQA: repeat KV heads if needed
        if self.num_kv_heads != self.num_heads:
            # [bsz, kv_heads, seq_k, dim] -> [bsz, heads, seq_k, dim]
            k = self.repeat_kv_bhsd(k, self.n_rep)
            v = self.repeat_kv_bhsd(v, self.n_rep)

        # 7. Scaled dot product attention
        #------ Attention ------
        attn_output = self._scaled_dot_product(q, k, v, attention_mask)
        
        # 8. Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        if self.resid_dropout is not None:
            attn_output = self.resid_dropout(attn_output)
        return attn_output, present


# =======================
# 4. Transformer block: LLMLayer
# =======================

class LLMLayer(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # ---- (a) GQA ----
        residual = hidden_states
        attn_output, present = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output  # residual connection

        # ---- (b) FFN ----
        residual = hidden_states
        ffn_output = self.mlp(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states, present


# =======================
# 5. Stacked Transformer: LLMModel
# =======================

class LLMModel(PreTrainedModel):
    config_class = LLMConfig

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        input_ids: [bsz, seq]
        attention_mask: [bsz, seq]  (1 = keep, 0 = pad)
        position_ids: [bsz, seq] (defaults to 0..seq-1, offset when using cache)
        past_key_values: list of len(num_layers), each a (k, v) tuple
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            if past_key_values is None:
                past_length = 0
            else:
                # past_key shapes: [batch_size, num_kv_heads, past_seq_len, head_dim]
                #[batch_size, num_kv_heads, past_seq_len, head_dim]
                past_length = past_key_values[0][0].size(2)  # seq_past

            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)

        # [bsz, seq] -> [bsz, 1, 1, seq] attention mask
        if attention_mask is not None:
            # 1 for valid, 0 for pad -> 0 and -inf
            extended_mask = (1.0 - attention_mask[:, None, None, :]) * -1e4
        else:
            extended_mask = None

        hidden_states = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past = [] if use_cache else None

        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=extended_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past.append(present)

        hidden_states = self.final_norm(hidden_states)

        return hidden_states, new_past


# =======================
# 6.  Causal LM: LLMForCausal
# =======================

class LLMForCausal(PreTrainedModel, GenerationMixin):
    config_class = LLMConfig

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config
        self.model = LLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # weight tying: the diagram shows Linear+Softmax fed back to the tokenizer decoder
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        hidden_states, new_past = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)  # [bsz, seq, vocab]

        loss = None
        if labels is not None:
            # shift lm loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
            hidden_states=None,
            attentions=None,
        )

    # ---- GenerationMixin  ----
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        # With cache present, feed only the last token
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:].contiguous()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # For beam search: reorder the cache
        if past_key_values is None:
            return past_key_values
        new_past = []
        for layer_past in past_key_values:
            new_past.append(
                (
                    layer_past[0].index_select(0, beam_idx),
                    layer_past[1].index_select(0, beam_idx),
                )
            )
        return new_past
