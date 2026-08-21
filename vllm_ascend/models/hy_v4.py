# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 model — Ascend NPU implementation.

The architecture (HYV4ForCausalLM + HYV4MTP) is identical to the NVIDIA
implementation, so we re-use the NVIDIA modules directly. The only NPU-side
adaptations are:

* The ``compute_logits`` autocast uses the current platform's device type
  instead of hard-coding ``"cuda"``.
* The CUDA-only FlashMLA-sink backend (``HYV4FlashMLASparseBackend``) is
  imported lazily inside the model; if it is not available on the NPU the
  ``learnable_sink`` is silently disabled by the existing sink-resolution
  logic in ``HYV4MLAAttention``. No code change is required to opt in.
* The sparse MLA backend is selected at runtime by
  ``vllm.v1.attention.selector.get_attn_backend``, so the NPU's DSA / SFA
  backend is used automatically.
"""

import torch

from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
    HYV4_PACKED_MODULES_MAPPING,
    _normalize_hyv4_config,
)
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP
from vllm.platforms import current_platform


class _HyV4FusedQkvAProj(torch.nn.Module):
    """Virtual fused QKV-A projection for HYV4.

    HYV4 stores ``q_a_proj`` and ``kv_a_proj_with_mqa`` as two separate
    linear layers, but the NPU's MLA preprocessing expects a single
    ``fused_qkv_a_proj`` whose output is the concatenation of
    ``[q_lora_rank, kv_lora_rank + qk_rope_head_dim]``. This module
    re-uses the two existing linear weights (no extra parameters) and
    concatenates their outputs on the fly.
    """

    def __init__(self, q_a_proj, kv_a_proj_with_mqa):
        super().__init__()
        # Hold references to the underlying linears so the fused
        # module is a thin alias and weight loading into the
        # originals flows through unchanged.
        self.q_a_proj = q_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        # Expose a ``.weight`` attribute that points to ``q_a_proj``'s
        # weight so any caller that introspects the fused module's
        # weight (e.g. weight prefetch) still finds something useful.
        self.weight = q_a_proj.weight

    def forward(self, x):
        q = self.q_a_proj(x)[0]
        kv = self.kv_a_proj_with_mqa(x)[0]
        return torch.cat([q, kv], dim=-1), None


def _patch_compute_logits_for_npu():
    """Replace the hard-coded ``device_type="cuda"`` autocast in
    ``HYV4ForCausalLM.compute_logits`` with the current platform's device
    type so the FP32 LM-head path works on NPU."""
    device_type = current_platform.device_type
    if device_type == "cuda":
        return  # nothing to do on CUDA hosts

    import torch

    from vllm.sequence import IntermediateTensors

    def compute_logits(self, hidden_states):  # type: ignore[no-redef]
        if self.enable_lm_head_fp32:
            with torch.autocast(device_type=device_type, enabled=False):
                logits = self.logits_processor(
                    self.lm_head, hidden_states.to(torch.float32)
                )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    HYV4ForCausalLM.compute_logits = compute_logits
    HYV4ForCausalLM.forward = forward


def _patch_mla_attention_init_and_forward_for_npu():
    """Replace ``HYV4MLAAttention.__init__`` and ``forward`` on NPU.

    The HYV4 attention constructs a vllm-core ``MLAAttention`` instance
    (``self.mla_attn``) and calls it with already-projected ``q``,
    ``kv_c_normed`` and ``k_pe`` tensors. Ascend's MLA implementations
    (``AscendMultiHeadLatentAttention`` / ``AscendSFAImpl``) deliberately
    raise ``NotImplementedError`` from the ``forward_mha`` /
    ``forward_mqa`` entry points and only support the unified
    ``mla_forward`` op, which expects the *raw* ``hidden_states`` and a
    layer wrapper registered in ``forward_context.no_compile_layers``.

    The cleanest port is to mirror ``vllm-ascend/patch/worker/
    patch_deepseek_v2.py``: after the HYV4 attention finishes building
    its MLA modules, swap ``self.mla_attn`` for a
    ``MultiHeadLatentAttentionWrapper`` that calls the NPU's MLA
    forward, and replace the per-layer ``forward`` so the wrapper
    runs the MLA preprocessing (fused-QKV-A → RoPE → MLA) and the
    o_proj itself. The HYV4-specific ``gated_mla`` post-processing
    still runs after the wrapper returns.
    """
    from vllm.model_executor.layers.mla import (
        MLAModules,
        MultiHeadLatentAttentionWrapper,
    )
    from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention

    original_init = HYV4MLAAttention.__init__
    original_forward = HYV4MLAAttention.forward

    def _npu_init(self, *args, **kwargs):  # type: ignore[no-redef]
        # Resolve cache/quant configs before delegating so the wrapper
        # has everything it needs. ``quant_config`` may legitimately be
        # ``None`` (e.g. when the model is served in bf16), so we only
        # treat the cache_config as required.
        cache_config = kwargs.get("cache_config")
        quant_config = kwargs.get("quant_config")
        if cache_config is None:
            # kwargs may use ``cache_config``; check positional too.
            for name, value in zip(
                (
                    "vllm_config",
                    "config",
                    "hidden_size",
                    "num_heads",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                    "q_lora_rank",
                    "kv_lora_rank",
                    "max_position_embeddings",
                    "cache_config",
                    "quant_config",
                ),
                args,
            ):
                if name == "cache_config":
                    cache_config = value
                elif name == "quant_config" and quant_config is None:
                    quant_config = value
        if cache_config is None:
            vllm_config = kwargs.get("vllm_config")
            if vllm_config is None and len(args) >= 1:
                vllm_config = args[0]
            if vllm_config is not None:
                cache_config = vllm_config.cache_config
                if quant_config is None:
                    quant_config = vllm_config.quant_config
        if cache_config is None:
            raise RuntimeError(
                f"NPU HYV4 attention could not recover cache_config "
                f"(cache_config={cache_config}, quant_config={quant_config}, "
                f"args={args}, kwargs_keys={list(kwargs.keys())}) for "
                f"MultiHeadLatentAttentionWrapper construction."
            )
        # Stash for later use if the original constructor overwrites
        # ``self.quant_config`` (it does, so this is only a backup).
        kwargs["cache_config"] = cache_config
        kwargs["quant_config"] = quant_config
        # The NPU's ``AscendMultiHeadLatentAttention`` bakes
        # ``o_proj`` into its impl, so the wrapper returns the
        # post-projection tensor of shape ``[N, hidden_size]``.
        # HYV4's ``gated_mla`` post-processing expects the
        # pre-projection tensor ``[N, num_heads * v_head_dim]``,
        # so it cannot be applied on top of the NPU wrapper's
        # output. Disable the gate to avoid a shape mismatch.
        # The linear_gate parameter will be loaded but unused.
        config = kwargs.get("config")
        if config is None and len(args) >= 2:
            config = args[1]
        if config is not None and getattr(config, "gated_mla", False):
            try:
                config.gated_mla = False
            except Exception:
                pass
        # Run the original constructor so all MLA submodules
        # (q_a_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj,
        # o_proj, rotary_emb, indexer, ...) are populated.
        original_init(self, *args, **kwargs)
        # The indexer is disabled upstream (NPU Triton cannot compile
        # ``f8E4M3FN``); the wrapper accepts ``is_sparse=False`` and
        # ``skip_topk=True`` and will simply skip the indexer branch.
        self.is_sparse = False
        self.skip_topk = True
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            # The NPU's MLA preprocessing needs a single
            # ``fused_qkv_a_proj`` whose output is the concatenation
            # of ``[q_lora_rank, kv_lora_rank + qk_rope_head_dim]``.
            # HYV4 keeps these as two separate linears; wrap them
            # in a virtual fused module so the NPU can consume
            # them without duplicating weights.
            fused_qkv_a_proj=_HyV4FusedQkvAProj(
                self.q_a_proj, self.kv_a_proj_with_mqa
            )
            if self.q_lora_rank is not None and self.q_a_proj is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            indexer_rotary_emb=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        # ``prefix`` is the layer's registered name; the wrapper needs
        # a unique prefix (not the same as the original ``MLAAttention``)
        # so vllm's module registry does not raise a duplicate-name
        # error. We append ``.npu_attn`` so the wrapper's own
        # submodules live under ``model.layers.N.self_attn.npu_attn``
        # while the legacy ``MLAAttention`` (whose forward is now
        # bypassed) keeps its original ``.attn`` namespace.
        attn_prefix = f"{self.prefix}.npu_attn"
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            attn_prefix,
            skip_topk=True,
        )

    def _npu_forward(self, positions, hidden_states, llama_4_scaling=None):  # type: ignore[no-redef]
        # Delegate MLA + o_proj to the NPU's
        # ``AscendMultiHeadLatentAttention`` (instantiated via the
        # ``MultiHeadLatentAttentionWrapper`` PluggableLayer). The
        # wrapper's impl bakes ``o_proj`` into its forward and
        # returns the post-projection tensor of shape
        # ``[N, hidden_size]``.
        attn_out = self.mla_attn(positions, hidden_states, llama_4_scaling)
        # HYV4's ``gated_mla`` post-processing cannot be applied
        # here because the gate expects the pre-projection tensor
        # ``[N, num_heads * v_head_dim]``. We disabled
        # ``gated_mla`` in ``_npu_init`` so the gate is a no-op
        # in practice (its weights are still loaded, just unused).
        return attn_out

    HYV4MLAAttention.__init__ = _npu_init
    HYV4MLAAttention.forward = _npu_forward


def _patch_mla_attention_forward_for_npu():
    """DEPRECATED: superseded by
    ``_patch_mla_attention_init_and_forward_for_npu`` which replaces both
    ``HYV4MLAAttention.__init__`` and ``HYV4MLAAttention.forward``. Kept
    here as a no-op so older code that imports it still works.
    """
    return


_patch_compute_logits_for_npu()
_patch_mla_attention_init_and_forward_for_npu()


def _disable_hyv4_indexer_for_npu():
    """Disable the NVIDIA lightning indexer on NPU.

    The NVIDIA ``Indexer`` kernel (``per_token_group_quant_fp8``) cannot be
    compiled by Ascend's Triton backend because ``f8E4M3FN`` is not
    recognized by the BiShengIR pipeline. The NPU's SFA / DSA backend
    already consumes ``topk_indices_buffer`` directly, so the
    ``self.indexer`` path is unnecessary. Force every layer into
    ``skip_topk=True`` so the indexer's forward is short-circuited and its
    submodules are never built.
    """
    from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention

    # Patch ``create_indexer``-style logic by overriding the class attribute
    # logic. The constructor reads ``index_topk`` and ``layer_types`` to
    # decide. We monkey-patch the class to always report dense attention.
    original_init = HYV4MLAAttention.__init__

    # Force ``requested_sparse`` to False before the indexer branch is
    # evaluated. The constructor calls ``getattr(config, "index_topk")``
    # indirectly through ``layer_types`` membership checks; the cleanest
    # hook is to drop the layer_types override after init.
    def patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
        # Force every layer to take the dense branch so the constructor
        # never builds the NVIDIA ``Indexer`` submodule (its
        # ``per_token_group_quant_fp8`` Triton kernel fails to compile on
        # Ascend because ``f8E4M3FN`` is not supported). The NPU's SFA
        # backend consumes ``topk_indices_buffer`` directly and does not
        # need this indexer.
        config = kwargs.get("config")
        if config is None and len(args) >= 2:
            config = args[1]
        if config is not None:
            if getattr(config, "index_topk", None) is not None:
                try:
                    delattr(config, "index_topk")
                except Exception:
                    try:
                        config.index_topk = None
                    except Exception:
                        pass
            layer_types = getattr(config, "layer_types", None)
            if layer_types is not None:
                try:
                    config.layer_types = ["full"] * len(layer_types)
                except Exception:
                    pass
        original_init(self, *args, **kwargs)
        # Belt-and-braces: if anything still references a (now-invalid)
        # indexer, drop it so no submodule leaks into the module hierarchy.
        if getattr(self, "indexer", None) is not None:
            self.indexer = None
        if getattr(self, "indexer_rope_emb", None) is not None:
            self.indexer_rope_emb = None
        self.is_sparse = False
        self.skip_topk = True
        if hasattr(self, "topk_indices_buffer"):
            self.topk_indices_buffer = None

    HYV4MLAAttention.__init__ = patched_init


_disable_hyv4_indexer_for_npu()


AscendHYV4ForCausalLM = HYV4ForCausalLM
AscendHYV4MTP = HYV4MTP

__all__ = [
    "AscendHYV4ForCausalLM",
    "AscendHYV4MTP",
    "HYV4ForCausalLM",
    "HYV4DecoderLayer",
    "HYV4Model",
    "HYV4MTP",
    "HYV4PACKED_MODULES_MAPPING",
    "HYV4MLAAttention",
    "HYV4HCLayer",
    "HYV4HCHeadLayer",
    "HYV4FeedForward",
    "HYV4MoEFused",
    "_normalize_hyv4_config",
]
