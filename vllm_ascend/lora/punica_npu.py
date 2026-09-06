# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.lora_ops import _LORA_WRAPPER_IDS, _LORA_WRAPPERS, lora_linear
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        refresh_all_lora_classes()
        self._max_num_batched_tokens = max_num_batched_tokens
        # Persistent shrink buffers keyed by (n_slices, rank), see _get_shrink_buffer
        self._lora_shrink_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self.lora_config = kwargs.get("lora_config")
        ascend_device_type = get_ascend_device_type()
        if not get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS) or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            from vllm.lora.ops.torch_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink
        self._single_lora_slot = (
            ascend_device_type == AscendDeviceType.A2
            and self.lora_config is not None
            and self.lora_config.max_loras == 1
            and not self.lora_config.fully_sharded_loras
        )
        self._single_lora_mask = None
        # LoRA kernel routing flag, updated per step by the model runner
        # (True -> AscendC kernels, False -> masked-GEMM fast path)
        self._kernel_only_small_batch = True
        # Register for the opaque lora_linear custom op (see _get_lora_linear_op)
        self._wrapper_id = next(_LORA_WRAPPER_IDS)
        _LORA_WRAPPERS[self._wrapper_id] = self
        if self._single_lora_slot:
            assert self.lora_config is not None
            lora_dtype = self.lora_config.lora_dtype
            if not isinstance(lora_dtype, torch.dtype):
                raise ValueError(f"LoRA dtype must be resolved before creating the Punica wrapper, got {lora_dtype!r}")
            self._single_lora_mask = torch.empty(
                (max_num_batched_tokens, 1),
                dtype=lora_dtype,
                device=device,
            )

    def _update_base_metadata(
        self,
        mapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
    ) -> None:
        super()._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        if self._single_lora_mask is None:
            return

        token_count = self.indices_len[0]
        assert token_count is not None
        self._single_lora_mask[:token_count].copy_(self._token_lora_indices[:token_count].eq(0).unsqueeze(1))

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        self.no_lora = not any(lora_id > 0 for lora_id in mapping.index_mapping)

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        self.bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x), scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        self.bgmv_expand(x, w_t_all, y, self._get_token_lora_indices(x), add_inputs)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        self.bgmv_expand_slice(
            x,
            w_t_all,
            y,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.Tensor:
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = self._expand_slice_prefill if self.is_prefill else self._expand_slice_decode
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = self._shrink_prefill if self.is_prefill else self._shrink_decode
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx], scale)

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (Tuple[int, ...]): Every slice's size
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0) @ lora_a_stacked[
                    indices[i], layer_idx, :, :] @ lora_b_stacked[
                    indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is not None:
            # Explicit shrink buffer: no current caller passes one; keep the
            # plain kernel path for compatibility.
            self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
            self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)
            return

        # Dispatch through the opaque custom op so that the kernel-vs-matmul
        # routing flag is read when the op executes (per step / per capture),
        # not frozen into the compiled artifact at dynamo trace time.
        # NOTE: everything in this method is traced inline by dynamo, so only
        # dynamo-safe code (op calls) may live here.
        lora_linear(
            self._wrapper_id,
            y,
            x,
            list(lora_a_stacked),
            list(lora_b_stacked),
            scale,
            list(output_slices),
            kwargs.get("packed_lora_a"),
            kwargs.get("packed_lora_b"),
            kwargs.get("add_inputs", True),
        )

    def _lora_linear_kernel(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
    ) -> None:
        """AscendC sgmv/bgmv kernel path (decode-sized batches)."""
        r = lora_b_stacked[0].size(-1)
        buffer = self._get_shrink_buffer(len(output_slices), x.size(0), r, x.device)
        self.add_shrink(buffer, x, lora_a_stacked, scale)
        self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True)

    def _get_shrink_buffer(
        self,
        n_slices: int,
        num_tokens: int,
        rank: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """
        Persistent fp32 shrink buffer shared by all layers with the same
        (n_slices, rank), replacing the per-layer torch.zeros allocation.
        No re-zeroing is needed: the bgmv/sgmv kernels skip tokens whose
        lora index is negative, so rows of no-LoRA tokens are never read
        back by the expand kernel.
        """
        key = (n_slices, rank)
        buf = self._lora_shrink_buffers.get(key)
        if buf is None or buf.shape[1] < num_tokens:
            alloc_rows = max(num_tokens, self._max_num_batched_tokens)
            buf = torch.zeros(
                (n_slices, alloc_rows, rank),
                dtype=torch.float32,
                device=device,
            )
            self._lora_shrink_buffers[key] = buf
        return tuple(buf[i][:num_tokens] for i in range(n_slices))

    def _lora_linear_matmul(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
        packed_lora_a: torch.Tensor | None,
        packed_lora_b: torch.Tensor | None,
        add_inputs: bool,
    ) -> None:
        """Masked-GEMM fast path for prefill batches (single-LoRA slot)."""
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        assert self._single_lora_mask is not None
        adapter_mask = self._single_lora_mask[: x.size(0)]

        # Resolve a single GEMM operand for the expand step so the whole
        # expand is one fused addmm (matmul + scale + residual add):
        # - 1 slice: a transposed view of B, no packing needed
        # - >1 slices: the pre-built block-diagonal packed B from the layer
        if len(lora_b_stacked) == 1:
            b_mat = lora_b_stacked[0][0, 0, : output_slices[0]].transpose(0, 1)
        elif (
            packed_lora_b is not None
            and packed_lora_a is not None
            and sum(output_slices) == y.shape[1]
        ):
            b_mat = packed_lora_b[0, 0]
        else:
            b_mat = None

        if b_mat is not None:
            if packed_lora_a is not None and len(lora_a_stacked) > 1:
                shrink = torch.matmul(x, packed_lora_a[0, 0].transpose(0, 1))
            else:
                shrink = torch.matmul(x, lora_a_stacked[0][0, 0].transpose(0, 1))
            shrink.mul_(adapter_mask)
            # Fold scale into the tiny shrink tensor ([tokens, rank]) so the
            # big output update needs no alpha (add_ with alpha is measurably
            # slower on NPU).
            if scale != 1.0:
                shrink.mul_(scale)
            # NOTE: y.addmm_(shrink, b_mat, beta=1) looks like the cheapest
            # form but CANN's read-modify-write GEMM epilogue is pathologically
            # slow for small-K GEMMs (2.5ms vs 0.66ms for the split form at
            # 13.6k tokens on A2). Write the LoRA delta with a pure GEMM
            # (fast beta=0 path) and accumulate with an elementwise add.
            if shrink.dtype == y.dtype:
                if add_inputs:
                    y.add_(torch.matmul(shrink, b_mat))
                else:
                    torch.matmul(shrink, b_mat, out=y)
            else:
                delta = torch.matmul(shrink, b_mat)
                PunicaWrapperNPU._update_single_lora_output(y, delta, 1.0, add_inputs)
            return

        # Fallback: per-slice shrink/expand for layers without packed weights.
        deltas = []
        if packed_lora_a is not None and len(lora_a_stacked) > 1:
            rank = lora_b_stacked[0].size(-1)
            shrink = torch.matmul(x, packed_lora_a[0, 0].transpose(0, 1))
            shrink.mul_(adapter_mask)
            shrink_slices = tuple(
                shrink.narrow(1, slice_index * rank, rank) for slice_index in range(len(lora_a_stacked))
            )
        else:
            shrink_slices = tuple(
                torch.matmul(x, lora_a[0, 0].transpose(0, 1)).mul_(adapter_mask) for lora_a in lora_a_stacked
            )

        for shrink, lora_b, output_size in zip(shrink_slices, lora_b_stacked, output_slices, strict=True):
            b_weight = lora_b[0, 0, :output_size]
            deltas.append(torch.matmul(shrink, b_weight.transpose(0, 1)))

        if len(deltas) > 1 and sum(output_slices) == y.shape[1]:
            delta = torch.cat(deltas, dim=1)
            PunicaWrapperNPU._update_single_lora_output(y, delta, scale, add_inputs)
            return

        offset = 0
        for delta, output_size in zip(deltas, output_slices, strict=True):
            y_slice = y.narrow(1, offset, output_size)
            PunicaWrapperNPU._update_single_lora_output(y_slice, delta, scale, add_inputs)
            offset += output_size

    @staticmethod
    def _update_single_lora_output(
        y: torch.Tensor,
        delta: torch.Tensor,
        scale: float,
        add_inputs: bool,
    ) -> None:
        if add_inputs:
            if scale == 1.0:
                y.add_(delta)
            else:
                y.add_(delta, alpha=scale)
        elif scale == 1.0:
            y.copy_(delta)
        else:
            torch.mul(delta, scale, out=y)

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        *,
        topk_weights: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None = None,
        max_lora_rank: int = 0,
        top_k_num: int = 1,
        shrink_config=None,
        expand_config=None,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ) -> None:
        """
        Ascend-native fused MoE LoRA (v2): static-shape per-row gather via the
        same bgmv_shrink/bgmv_expand AscendC kernels (csrc/kernels/bgmv_*.cpp)
        used by the dense Linear LoRA layers, instead of grouping rows by a
        data-dependent ``torch.unique`` over active LoRA ids. The previous
        ``torch.unique``/``nonzero`` version produced output whose *shape*
        depended on tensor values, which ACL Graph capture cannot record
        (it failed with an `aclnnUnique2` error as soon as `enforce_eager`
        was turned off) -- every tensor below has a shape that depends only
        on input shapes, never on values, so this stays graph-capturable.

        Rows are already one-token-per-row (top_k_num=1). Each row needs the
        LoRA slot for (lora_id, expert_id), so we fold both into a single
        gather index into a ``[max_loras * num_experts, ...]`` view of the
        existing per-(lora, expert) weight stacks:
            combined_idx[row] = lora_id[row] * num_experts + expert_id[row]
        or -1 when the row has no active adapter, mirroring the -1 sentinel
        ``PunicaWrapperBase.token_lora_indices`` already uses. bgmv_shrink/
        bgmv_expand skip any row whose index is negative (leaving the
        zero-initialized shrink buffer / unmodified ``y`` in place), so
        inactive rows get a zero delta for free -- no Python-level branching
        needed.
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config, fully_sharded
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if token_lora_mapping is None:
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        expert_idx = expert_ids.view(-1).to(torch.long)
        num_experts = lora_a_stacked[0].shape[1]

        lora_idx_safe = token_lora_mapping.clamp(min=0)
        enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
        combined_idx = torch.where(
            enabled,
            lora_idx_safe * num_experts + expert_idx,
            torch.full_like(token_lora_mapping, -1),
        ).contiguous()

        # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32 (its X_T),
        # so the shrink buffer is fp32.
        rank = lora_a_stacked[0].shape[-2]
        shrink_out = torch.zeros((x2d.shape[0], rank), dtype=torch.float32, device=x2d.device)

        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts, rank, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            out_size = b.shape[-2]
            a_flat = a.view(-1, rank, a.shape[-1])
            b_flat = b.view(-1, out_size, rank)

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, combined_idx, cur_offset, out_size, add_inputs=True)
            cur_offset += out_size

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
