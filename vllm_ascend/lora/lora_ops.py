#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os

import torch


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )


# Kill switch for the masked-GEMM fast path; set to "0" to always use the
# AscendC sgmv/bgmv kernels (e.g. for A/B performance comparison).
_LORA_MATMUL_FASTPATH = os.environ.get("VLLM_ASCEND_LORA_MATMUL_FASTPATH", "1") != "0"

# PunicaWrapperNPU instances registered for the lora_linear custom op. The op
# schema only allows schema-friendly types, so the body receives the wrapper
# id and resolves the wrapper here; this lets the op body read the per-step
# routing flag at execution time instead of at trace time.
_LORA_WRAPPERS: dict[int, "PunicaWrapperNPU"] = {}
_LORA_WRAPPER_IDS = itertools.count()


@torch.library.custom_op("_vllm_ascend_lora::lora_linear", mutates_args={"y"})
def lora_linear(
    wrapper_id: int,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    scale: float,
    output_slices: list[int],
    packed_lora_a: torch.Tensor | None,
    packed_lora_b: torch.Tensor | None,
    add_inputs: bool,
) -> None:
    """
    Opaque entry point for one LoRA linear, dispatched through from
    PunicaWrapperNPU.add_lora_linear.

    vLLM compiles the model with torch.compile(fullgraph=True) and drops all
    guards (evaluate_guards=False), so dynamo traces the forward exactly once
    and never retraces: any Python-level branch in the traced code (such as
    the kernel-vs-matmul routing flag) is constant-folded to the value it
    held at trace time and frozen into every later execution, including
    replayed ACL graphs. Custom ops are opaque to dynamo -- the graph only
    records the call and the Python body executes on every invocation, so
    the body below reads the live routing flag. ACL graph capture runs the
    same body while the model runner forces the kernel path
    (NPUModelRunner._route_lora_kernels with force_kernel=True), which is
    what gets recorded into the replayed decode graphs, while prefill
    steps execute the body with the masked-GEMM path.
    """
    wrapper = _LORA_WRAPPERS[wrapper_id]
    # Route by batch type (flag maintained by the model runner, see
    # NPUModelRunner.set_active_loras): decode-sized batches run the
    # AscendC sgmv/bgmv vector kernels (they beat Cube-based matmul at
    # small batch sizes: 58ms vs 67ms per decode iteration measured on
    # qwen3.5-27B TP4), while prefill batches run the masked-GEMM fast
    # path, which requires the single-LoRA-slot layout. NOTE: vLLM v1
    # keeps LoRAMapping.is_prefill always True, so the batch type must
    # be routed by the model runner instead.
    use_kernel = (
        not wrapper._single_lora_slot
        or wrapper._kernel_only_small_batch
        or not _LORA_MATMUL_FASTPATH
    )
    if use_kernel:
        wrapper._lora_linear_kernel(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices)
    else:
        wrapper._lora_linear_matmul(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            packed_lora_a,
            packed_lora_b,
            add_inputs,
        )


@lora_linear.register_fake
def _lora_linear_fake(
    wrapper_id,
    y,
    x,
    lora_a_stacked,
    lora_b_stacked,
    scale,
    output_slices,
    packed_lora_a,
    packed_lora_b,
    add_inputs,
) -> None:
    # Only mutates y in place; there are no outputs to infer.
    return None
