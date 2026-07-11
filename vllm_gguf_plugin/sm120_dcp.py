# SPDX-License-Identifier: Apache-2.0
"""DCP (decode context parallel) support for FLASHINFER_MLA_SPARSE_SM120.

Upstream vLLM's SM120 sparse-MLA impl (flashinfer_mla_sparse_sm120.py)
converts the indexer's per-request logical top-k indices straight to global
physical indices and passes ``seq_lens=None`` ("every column in block_tables
is active") to the FlashInfer kernel. Under DCP each rank holds only an
interleaved ``1/dcp_world_size`` shard of the KV cache, so three things are
missing (all present in the SM100 impl, flashinfer_mla_sparse.py):

- logical indices must be filtered to rank-local entries and converted
  against the local block table (``triton_filter_and_convert_dcp_index``),
- the kernel must receive the per-token valid counts as ``seq_lens``
  (FlashInfer 0.6.14 sm120 sparse path: "active top-k lengths; if None,
  every column in block_tables is active"),
- the kernel must return LSE so vLLM's shared DCP reducer
  (cp_lse_ag_out_rs) can combine partial attention across ranks.
  ``lse_base_on_e=False``: the trtllm-gen kernel family returns base-2
  LSE — same setting as the SM100 impl; wrong base silently corrupts the
  cross-shard softmax denominator.

Rows whose top-k tokens all live on other ranks get out=0 / lse=-inf so
they contribute nothing to the combine (mirrors SM100).

Unlike upstream SM120, no ``out=`` buffer is passed: under the ag_rs DCP
comm backend the query arrives already all-gathered across the DCP group
(``num_heads * dcp_world_size`` heads), so any buffer sized from
``self.num_heads`` is wrong for DCP>1. FlashInfer allocates the output
from the query shape itself (same as the SM100 impl).

At DCP=1 the patched ``forward_mqa`` is behavior-identical to upstream:
same index conversion call, ``seq_lens=None``, no LSE request.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# vLLM's dictConfig only configures the "vllm" logger tree.
# vllm_gguf_plugin is a separate top-level logger and inherits root
# (WARNING by default). Force our level to INFO so our instrumentation
# is visible without requiring VLLM_LOGGING_LEVEL=DEBUG.
_gguf_log_level = os.environ.get("VLLM_GGUF_LOG_LEVEL", "INFO")
logger.setLevel(getattr(logging, _gguf_log_level.upper(), logging.INFO))
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.propagate = True

# Throttle: log first call, then every Nth
_fwd_mqa_call_count = 0
_fwd_mqa_log_interval = 50


def apply_sm120_dcp_patch() -> None:
    """Install the DCP-aware forward_mqa on FlashInferMLASparseSM120Impl."""
    try:
        from vllm.v1.attention.backend import AttentionLayer
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            FlashInferMLASparseImpl,
            FlashInferMLASparseMetadata,
        )
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
            _get_workspace_buffer,
        )
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
            triton_filter_and_convert_dcp_index,
        )
    except ImportError:
        logger.warning(
            "sm120 DCP patch NOT applied — vLLM module layout changed; "
            "DCP>1 must not be used with FLASHINFER_MLA_SPARSE_SM120",
            exc_info=True,
        )
        return

    if getattr(FlashInferMLASparseSM120Impl, "_gguf_sm120_dcp_patched", False):
        return

    def forward_mqa(
        self,
        q: "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        global _fwd_mqa_call_count
        _fwd_mqa_call_count += 1
        _should_log = (
            _fwd_mqa_call_count == 1
            or _fwd_mqa_call_count % _fwd_mqa_log_interval == 0
        )

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if _should_log:
            logger.info(
                "SM120 DCP forward_mqa call #%d: dcp_world_size=%d dcp_rank=%d "
                "num_actual_toks=%d q.shape=%s topk_indices.shape=%s "
                "num_heads=%s kv_lora_rank=%s need_lse=%s",
                _fwd_mqa_call_count,
                self.dcp_world_size,
                self.dcp_rank,
                num_actual_toks,
                tuple(q.shape),
                tuple(topk_indices.shape),
                getattr(self, 'num_heads', '?'),
                getattr(self, 'kv_lora_rank', '?'),
                getattr(self, 'need_to_return_lse_for_decode', '?'),
            )

        if self.dcp_world_size > 1:
            # Filter the logical top-k indices down to the entries this DCP
            # rank owns and convert them against the local block table;
            # seq_lens carries the per-token count of surviving entries.
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(
                    attn_metadata.cp_kv_cache_interleave_size
                ),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )
            seq_lens = None

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o = kernel_out
            lse = None

        out = o.view(-1, o.shape[-2], o.shape[-1])
        if lse is not None:
            lse = FlashInferMLASparseImpl._normalize_lse(
                lse, out.shape[0], out.shape[1]
            )
            empty_rows = (topk_indices_physical == -1).all(dim=-1)
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        else:
            empty_rows = None

        if _should_log:
            n_empty = int(empty_rows.sum()) if empty_rows is not None else -1
            logger.info(
                "SM120 DCP forward_mqa call #%d: kernel_out type=%s "
                "out.shape=%s lse=%s empty_rows=%d seq_lens=%s "
                "topk_indices_physical.shape=%s",
                _fwd_mqa_call_count,
                type(kernel_out).__name__,
                tuple(out.shape),
                tuple(lse.shape) if lse is not None else None,
                n_empty,
                tuple(seq_lens[:4]) if seq_lens is not None else None,
                tuple(topk_indices_physical.shape),
            )
        return out, lse

    FlashInferMLASparseSM120Impl.forward_mqa = forward_mqa
    # Must be set on the class BEFORE instances exist:
    # AttentionImplBase.__new__ computes need_to_return_lse_for_decode from
    # can_return_lse_for_decode, and the DCP combine branches on
    # lse_base_on_e (trtllm-gen returns base-2 LSE).
    FlashInferMLASparseSM120Impl.can_return_lse_for_decode = True
    FlashInferMLASparseSM120Impl.lse_base_on_e = False
    FlashInferMLASparseSM120Impl._gguf_sm120_dcp_patched = True
    logger.info(
        "Applied FLASHINFER_MLA_SPARSE_SM120 DCP patch "
        "(forward_mqa index filtering + LSE for cross-rank combine)"
    )
