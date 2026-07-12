# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter import get_weights_adapter
from .weights_adapter.default import _indexer_parameter_candidates

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _prepare_adapter(self, model_config: ModelConfig):
        local_model_path = self._prepare_weights(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        adapter.prepare_loading(local_model_path, model_config)
        return adapter

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter = self._prepare_adapter(model_config)
        adapter.restrict_to_model(model)
        model.load_weights(adapter.prepare_weights(model_config))

    @staticmethod
    def _split_indexer_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]:
        """Separate fused indexer weights from the main weight stream.

        The plugin pre-fuses indexer wk + weights_proj into
        wk_weights_proj.weight (dequantized, coalesced).  If these reach
        vLLM's DeepseekV2Model.load_weights, the stacked_params_mapping
        substring collision (``"wk" in "wk_weights_proj"``) doubles the
        suffix.  We intercept them here and load them directly into model
        params with shard_id=None ("already fused, copy directly"),
        bypassing stacked_params_mapping entirely.
        """
        indexer_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if ".indexer.wk_weights_proj.weight" in name:
                indexer_weights.append((name, tensor))
            else:
                other_weights.append((name, tensor))
        return indexer_weights, other_weights

    @staticmethod
    def _load_indexer_weights(
        model: nn.Module, indexer_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Load fused indexer shards directly without a concatenation buffer."""
        by_name: dict[str, list[torch.Tensor]] = {}
        for name, tensor in indexer_weights:
            by_name.setdefault(name, []).append(tensor)

        params_dict = dict(model.named_parameters())
        for name, tensors in by_name.items():
            candidates = _indexer_parameter_candidates(name)
            param_name = next(
                (candidate for candidate in candidates if candidate in params_dict),
                None,
            )
            if param_name is None:
                raise ValueError(
                    f"Required indexer weight {name} not found in model params; "
                    f"candidates={candidates}"
                )

            param = params_dict[param_name]
            expected_shape = list(param.data.shape)
            if not expected_shape:
                raise ValueError(f"Indexer weight {name} cannot target a scalar")
            expected_rows = sum(tensor.shape[0] for tensor in tensors)
            if expected_rows != expected_shape[0] or any(
                tensor.shape[1:] != param.data.shape[1:] for tensor in tensors
            ):
                raise ValueError(
                    f"Indexer weight {name} shape mismatch: "
                    f"param={param.data.shape} "
                    f"shards={[tensor.shape for tensor in tensors]}"
                )

            row_offset = 0
            for tensor in tensors:
                rows = tensor.shape[0]
                param.data.narrow(0, row_offset, rows).copy_(tensor)
                row_offset += rows
            logger.debug(
                "Stream-loaded indexer weight %s from %d shards shape=%s",
                name,
                len(tensors),
                param.data.shape,
            )

    @staticmethod
    def _load_kv_b_weights(
        model: nn.Module,
        name: str,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> None:
        """Load rank-local GLM K/V shards directly into fused kv_b_proj."""
        if first.ndim != 3 or second.ndim != 3:
            raise ValueError(
                f"GLM kv_b_proj shards must be 3D: {first.shape} and {second.shape}"
            )
        if first.shape[1] == second.shape[2]:
            k3, v3 = first, second
        elif second.shape[1] == first.shape[2]:
            k3, v3 = second, first
        else:
            raise ValueError(
                f"Cannot identify GLM kv_b_proj K/V shards: "
                f"{first.shape} and {second.shape}"
            )

        candidates = [name]
        if name.startswith("model."):
            candidates.append(name[len("model.") :])
        else:
            candidates.append("model." + name)
        for candidate in tuple(candidates):
            if ".self_attn." in candidate:
                candidates.append(
                    candidate.replace(
                        ".self_attn.",
                        ".mtp_block.self_attn.",
                        1,
                    )
                )

        params_dict = dict(model.named_parameters())
        param_name = next(
            (candidate for candidate in candidates if candidate in params_dict),
            None,
        )
        if param_name is None:
            raise ValueError(f"GLM kv_b_proj weight {name} not found in model params")

        param = params_dict[param_name]
        num_heads, kv_lora_rank, qk_nope_dim = k3.shape
        if v3.shape[0] != num_heads or v3.shape[2] != kv_lora_rank:
            raise ValueError(
                f"GLM kv_b_proj shard shape mismatch: K={k3.shape} V={v3.shape}"
            )
        value_head_dim = v3.shape[1]
        expected_shape = (
            num_heads * (qk_nope_dim + value_head_dim),
            kv_lora_rank,
        )
        if tuple(param.data.shape) != expected_shape:
            raise ValueError(
                f"GLM kv_b_proj parameter shape mismatch: "
                f"param={param.data.shape} expected={expected_shape}"
            )

        destination = param.data.view(
            num_heads,
            qk_nope_dim + value_head_dim,
            kv_lora_rank,
        )
        destination[:, :qk_nope_dim, :].copy_(k3.transpose(1, 2))
        destination[:, qk_nope_dim:, :].copy_(v3)
        logger.debug(
            "Stream-loaded rank-local GLM kv_b_proj %s shape=%s",
            name,
            param.data.shape,
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        vllm_config.model_config.hf_config = model_config.hf_config
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)

            adapter.restrict_to_model(model)

            pending_indexer: dict[str, torch.Tensor] = {}
            pending_kv_b: dict[str, torch.Tensor] = {}
            loaded_indexer_tensors = 0
            loaded_kv_b_tensors = 0
            peak_pending_indexers = 0
            peak_pending_kv_b = 0

            def _filtered(
                weights: Iterable[tuple[str, torch.Tensor]],
            ) -> Iterable[tuple[str, torch.Tensor]]:
                nonlocal pending_indexer, pending_kv_b
                nonlocal loaded_indexer_tensors, loaded_kv_b_tensors
                nonlocal peak_pending_indexers, peak_pending_kv_b
                for name, tensor in weights:
                    if ".indexer.wk_weights_proj" in name:
                        first = pending_indexer.pop(name, None)
                        if first is None:
                            pending_indexer[name] = tensor
                            peak_pending_indexers = max(
                                peak_pending_indexers,
                                len(pending_indexer),
                            )
                            continue
                        self._load_indexer_weights(
                            model,
                            [(name, first), (name, tensor)],
                        )
                        loaded_indexer_tensors += 2
                        continue

                    if ".self_attn.kv_b_proj.weight" in name:
                        first = pending_kv_b.pop(name, None)
                        if first is None:
                            pending_kv_b[name] = tensor
                            peak_pending_kv_b = max(
                                peak_pending_kv_b,
                                len(pending_kv_b),
                            )
                            continue
                        self._load_kv_b_weights(model, name, first, tensor)
                        loaded_kv_b_tensors += 2
                        continue

                    yield name, tensor

            model.load_weights(_filtered(adapter.prepare_weights(model_config)))

            if pending_indexer:
                remaining = list(pending_indexer.items())
                self._load_indexer_weights(model, remaining)
                loaded_indexer_tensors += len(remaining)
                pending_indexer.clear()

            if pending_kv_b:
                raise ValueError(
                    f"Missing GLM kv_b_proj shard pairs: {sorted(pending_kv_b)}"
                )

            logger.info(
                "Stream-loaded %d indexer tensors and %d rank-local "
                "kv_b_proj tensors; peak pending pairs indexer=%d kv_b=%d",
                loaded_indexer_tensors,
                loaded_kv_b_tensors,
                peak_pending_indexers,
                peak_pending_kv_b,
            )
            process_weights_after_loading(model, model_config, target_device)
            del pending_indexer
            del pending_kv_b
            gc.collect()
            torch.accelerator.empty_cache()
            logger.info(
                "Released transient GGUF loader objects and allocator cache "
                "before runtime warmup"
            )
        return model
