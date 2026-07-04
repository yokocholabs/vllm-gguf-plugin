# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
        """Load pre-fused indexer weights directly into model params."""
        params_dict = dict(model.named_parameters())
        for name, tensor in indexer_weights:
            # Strip "model." prefix if present — params_dict keys may
            # or may not have it depending on model structure.
            param_name = name
            if param_name not in params_dict:
                # Try with and without "model." prefix
                if param_name.startswith("model."):
                    param_name = param_name[len("model."):]
                else:
                    param_name = "model." + param_name
            if param_name not in params_dict:
                logger.warning(
                    "Indexer weight %s not found in model params, skipping",
                    name,
                )
                continue
            param = params_dict[param_name]
            weight_loader = getattr(
                param, "weight_loader", None
            )
            if weight_loader is not None:
                # MergedColumnParallelLinear.weight_loader with
                # shard_id=None means "already fused, copy directly".
                weight_loader(param, tensor, None)
            else:
                # Fallback: direct copy
                param.data.copy_(tensor)
            logger.debug("Loaded indexer weight %s shape=%s", name, tensor.shape)

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

            # Stream weights through model.load_weights, collecting only
            # the small indexer tensors as a side effect.  Avoids
            # materializing the full weight set into a list (OOM on 128GB).
            indexer_weights: list[tuple[str, torch.Tensor]] = []

            def _filtered(
                weights: Iterable[tuple[str, torch.Tensor]],
            ) -> Iterable[tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    if ".indexer." in name:
                        indexer_weights.append((name, tensor))
                    else:
                        yield name, tensor

            model.load_weights(_filtered(adapter.prepare_weights(model_config)))

            if indexer_weights:
                logger.info(
                    "Loading %d indexer weight tensors directly "
                    "(bypassing stacked_params_mapping)",
                    len(indexer_weights),
                )
                self._load_indexer_weights(model, indexer_weights)
            process_weights_after_loading(model, model_config, target_device)
        return model
