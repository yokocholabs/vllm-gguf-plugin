# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM
from vllm.logger import init_logger

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class GGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Default adapter for GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def build_name_map(
        self, model_config: ModelConfig
    ) -> tuple[dict[str, str], list[str]]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []
        force_unquantized_modules: list[str] = []

        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type == "glm_moe_dsa":
            model_type = "glm-dsa"
            first_moe_layer = getattr(config, "first_k_dense_replace", 0)
            for idx in range(first_moe_layer, config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp.weight"] = (
                    f"model.layers.{idx}.mlp.gate.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_shexp.weight"] = (
                    f"model.layers.{idx}.mlp.shared_experts.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_shexp.weight"] = (
                    f"model.layers.{idx}.mlp.shared_experts.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_shexp.weight"] = (
                    f"model.layers.{idx}.mlp.shared_experts.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )

            # GLM-5.2 DSA Indexer: vLLM fuses wk + weights_proj into
            # MergedColumnParallelLinear (wk_weights_proj) with
            # quant_config=None, disable_tp=True.  The GGUF stores them
            # as separate tensors: blk.{idx}.indexer.attn_k (Q8_0, the "wk"
            # shard) and blk.{idx}.indexer.proj (F32, the "weights_proj"
            # shard).  Map to the UNFUSED HF shard names so vLLM's
            # stacked_params_mapping handles the fusion natively:
            #   ("wk_weights_proj", "wk", 0)
            #   ("wk_weights_proj", "weights_proj", 1)
            # Dequantize attn_k Q8_0 → bf16 via force_unquantized_modules.
            # weights_proj is already F32, no dequant needed.
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.indexer.attn_k.weight"] = (
                    f"model.layers.{idx}.self_attn.indexer.wk.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer.proj.weight"] = (
                    f"model.layers.{idx}.self_attn.indexer.weights_proj.weight"
                )
            # Dequant Q8_0 wk → bf16 (iterator path)
            force_unquantized_modules.append("indexer.wk")
            # Force UnquantizedLinearMethod for the fused layer
            force_unquantized_modules.append("indexer.wk_weights_proj")

            # GLM-5.2 DSA MLA kv_b_proj: vLLM has a single 2D
            # ColumnParallelLinear kv_b_proj [num_heads*(qk_nope+v_head),
            # kv_lora_rank] with quant_config.  The GGUF stores K and V as
            # SEPARATE 3D per-head tensors:
            #   attn_k_b: [n_head, kv_lora_rank, qk_nope_head_dim] Q8_0
            #   attn_v_b: [n_head, v_head_dim, kv_lora_rank]      Q8_0
            # These have different shapes (K: in=192, V: in=512) so cannot
            # be loaded as quantized shards.  Dequantize to bf16, reshape
            # to 2D, and concatenate along dim=0 in prepare_weights.
            # Memory: bf16 not fp32, streaming (1 layer at a time).
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.attn_k_b.weight"] = (
                    f"model.layers.{idx}.self_attn.kv_b_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_v_b.weight"] = (
                    f"model.layers.{idx}.self_attn.kv_b_proj.weight"
                )
            # Dequant Q8_0 → bf16 (iterator path)
            force_unquantized_modules.append("self_attn.kv_b_proj")

        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "olmoe":
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, config.vision_config.num_hidden_layers
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            return gguf_name + "." + suffix

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                # Don't overwrite explicit mappings (e.g. GLM-5.2 indexer
                # fused wk_weights_proj that differ from gguf lib defaults).
                if gguf_name_with_suffix not in gguf_to_hf_name_map:
                    gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map, force_unquantized_modules

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            yield hf_name, self.transform_weight(hf_name, weight)

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def update_tie_word_embeddings(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        if "lm_head.weight" not in gguf_to_hf_name_map.values():
            return

        all_extra_names = []
        for gguf_file in self._get_all_gguf_files(model_path):
            all_extra_names.extend(
                get_gguf_extra_tensor_names(gguf_file, gguf_to_hf_name_map)
            )
        hf_config.update({"tie_word_embeddings": "lm_head.weight" in all_extra_names})

    def get_weight_type_map(
        self,
        model_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = {}
        for gguf_file in self._get_all_gguf_files(model_path):
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        return weight_type_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        return [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_to_hf_name_map, force_unquantized = self.build_name_map(model_config)
        self.update_tie_word_embeddings(
            model_path, model_config.hf_config, gguf_to_hf_name_map
        )
        weight_type_map = self.get_weight_type_map(model_path, gguf_to_hf_name_map)
        unquantized_modules = self.get_unquantized_modules(weight_type_map)
        unquantized_modules.extend(force_unquantized)
        self.load_spec = GGUFLoadSpec(
            weights_source=self._get_all_gguf_files(model_path),
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=unquantized_modules,
        )
        logger.debug(
            "GGUFLoadSpec: unquantized_modules=%s",
            unquantized_modules,
        )
        # Log indexer-specific mappings for debugging
        indexer_maps = {
            k: v for k, v in gguf_to_hf_name_map.items()
            if "indexer" in k or "indexer" in v
        }
        if indexer_maps:
            logger.debug("GGUF indexer name mappings: %s", indexer_maps)
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
            self.load_spec.unquantized_modules,
        )
        mapped = self.map_weights(weights)
        # Only kv_b_proj needs pairing (attn_k_b + attn_v_b map to the
        # same name but are NOT adjacent in GGUF ordering).  Everything
        # else streams directly to avoid accumulating the full model
        # in RAM.
        kv_b_held: tuple[str, torch.Tensor] | None = None
        for name, tensor in mapped:
            if "indexer" in name or "kv_b_proj" in name:
                logger.debug(
                    "GGUF weight: %s shape=%s dtype=%s",
                    name, tensor.shape, tensor.dtype,
                )
            if "kv_b_proj.weight" in name:
                if kv_b_held is not None:
                    # Second shard (V) arrived — coalesce with held K.
                    a = kv_b_held[1]
                    b = tensor
                    # K: [n_head, kv_lora, qk_nope] -> transpose(1,2) ->
                    #    [n_head, qk_nope, kv_lora] -> [n_head*qk_nope, kv_lora]
                    # V: [n_head, v_head, kv_lora] -> [n_head*v_head, kv_lora]
                    if a.dim() == 3 and b.dim() == 3:
                        if a.shape[1] == b.shape[2]:
                            # a=K, b=V
                            a2 = a.transpose(1, 2).reshape(
                                a.shape[0] * a.shape[2], a.shape[1])
                            b2 = b.reshape(
                                b.shape[0] * b.shape[1], b.shape[2])
                        elif a.shape[2] == b.shape[1]:
                            # a=V, b=K
                            a2 = a.reshape(
                                a.shape[0] * a.shape[1], a.shape[2])
                            b2 = b.transpose(1, 2).reshape(
                                b.shape[0] * b.shape[2], b.shape[1])
                        else:
                            a2 = a.reshape(
                                a.shape[0] * a.shape[1], a.shape[2])
                            b2 = b.reshape(
                                b.shape[0] * b.shape[1], b.shape[2])
                    else:
                        a2, b2 = a, b
                    logger.debug(
                        "Coalescing %s: cat([%s, %s], dim=0)",
                        name, a2.shape, b2.shape,
                    )
                    yield name, torch.cat([a2, b2], dim=0)
                    kv_b_held = None
                else:
                    # First shard (K) — hold until V arrives.
                    kv_b_held = (name, tensor)
            else:
                # All other weights stream directly.
                yield name, tensor
        # Flush any remaining held kv_b_proj (shouldn't happen).
        if kv_b_held is not None:
            yield kv_b_held[0], kv_b_held[1]
