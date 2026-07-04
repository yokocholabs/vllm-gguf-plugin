# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import os
from collections.abc import Generator
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    allow_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]

    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    local_files: list[str] = []
    for pattern in allow_patterns:
        local_files.extend(glob.glob(os.path.join(folder, pattern)))

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


def resolve_local_gguf(local_dir: str, quant_type: str) -> str:
    """Find a GGUF file matching *quant_type* in a local directory."""
    import glob as glob_mod

    patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob_mod.glob(os.path.join(local_dir, pat)))
    if not matches:
        raise ValueError(
            f"No GGUF file matching quant_type '{quant_type}' found in {local_dir}"
        )
    matches.sort(key=lambda x: (x.count("-"), x))
    return matches[0]


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_weight_type_map(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    reader = gguf.GGUFReader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path,
    gguf_to_hf_name_map: dict[str, str] | None,
    unquantized_modules: list[str] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi(
        [gguf_file], gguf_to_hf_name_map, unquantized_modules
    )


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str],
    gguf_to_hf_name_map: dict[str, str] | None = None,
    unquantized_modules: list[str] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` for all tensors in *gguf_files*.

    When *gguf_to_hf_name_map* is ``None``, raw GGUF tensor names are used
    directly (useful when a caller will apply a :class:`WeightsMapper``
    afterwards).  When a mapping is provided, tensors not present in the map
    are skipped and names are translated accordingly.

    When *unquantized_modules* is provided, quantized tensors whose mapped
    name contains one of the listed module prefixes are dequantized on the
    fly via ``ggml_dequantize`` and yielded as ``.weight`` instead of
    ``.qweight`` / ``.qweight_type``.
    """
    _QUANT_TYPES = ("F32", "BF16", "F16")

    logger.debug(
        "gguf_quant_weights_iterator_multi: files=%s unquantized_modules=%s",
        [os.path.basename(f) for f in gguf_files], unquantized_modules,
    )

    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        for tensor in reader.tensors:
            if gguf_to_hf_name_map is not None:
                if tensor.name not in gguf_to_hf_name_map:
                    continue
                name = gguf_to_hf_name_map[tensor.name]
            else:
                name = tensor.name

            weight_type = tensor.tensor_type

            # Dequantize tensors belonging to modules marked as unquantized
            # but stored quantized in the GGUF (e.g. GLM-5.2 indexer
            # wk/weights_proj with Q8_0).  Yield as .weight, not .qweight.
            is_unquant_module = (
                weight_type.name not in _QUANT_TYPES
                and unquantized_modules
                and any(
                    mod in name.removesuffix(".weight")
                    for mod in unquantized_modules
                )
            )
            if "indexer" in name:
                logger.debug(
                    "GGUF indexer tensor: gguf=%s hf_name=%s type=%s "
                    "is_unquant_module=%s unquantized_modules=%s",
                    tensor.name, name, weight_type.name,
                    is_unquant_module, unquantized_modules,
                )
            if is_unquant_module:
                from gguf import GGML_QUANT_SIZES

                from vllm_gguf_plugin.ops import ggml_dequantize

                block_size, type_size = GGML_QUANT_SIZES[weight_type]
                raw = torch.tensor(tensor.data)
                rows = raw.shape[0] if raw.dim() > 1 else 1
                cols = raw.shape[-1] // type_size * block_size
                logger.debug(
                    "Dequantizing %s: %s %s -> fp32 shape [%s, %s]",
                    name, weight_type.name, tuple(raw.shape), rows, cols,
                )
                param = ggml_dequantize(
                    raw.cuda(), weight_type, rows, cols, torch.float32
                ).cpu()
                yield name, param
                continue

            if weight_type.name not in _QUANT_TYPES:
                yield name.replace("weight", "qweight_type"), torch.tensor(weight_type)
                name = name.replace("weight", "qweight")

            weight = tensor.data
            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            yield name, param


def get_gguf_unquantized_params(gguf_files: list[str]) -> list[str]:
    _QUANT_TYPES = ("F32", "BF16", "F16")
    return list(
        {
            tensor.name
            for gguf_file in gguf_files
            for tensor in gguf.GGUFReader(gguf_file).tensors
            if tensor.tensor_type.name in _QUANT_TYPES
        }
    )
    # for gguf_file in gguf_files:
    #     reader = gguf.GGUFReader(gguf_file)
    #     for tensor in reader.tensors:
    #         if tensor.tensor_type.name in unquant_types:
    #             yield tensor.name.rsplit(".", 1)[0]
