# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import mmap
import os
import warnings
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


def _mmap_tensor(array: np.ndarray) -> torch.Tensor:
    """Create a read-only, zero-copy tensor view over GGUF mmap data."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        return torch.from_numpy(array)


def _advise_gguf_mmap(reader: object, advice: int | None) -> None:
    """Apply mmap advice to the GGUF reader when its mapping is exposed."""
    if advice is None:
        return
    data = getattr(reader, "data", None)
    mapping = getattr(data, "_mmap", None)
    madvise = getattr(mapping, "madvise", None)
    if madvise is None:
        return
    try:
        madvise(advice)
    except (OSError, ValueError):
        logger.debug("Could not apply mmap advice to GGUF shard", exc_info=True)


def _drop_gguf_file_cache(path: str) -> None:
    """Release completed GGUF shard pages from the Linux filesystem cache."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        logger.info("Released GGUF shard file cache: %s", os.path.basename(path))
    except OSError:
        logger.debug("Could not release GGUF shard file cache: %s", path, exc_info=True)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str],
    gguf_to_hf_name_map: dict[str, str] | None = None,
    unquantized_modules: list[str] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield tensors from mmap without materializing full host-side copies.

    The consumer may narrow a tensor for TP before its single device copy.
    Completed shard pages are evicted from the filesystem cache so sequential
    GGUF reads cannot displace anonymous runtime memory into swap.
    """
    _QUANT_TYPES = ("F32", "BF16", "F16")

    logger.debug(
        "gguf_quant_weights_iterator_multi: files=%s unquantized_modules=%s",
        [os.path.basename(f) for f in gguf_files],
        unquantized_modules,
    )

    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        _advise_gguf_mmap(reader, getattr(mmap, "MADV_SEQUENTIAL", None))
        try:
            for tensor in reader.tensors:
                if gguf_to_hf_name_map is not None:
                    if tensor.name not in gguf_to_hf_name_map:
                        continue
                    name = gguf_to_hf_name_map[tensor.name]
                else:
                    name = tensor.name

                weight_type = tensor.tensor_type
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
                        tensor.name,
                        name,
                        weight_type.name,
                        is_unquant_module,
                        unquantized_modules,
                    )
                if is_unquant_module:
                    from gguf import GGML_QUANT_SIZES

                    from vllm_gguf_plugin.ops import ggml_dequantize

                    block_size, type_size = GGML_QUANT_SIZES[weight_type]
                    raw = _mmap_tensor(tensor.data)
                    if raw.dim() > 2:
                        leading = raw.shape[:-1]
                        packed_cols = raw.shape[-1]
                        raw_2d = raw.reshape(-1, packed_cols)
                        rows = raw_2d.shape[0]
                        cols = packed_cols // type_size * block_size
                        dequant_2d = ggml_dequantize(
                            raw_2d.cuda(),
                            weight_type,
                            rows,
                            cols,
                            torch.bfloat16,
                        ).cpu()
                        param = dequant_2d.reshape(*leading, cols)
                    else:
                        rows = raw.shape[0] if raw.dim() > 1 else 1
                        cols = raw.shape[-1] // type_size * block_size
                        param = ggml_dequantize(
                            raw.cuda(),
                            weight_type,
                            rows,
                            cols,
                            torch.bfloat16,
                        ).cpu()
                    logger.debug(
                        "Dequantized %s: %s -> bf16 %s",
                        name,
                        weight_type.name,
                        tuple(param.shape),
                    )
                    yield name, param
                    del param
                    continue

                if weight_type.name not in _QUANT_TYPES:
                    yield name.replace("weight", "qweight_type"), torch.tensor(
                        weight_type
                    )
                    name = name.replace("weight", "qweight")

                weight = tensor.data
                if weight_type.name == "BF16" and weight.dtype == np.uint8:
                    weight = weight.view(np.uint16)
                    if reader.byte_order == "S":
                        weight = weight.byteswap()
                    param = _mmap_tensor(weight).view(torch.bfloat16)
                else:
                    param = _mmap_tensor(weight)
                yield name, param
                del param
        finally:
            _advise_gguf_mmap(reader, getattr(mmap, "MADV_DONTNEED", None))
            del reader
            _drop_gguf_file_cache(gguf_file)


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
