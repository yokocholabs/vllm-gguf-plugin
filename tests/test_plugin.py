# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.transformers_utils.config as config_module
from transformers import PretrainedConfig
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.quantization as gguf_quantization
import vllm_gguf_plugin.weight_utils as weight_utils_module
import vllm_gguf_plugin.weights_adapter.default as default_adapter_module
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.loader import GGUFModelLoader
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)


def test_register_overrides_gguf_config():
    register()

    quant_config = get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader
    )
    assert isinstance(get_config_parser("gguf"), GGUFConfigParser)


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"


def test_supported_act_dtypes_includes_bfloat16():
    quant_config = OOTGGUFConfig.from_config({})
    supported = quant_config.get_supported_act_dtypes()

    assert torch.half in supported
    assert torch.bfloat16 in supported
    assert torch.float32 in supported


def test_gguf_linear_uses_weight_loader_v2(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )

    assert "GGUFLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED
    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)
    assert layer.qweight.weight_loader.__name__.endswith("weight_loader_v2")

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(4, dtype=torch.uint8), 1)

    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert len(layer.qweight.data_container) == 2
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)

    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shard_id == [0, 1]
    assert layer.qweight_type.shard_weight_type == {0: 3, 1: 4}


def test_gguf_embedding_uses_plugin_weight_loader(monkeypatch):
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
        quant_config=OOTGGUFConfig.from_config({}),
    )

    loaded_qweight = torch.arange(60, dtype=torch.uint8).reshape(10, 6)
    layer.qweight.weight_loader(layer.qweight, loaded_qweight)
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(7, dtype=torch.uint8)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shape == (16, 6)
    assert torch.equal(layer.qweight[:10], loaded_qweight)
    assert torch.equal(layer.qweight[10:], torch.zeros((6, 6), dtype=torch.uint8))
    assert torch.equal(layer.qweight_type, torch.tensor([7], dtype=torch.uint8))
    assert layer.qweight_type.weight_type == 7


def test_gguf_linear_same_type_shards_skip_concat(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )
    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, torch.nn.Parameter)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_fused_mul_mat_gguf(x, qweight, qweight_type):
        calls.append((tuple(qweight.shape), qweight_type))
        return torch.zeros(
            (x.shape[0], qweight.shape[0]), dtype=x.dtype, device=x.device
        )

    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", fake_fused_mul_mat_gguf
    )
    out = layer.quant_method.apply(layer, torch.ones((2, 4), dtype=torch.float32))

    assert calls == [((8, 4), 3)]
    assert out.shape == (2, 8)


def test_gguf_config_parser_uses_parent_dir_for_local_file(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        return {}, PretrainedConfig(model_type="qwen3_moe")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    config_dict, config = GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert calls["model"] == gguf_path.parent
    assert calls["trust_remote_code"] is False
    assert config_dict["norm_topk_prob"] is True
    assert config.architectures == ["Qwen3MoeForCausalLM"]


def test_register_sets_engine_args_for_gguf_model(monkeypatch):
    register()
    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="/tmp/model.gguf", tokenizer="/tmp/tokenizer")

    engine_args.create_model_config()

    assert captured["config_format"] == "gguf"
    assert captured["model"] == "/tmp/tokenizer"
    assert captured["model_weights"] == "/tmp/model.gguf"
    assert captured["quantization"] == "gguf"
    assert engine_args.load_format == "gguf"


def test_register_skips_speculator_probe_for_gguf():
    register()

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model="/tmp/model.gguf",
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config={"foo": "bar"},
            hf_token=None,
        )
    )

    assert model == "/tmp/model.gguf"
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config == {"foo": "bar"}


def test_gguf_qkv_shards_are_padded_in_qkv_order(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )

    q = torch.full((4, 4), 1, dtype=torch.uint8)
    k = torch.full((2, 4), 2, dtype=torch.uint8)
    v = torch.full((2, 4), 3, dtype=torch.uint8)
    # Load out of canonical order to match GGUF tensor iteration order.
    layer.weight_loader_v2(layer.qweight, k, "k")
    layer.weight_loader_v2(layer.qweight, q, "q")
    layer.weight_loader_v2(layer.qweight, v, "v")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "k")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "q")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "v")

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.shard_id == ["q", "k", "v"]
    assert layer.qweight.shard_offset_map == {
        "q": (0, 4, 4),
        "k": (4, 6, 4),
        "v": (6, 8, 4),
    }
    assert torch.equal(layer.qweight[:4], q)
    assert torch.equal(layer.qweight[4:6], k)
    assert torch.equal(layer.qweight[6:8], v)


def test_gguf_linear_preserves_cuda_weight_device(monkeypatch):
    if not torch.cuda.is_available():
        return

    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    with torch.device("cuda"):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_sizes=[4, 4],
            bias=False,
            quant_config=OOTGGUFConfig.from_config({}),
            disable_tp=True,
        )

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.device.type == "cuda"
    assert layer.qweight_type.device.type == "cuda"


def test_gguf_pp_filter_uses_vllm_missing_parameter_predicate(monkeypatch):
    adapter = default_adapter_module.GGUFWeightsAdapter.__new__(
        default_adapter_module.GGUFWeightsAdapter
    )
    original_map = {
        "blk.0.weight": "model.layers.0.weight",
        "blk.40.weight": "model.layers.40.weight",
        "output.weight": "lm_head.weight",
    }
    adapter.load_spec = default_adapter_module.GGUFLoadSpec(
        weights_source=[],
        unquantized_modules=[],
        gguf_to_hf_name_map=original_map.copy(),
    )
    model = torch.nn.Module()
    missing = {"model.layers.40.weight", "lm_head.weight"}
    checked = []

    def fake_is_pp_missing_parameter(name, candidate_model):
        assert candidate_model is model
        checked.append(name)
        return name in missing

    monkeypatch.setattr(
        default_adapter_module,
        "is_pp_missing_parameter",
        fake_is_pp_missing_parameter,
    )

    adapter.restrict_to_model(model)

    assert checked == list(original_map.values())
    assert adapter.load_spec.gguf_to_hf_name_map == {
        "blk.0.weight": "model.layers.0.weight"
    }


def _mtp_indexer_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleDict({"78": torch.nn.Module()})
    layer = model.model.layers["78"]
    layer.mtp_block = torch.nn.Module()
    layer.mtp_block.probe = torch.nn.Linear(1, 1, bias=False)
    layer.mtp_block.self_attn = torch.nn.Module()
    layer.mtp_block.self_attn.indexer = torch.nn.Module()
    layer.mtp_block.self_attn.indexer.wk_weights_proj = torch.nn.Linear(
        4, 3, bias=False
    )
    return model


def test_gguf_filter_limits_mtp_draft_to_materialized_layer():
    adapter = default_adapter_module.GGUFWeightsAdapter.__new__(
        default_adapter_module.GGUFWeightsAdapter
    )
    adapter.load_spec = default_adapter_module.GGUFLoadSpec(
        weights_source=[],
        unquantized_modules=[],
        gguf_to_hf_name_map={
            "blk.0.weight": "model.layers.0.weight",
            "blk.77.weight": "model.layers.77.weight",
            "blk.78.weight": "model.layers.78.mtp_block.probe.weight",
            "token_embd.weight": "model.embed_tokens.weight",
        },
    )

    adapter.restrict_to_model(_mtp_indexer_model())

    assert adapter.load_spec.gguf_to_hf_name_map == {
        "blk.78.weight": "model.layers.78.mtp_block.probe.weight",
        "token_embd.weight": "model.embed_tokens.weight",
    }


def test_gguf_iterator_streams_from_mmap_and_releases_shard_cache(monkeypatch):
    data = np.arange(16, dtype=np.uint8)

    class FakeWeightType:
        name = "F32"

    class FakeTensor:
        name = "blk.0.weight"
        tensor_type = FakeWeightType()

        def __init__(self):
            self.data = data

    class FakeReader:
        byte_order = "L"

        def __init__(self, _path):
            self.tensors = [FakeTensor()]

    released = []
    monkeypatch.setattr(weight_utils_module.gguf, "GGUFReader", FakeReader)
    monkeypatch.setattr(
        weight_utils_module,
        "_drop_gguf_file_cache",
        released.append,
    )

    weights = list(
        weight_utils_module.gguf_quant_weights_iterator_multi(
            ["model-00001.gguf"],
            {"blk.0.weight": "model.layers.0.weight"},
        )
    )

    assert len(weights) == 1
    _, loaded = weights[0]
    assert loaded.data_ptr() == data.__array_interface__["data"][0]
    assert released == ["model-00001.gguf"]


def test_streaming_copy_uses_bounded_mmap_windows():
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    destination = torch.empty_like(source)
    releases = []
    source._gguf_mmap_release = lambda: releases.append(True)

    weight_utils_module.streaming_copy_(
        destination,
        source,
        chunk_bytes=source.stride(0) * source.element_size() * 2,
    )

    assert torch.equal(destination, source)
    assert len(releases) == 4


def test_indexer_loader_resolves_mtp_block():
    model = _mtp_indexer_model()
    mapped_name = "model.layers.78.self_attn.indexer.wk_weights_proj.weight"
    wk = torch.ones((2, 4))
    weights_proj = 2 * torch.ones((1, 4))

    GGUFModelLoader._load_indexer_weights(
        model,
        [(mapped_name, wk), (mapped_name, weights_proj)],
    )

    actual = dict(model.named_parameters())[
        "model.layers.78.mtp_block.self_attn.indexer.wk_weights_proj.weight"
    ]
    assert torch.equal(actual[:2], wk)
    assert torch.equal(actual[2:], weights_proj)


def test_indexer_loader_rejects_shape_mismatch():
    model = _mtp_indexer_model()
    mapped_name = "model.layers.78.self_attn.indexer.wk_weights_proj.weight"

    with pytest.raises(ValueError, match="shape mismatch"):
        GGUFModelLoader._load_indexer_weights(
            model,
            [
                (mapped_name, torch.ones((2, 4))),
                (mapped_name, torch.ones((2, 4))),
            ],
        )


def test_indexer_loader_rejects_missing_mtp_parameter():
    model = _mtp_indexer_model()
    del model.model.layers["78"].mtp_block.self_attn.indexer.wk_weights_proj
    mapped_name = "model.layers.78.self_attn.indexer.wk_weights_proj.weight"

    with pytest.raises(ValueError, match="Required MTP indexer weight"):
        GGUFModelLoader._load_indexer_weights(
            model,
            [(mapped_name, torch.ones((3, 4)))],
        )
