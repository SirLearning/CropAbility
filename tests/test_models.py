"""测试 TorchScript 模型导出/加载。"""

import pytest
import torch
from cropability.models.torchscript import (
    AddModule,
    GenomicEmbedding,
    SNPEffectPredictor,
    export_model,
    load_model,
)


class TestAddModule:
    def test_forward(self):
        model = AddModule()
        x = torch.randn(100)
        y = torch.randn(100)
        result = model(x, y)
        assert torch.allclose(result, x + y)

    def test_batch(self):
        model = AddModule()
        x = torch.randn(32, 256)
        y = torch.randn(32, 256)
        result = model(x, y)
        assert result.shape == (32, 256)


class TestGenomicEmbedding:
    def test_forward_shape(self):
        model = GenomicEmbedding(embed_dim=32, hidden_dim=64, output_dim=128)
        x = torch.randint(0, 5, (4, 64))  # batch=4, seq_len=64
        out = model(x)
        assert out.shape == (4, 128)

    def test_padding_idx_n(self):
        model = GenomicEmbedding(embed_dim=16, hidden_dim=32, output_dim=64)
        x = torch.zeros(2, 50, dtype=torch.long)  # all-A
        x_n = torch.full((2, 50), 4, dtype=torch.long)  # all-N
        out_a = model(x)
        out_n = model(x_n)
        # N 被 padding_idx=4 忽略，输出不应崩溃
        assert out_a.shape == (2, 64)
        assert out_n.shape == (2, 64)

    def test_gradient_flow(self):
        model = GenomicEmbedding(embed_dim=16, hidden_dim=32, output_dim=64)
        x = torch.randint(0, 5, (2, 32))
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestSNPEffectPredictor:
    def test_forward(self):
        model = SNPEffectPredictor(seq_embed_dim=64, genomic_features=8)
        seq = torch.randint(0, 5, (4, 128))
        feat = torch.randn(4, 8)
        out = model(seq, feat)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()


class TestModelExport:
    def test_export_add_module(self, tmp_path):
        model = AddModule()
        path = export_model(
            model,
            tmp_path / "add.pt",
            example_inputs=(torch.randn(10), torch.randn(10)),
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_and_load(self, tmp_path):
        model = AddModule()
        path = export_model(
            model,
            tmp_path / "add.pt",
            example_inputs=(torch.randn(10), torch.randn(10)),
        )
        loaded = load_model(path, device=torch.device("cpu"))
        x, y = torch.randn(10), torch.randn(10)
        result = loaded(x, y)
        assert torch.allclose(result, x + y)

    def test_export_embedding(self, tmp_path):
        model = GenomicEmbedding(embed_dim=16, hidden_dim=32, output_dim=64)
        path = export_model(
            model,
            tmp_path / "embed.pt",
            example_inputs=(torch.randint(0, 5, (2, 32)),),
        )
        assert path.exists()

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pt")
