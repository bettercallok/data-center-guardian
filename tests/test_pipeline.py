"""
tests/test_pipeline.py — ETL safety and training-data utilities
"""
import zipfile

import numpy as np
import polars as pl
import pytest

from src.etl.pipeline import ETLError, _validate_url, safe_extract
from src.ml.train_survival import check_data_quality, compute_real_ttf, shuffled_split


class TestETLSafety:

    def test_accepts_backblaze_archive_url(self):
        _validate_url("https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2026.zip")

    @pytest.mark.parametrize("url", [
        "http://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2026.zip",
        "https://evil.example/file/Backblaze-Hard-Drive-Data/data_Q1_2026.zip",
        "https://f001.backblazeb2.com/file/other/payload.zip",
        "https://f001.backblazeb2.com.evil.example/file/Backblaze-Hard-Drive-Data/data_Q1_2026.zip",
    ])
    def test_rejects_unexpected_urls(self, url):
        with pytest.raises(ETLError):
            _validate_url(url)

    def test_safe_extract_rejects_path_traversal(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../escape.txt", "pwned")
        with pytest.raises(ETLError, match="Unsafe path"):
            safe_extract(archive, tmp_path / "out")
        assert not (tmp_path.parent / "escape.txt").exists()

    def test_safe_extract_rejects_zip_bombs(self, tmp_path, monkeypatch):
        import src.etl.pipeline as pipeline
        monkeypatch.setattr(pipeline, "MAX_UNCOMPRESSED_BYTES", 10)
        archive = tmp_path / "big.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.csv", "0" * 1000)
        with pytest.raises(ETLError, match="limit"):
            safe_extract(archive, tmp_path / "out")

    def test_safe_extract_happy_path(self, tmp_path):
        archive = tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("data_Q1/2026-01-01.csv", "a,b\n1,2\n")
        out = tmp_path / "out"
        safe_extract(archive, out)
        assert (out / "data_Q1" / "2026-01-01.csv").read_text().startswith("a,b")


class TestTrainingUtilities:

    @pytest.fixture
    def raw(self):
        return pl.DataFrame({
            "date": ["2026-01-01", "2026-01-10", "2026-01-01", "2026-03-01"],
            "serial_number": ["A", "A", "B", "B"],
            "failure": [0, 1, 0, 0],
            "smart_5_raw": [0.0, 8.0, 0.0, 0.0],
            "smart_187_raw": [0.0, 1.0, 0.0, 0.0],
            "smart_188_raw": [0.0, 0.0, 0.0, 0.0],
            "smart_197_raw": [0.0, 4.0, 0.0, 0.0],
            "smart_198_raw": [0.0, 4.0, 0.0, 0.0],
        })

    def test_real_ttf_and_censoring(self, raw):
        stats = compute_real_ttf(raw).sort("serial_number")
        a, b = stats.row(0, named=True), stats.row(1, named=True)
        assert a["ttf"] == 10 and a["y_upper"] == 10          # failed: exact interval
        assert b["ttf"] == 60 and b["y_upper"] == float("inf")  # right-censored

    def test_quality_gate(self, raw):
        stats = compute_real_ttf(raw)
        with pytest.raises(ValueError, match="Quality gate"):
            check_data_quality(stats, min_drives=1000, min_failures=10)
        check_data_quality(stats, min_drives=2, min_failures=1)

    def test_shuffled_split_is_disjoint_and_seeded(self):
        tr, ev = shuffled_split(100)
        assert len(ev) == 20 and len(tr) == 80
        assert not set(tr) & set(ev)
        assert np.array_equal(shuffled_split(100)[1], ev)
        assert not np.array_equal(ev, np.arange(80, 100))  # not a tail slice


def test_training_smoke(tmp_path):
    """Train end-to-end on a small synthetic fleet and load the exported model."""
    import xgboost as xgb
    from src.ml.train_survival import train_survival_model

    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        failed = i % 10 == 0
        days = int(rng.integers(5, 80))
        bad = 50.0 if failed else 0.0
        for d in (0, days):
            rows.append({
                "date": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}",
                "serial_number": f"S{i}",
                "failure": int(failed and d == days),
                "smart_5_raw": bad, "smart_187_raw": bad / 5, "smart_188_raw": 0.0,
                "smart_197_raw": bad, "smart_198_raw": bad,
            })
    data = tmp_path / "survival.parquet"
    pl.DataFrame(rows).write_parquet(data)
    out = tmp_path / "model.json"
    train_survival_model(data, use_gpu=False, output_path=out, min_drives=100, min_failures=5)
    booster = xgb.Booster()
    booster.load_model(str(out))
    preds = booster.predict(xgb.DMatrix(np.array([[0, 0, 0, 0, 0], [50, 10, 0, 50, 50]], dtype=np.float32)))
    assert np.all(np.isfinite(preds)) and np.all(preds > 0)
