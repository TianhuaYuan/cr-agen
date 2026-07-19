"""评测 summary API 测试（Task 3）。

测试 GET /api/v1/evaluation/summary：
- 返回结构正确（total / composite_avg / prf_avg / by_category / per_sample）
- 26 条样本数据
- 内存缓存生效（第二次请求快）
- ?force=1 强制重新计算
"""
import time

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.database import Base, get_db


from pathlib import Path
EVAL_DATASET = str(Path(__file__).resolve().parent / "eval_samples" / "dataset.json")


@pytest_asyncio.fixture
async def eval_api_client(monkeypatch):
    """评测 API 测试客户端：内存 DB + mock LLM + httpx。"""
    import backend.models  # noqa: F401

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with maker() as session:
            yield session

    monkeypatch.setattr("backend.api.evaluation.DEFAULT_DATASET", EVAL_DATASET)
    monkeypatch.setattr("backend.api.evaluation._cache", {"data": None, "loaded_at": 0.0})

    from backend.main import create_app

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()


class TestEvaluationSummaryAPI:
    """GET /api/v1/evaluation/summary。"""

    async def test_summary_200_and_structure(self, eval_api_client):
        """返回 200，包含关键字段。"""
        res = await eval_api_client.get("/api/v1/evaluation/summary")
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "composite_avg" in data
        assert "prf_avg" in data
        assert "by_category" in data
        assert "per_sample" in data

    async def test_summary_total_26(self, eval_api_client):
        """26 条样本。"""
        res = await eval_api_client.get("/api/v1/evaluation/summary")
        data = res.json()
        assert data["total"] == 26

    async def test_summary_prf_avg_structure(self, eval_api_client):
        """prf_avg 包含 precision/recall/f1。"""
        res = await eval_api_client.get("/api/v1/evaluation/summary")
        data = res.json()
        prf = data["prf_avg"]
        assert "precision" in prf
        assert "recall" in prf
        assert "f1" in prf
        assert 0.0 <= prf["precision"] <= 1.0
        assert 0.0 <= prf["recall"] <= 1.0
        assert 0.0 <= prf["f1"] <= 1.0

    async def test_summary_by_category_keys(self, eval_api_client):
        """by_category 包含 4 个分类。"""
        res = await eval_api_client.get("/api/v1/evaluation/summary")
        data = res.json()
        cats = data["by_category"]
        assert set(cats.keys()) == {"security", "quality", "performance", "structure"}
        for cat, info in cats.items():
            assert "count" in info
            assert "composite_avg" in info
            assert "prf" in info

    async def test_summary_per_sample_count(self, eval_api_client):
        """per_sample 数量 = total。"""
        res = await eval_api_client.get("/api/v1/evaluation/summary")
        data = res.json()
        assert len(data["per_sample"]) == data["total"]

    async def test_summary_cache_faster_second_call(self, eval_api_client):
        """第二次请求更快（缓存命中）。"""
        # 清空缓存，确保第一次是冷启动
        t1 = time.perf_counter()
        await eval_api_client.get("/api/v1/evaluation/summary")
        t2 = time.perf_counter()
        await eval_api_client.get("/api/v1/evaluation/summary")
        t3 = time.perf_counter()
        first = t2 - t1
        second = t3 - t2
        assert second < first, f"第二次({second:.4f}s) 应比第一次({first:.4f}s) 快"

    async def test_summary_force_refresh(self, eval_api_client):
        """?force=1 强制刷新，from_cache=False。"""
        res1 = await eval_api_client.get("/api/v1/evaluation/summary")
        data1 = res1.json()
        res2 = await eval_api_client.get("/api/v1/evaluation/summary?force=1")
        data2 = res2.json()
        assert data2["total"] == data1["total"]
        assert "from_cache" in data2
        assert data2["from_cache"] is False


class TestScanThresholdAPI:
    """GET /api/v1/evaluation/scan-threshold（Task 13.4）。"""

    async def test_scan_threshold_200_and_structure(self, eval_api_client):
        """返回 200，包含 table / current_threshold / best_f1_threshold。"""
        res = await eval_api_client.get("/api/v1/evaluation/scan-threshold")
        assert res.status_code == 200
        data = res.json()
        assert "table" in data
        assert "current_threshold" in data
        assert "best_f1_threshold" in data

    async def test_scan_threshold_table_10_rows(self, eval_api_client):
        """table 含 10 行（0.0~0.9 步长 0.1）。"""
        res = await eval_api_client.get("/api/v1/evaluation/scan-threshold")
        data = res.json()
        assert len(data["table"]) == 10
        # 阈值递增 0.0 → 0.9
        thresholds = [row["threshold"] for row in data["table"]]
        assert thresholds == [round(i / 10, 1) for i in range(10)]

    async def test_scan_threshold_row_fields(self, eval_api_client):
        """每行含 threshold / precision / recall / f1。"""
        res = await eval_api_client.get("/api/v1/evaluation/scan-threshold")
        data = res.json()
        for row in data["table"]:
            assert "threshold" in row
            assert "precision" in row
            assert "recall" in row
            assert "f1" in row
            assert 0.0 <= row["precision"] <= 1.0
            assert 0.0 <= row["recall"] <= 1.0
            assert 0.0 <= row["f1"] <= 1.0

    async def test_scan_threshold_best_f1_in_range(self, eval_api_client):
        """best_f1_threshold 在 0.0~0.9 之间。"""
        res = await eval_api_client.get("/api/v1/evaluation/scan-threshold")
        data = res.json()
        assert 0.0 <= data["best_f1_threshold"] <= 0.9

    async def test_scan_threshold_current_threshold_from_settings(self, eval_api_client):
        """current_threshold 来自 settings.DEFAULT_CONFIDENCE_THRESHOLD。"""
        res = await eval_api_client.get("/api/v1/evaluation/scan-threshold")
        data = res.json()
        # 默认 0.0
        assert data["current_threshold"] == 0.0

