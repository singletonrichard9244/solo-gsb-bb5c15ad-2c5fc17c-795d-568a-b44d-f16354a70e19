"""接口与判定逻辑测试。"""

import copy
import itertools
import math
import random
from datetime import datetime, timedelta, timezone
from fractions import Fraction

from fastapi.testclient import TestClient

from app.core import (
    clip_merge_intervals,
    compare_dual_meter,
    reconcile_meter_total,
    reconcile_total,
    solve_daily_segment,
    solve_minimum_rate,
    solve_missing_reading,
    solve_outage_segment,
)
from app.main import app

client = TestClient(app)


def _post(payload, path="/v1/usage/verify"):
    return client.post(path, json=payload)


def _reconcile(payload):
    return _post(payload, path="/v1/usage/reconcile")


def _daily_range(payload):
    return _post(payload, path="/v1/usage/daily-range")


def _meter_change(payload):
    return _post(payload, path="/v1/usage/meter-change/reconcile")


def _missing_reading(payload):
    return _post(payload, path="/v1/usage/missing-reading")


def _minimum_rate(payload):
    return _post(payload, path="/v1/usage/minimum-rate")


def _dual_meter(payload):
    return _post(payload, path="/v1/usage/dual-meter/compare")


def _outage(payload):
    return _post(payload, path="/v1/usage/outage-reconcile")


# ---------------------------------------------------------------------------
# 正常场景
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_no_wrap_unique():
    """无回绕，增量在速率上限内（60s * 10/s = 600，不足以多绕一圈）。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:01:00Z", "value": 600},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_usage"] == 500
    assert body["errors"] == []
    assert body["segments"][0]["wraps"] == 0
    assert body["segments"][0]["usage"] == 500


def test_single_wrap_unique():
    """一次回绕且唯一：900 -> 100，M=1000，10s 内上限 200。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 20,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 900},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["segments"][0]["wraps"] == 1
    assert body["segments"][0]["usage"] == 200
    assert body["total_usage"] == 200


def test_multi_segment_total():
    """多段混合（回绕 + 不回绕）全部唯一，总用量为各段之和。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 80,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 250},  # +150
                {"timestamp": "2026-09-24T00:00:20Z", "value": 50},   # 回绕 +800
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert [s["usage"] for s in body["segments"]] == [150, 800]
    assert body["segments"][1]["wraps"] == 1
    assert body["total_usage"] == 950


def test_zero_rate_equal_readings():
    """速率为 0 且读数不变：唯一解，用量 0。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 0,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 42},
                {"timestamp": "2026-09-24T01:00:00Z", "value": 42},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_usage"] == 0


def test_fractional_rate_boundary():
    """小数速率按精确分数判定：0.5/s * 10s = 5，增量 5 恰好唯一。"""
    resp = _post(
        {
            "modulus": 100,
            "max_rate_per_second": 0.5,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 10},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 15},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_usage"] == 5


def test_timezone_offset_normalized_to_utc():
    """带偏移的时区会被换算为 UTC 后再判定。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T08:00:00+08:00", "value": 0},
                {"timestamp": "2026-09-24T00:00:30Z", "value": 5},  # 同一时刻后 30s
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["segments"][0]["elapsed_seconds"] == 30
    assert body["total_usage"] == 5


# ---------------------------------------------------------------------------
# 无解 / 多解
# ---------------------------------------------------------------------------

def test_no_solution():
    """增量超过速率上限：1s 内最多 10，但需 800。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 900},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "error"
    assert body["total_usage"] is None
    assert body["segments"] == []
    err = body["errors"][0]
    assert err["reason"] == "no_solution"
    assert err["index"] == 0
    assert err["max_wraps"] < err["min_wraps"]


def test_ambiguous():
    """时间跨度过大导致多个回绕次数都可行。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},  # 1000s, 上限 10000
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "error"
    assert body["total_usage"] is None
    err = body["errors"][0]
    assert err["reason"] == "ambiguous"
    assert err["min_wraps"] == 0
    assert err["max_wraps"] == 9  # k=0..9 均满足


def test_mixed_segments_partial_errors():
    """部分区间唯一、部分出错：返回已判定分段与出错区间，总用量为 null。"""
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 20,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 300},   # 唯一 +200
                {"timestamp": "2026-09-24T00:00:11Z", "value": 900},   # 无解
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "error"
    assert body["total_usage"] is None
    assert len(body["segments"]) == 1
    assert body["segments"][0]["usage"] == 200
    assert len(body["errors"]) == 1
    assert body["errors"][0]["index"] == 1
    assert body["errors"][0]["reason"] == "no_solution"


# ---------------------------------------------------------------------------
# 请求校验（422）
# ---------------------------------------------------------------------------

def test_timestamps_not_strictly_increasing():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


def test_reading_out_of_modulus_range():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 1000},
            ],
        }
    )
    assert resp.status_code == 422


def test_naive_timestamp_rejected():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


def test_subsecond_timestamp_rejected():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00.500Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


def test_too_few_readings():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [{"timestamp": "2026-09-24T00:00:00Z", "value": 100}],
        }
    )
    assert resp.status_code == 422


def test_negative_rate_rejected():
    resp = _post(
        {
            "modulus": 1000,
            "max_rate_per_second": -1,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 账单总量对账 /v1/usage/reconcile
# ---------------------------------------------------------------------------

def test_reconcile_resolves_ambiguous_segment():
    """单段多解 [0,9]：账单 3100 隐含 K=3，圈数唯一确定。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 3100,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 3
    assert body["conflicts"] == []
    seg = body["segments"][0]
    assert seg["wraps"] == 3
    assert seg["usage"] == 3100
    assert seg["min_wraps"] == seg["max_wraps"] == 3
    assert seg["min_usage"] is None and seg["max_usage"] is None


def test_reconcile_mixed_unique_and_ranges():
    """两段多解，K=5：段1 仍为范围 [0,4]，段2 为范围 [1,5]，均不给确定用量。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 5000,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 5
    s1, s2 = body["segments"]
    assert (s1["min_wraps"], s1["max_wraps"]) == (0, 4)
    assert s1["wraps"] is None and s1["usage"] is None
    assert (s1["min_usage"], s1["max_usage"]) == (100, 4100)
    assert (s2["min_wraps"], s2["max_wraps"]) == (1, 5)
    assert (s2["min_usage"], s2["max_usage"]) == (900, 4900)


def test_reconcile_feasible_endpoint_all_unique():
    """K 取区间端点时各段圈数全部唯一，确定用量之和恰为账单总量。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 1000,  # K = sum_min = 1
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert [s["usage"] for s in body["segments"]] == [100, 900]
    assert sum(s["usage"] for s in body["segments"]) == 1000


def test_reconcile_incompatible_modulus():
    """总量与首尾读数差模 M 不同余：返回不相容冲突及可行区间。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 5500,  # 应为 1000 的整数倍
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "conflict"
    assert body["segments"] == []
    assert body["total_wraps"] is None
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_incompatible_modulus"
    assert conflict["expected_remainder"] == 0
    assert conflict["feasible_min_total"] == 1000
    assert conflict["feasible_max_total"] == 19000


def test_reconcile_out_of_range():
    """总量同余但总圈数超出可行和：返回超界冲突。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 0,  # 首尾相同要求 K=0，但回绕段至少 1 圈
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "conflict"
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "total_out_of_range"
    assert conflict["total_wraps"] == 0
    assert conflict["feasible_min_total"] == 1000


def test_reconcile_segment_no_solution_conflict():
    """某段本身无解（超过速率上限）：以分段冲突返回，不做总量推断。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 20,
            "total_usage": 300,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 300},  # 可行
                {"timestamp": "2026-09-24T00:00:11Z", "value": 900},  # 无解
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["segments"] == []
    assert len(body["conflicts"]) == 1
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "segment"
    assert conflict["index"] == 1
    assert conflict["reason"] == "no_solution"
    assert conflict["max_wraps"] < conflict["min_wraps"]


def test_reconcile_negative_total_rejected():
    """账单总用量为负数：422。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": -1,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


def test_reconcile_total_must_be_integer():
    """账单总用量必须为整数：422。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 12.5,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


def test_reconcile_zero_total_with_flat_readings():
    """首尾相同且全部段都允许 0 圈时，账单 0 可行且各段唯一为 0。"""
    resp = _reconcile(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "total_usage": 0,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    seg = body["segments"][0]
    assert seg["wraps"] == 0 and seg["usage"] == 0


def test_reconcile_reuses_reading_validation():
    """新接口复用读数校验：读数越界、时间戳不递增均 422。"""
    base = {
        "modulus": 1000,
        "max_rate_per_second": 10,
        "total_usage": 100,
    }
    resp = _reconcile(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 1000},
            ],
        }
    )
    assert resp.status_code == 422
    resp = _reconcile(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 每日用量上下界 /v1/usage/daily-range
# ---------------------------------------------------------------------------

def test_daily_range_single_day_unique():
    """区间不跨日：唯一增量，当日 min == max。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 10,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:01:00Z", "value": 600},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["conflicts"] == []
    assert body["days"] == [
        {
            "date": "2026-09-24",
            "min_usage": 500,
            "max_usage": 500,
            "segment_count": 1,
        }
    ]
    assert "分别达到" in body["boundary_note"]


def test_daily_range_cross_midnight_wrap():
    """跨午夜 + 一次回绕：250 总量在两个容量 500 的子段间自由分配。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 100,
            "readings": [
                {"timestamp": "2026-09-24T23:59:55Z", "value": 800},
                {"timestamp": "2026-09-25T00:00:05Z", "value": 50},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    days = body["days"]
    assert [d["date"] for d in days] == ["2026-09-24", "2026-09-25"]
    assert [(d["min_usage"], d["max_usage"]) for d in days] == [(0, 250), (0, 250)]
    assert all(d["segment_count"] == 1 for d in days)
    # 每日上界 250 可分别达到（(250,0) 或 (0,250)），但不能同时为 250
    assert days[0]["max_usage"] + days[1]["max_usage"] > 250


def test_daily_range_upper_bounds_not_simultaneous():
    """两个容量 5 的子段承担总量 8：每日 [3,5]，上界无法同时达到。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 1,
            "readings": [
                {"timestamp": "2026-09-24T23:59:55Z", "value": 992},
                {"timestamp": "2026-09-25T00:00:05Z", "value": 0},
            ],
        }
    )
    body = resp.json()
    days = body["days"]
    assert [(d["min_usage"], d["max_usage"]) for d in days] == [(3, 5), (3, 5)]
    # 可行分配只有 (3,5)/(4,4)/(5,3)：任何分配下两日用量之和恒为 8
    assert sum(d["max_usage"] for d in days) == 10  # 上界之和不可同时达到
    assert sum(d["min_usage"] for d in days) == 6   # 下界之和同样不可同时达到


def test_daily_range_aggregates_multiple_intervals():
    """同一天两个独立区间的子段界求和，segment_count 计数。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 100,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},  # k∈[0,1]
                {"timestamp": "2026-09-24T00:00:20Z", "value": 100},  # k∈[0,1]
            ],
        }
    )
    body = resp.json()
    assert body["days"] == [
        {
            "date": "2026-09-24",
            "min_usage": 0,
            "max_usage": 2000,
            "segment_count": 2,
        }
    ]


def test_daily_range_multi_day_span():
    """跨三天：每天一条子段，容量按整日秒数取整。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 1,
            "readings": [
                {"timestamp": "2026-09-24T12:00:00Z", "value": 0},
                {"timestamp": "2026-09-26T12:00:00Z", "value": 0},
            ],
        }
    )
    body = resp.json()
    days = body["days"]
    assert [d["date"] for d in days] == [
        "2026-09-24",
        "2026-09-25",
        "2026-09-26",
    ]
    assert [(d["min_usage"], d["max_usage"]) for d in days] == [
        (0, 43200),
        (0, 86400),
        (0, 43200),
    ]


def test_daily_range_fractional_rate_floors_capacity():
    """0.5/s × 3s 容量 floor 为 1：总量 2 迫使两子段各为 1。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 0.5,
            "readings": [
                {"timestamp": "2026-09-24T23:59:57Z", "value": 998},
                {"timestamp": "2026-09-25T00:00:03Z", "value": 0},
            ],
        }
    )
    body = resp.json()
    assert [(d["min_usage"], d["max_usage"]) for d in body["days"]] == [
        (1, 1),
        (1, 1),
    ]


def test_daily_range_no_feasible_allocation_conflict():
    """整段速率判定可行、但切日取整后总容量不足：返回区间索引与冲突原因。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 1,
            "readings": [
                {"timestamp": "2026-09-24T23:59:55Z", "value": 950},
                {"timestamp": "2026-09-25T00:00:05Z", "value": 100},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["days"] == []
    conflict = body["conflicts"][0]
    assert conflict["index"] == 0
    assert conflict["reason"] == "no_feasible_allocation"
    assert conflict["total_capacity"] == 10
    assert conflict["min_wraps"] == 1
    assert conflict["max_wraps"] == 0


def test_daily_range_conflict_skips_no_daily_totals():
    """存在冲突区间时不返回任何日用量，但列出全部冲突区间。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 1,
            "readings": [
                {"timestamp": "2026-09-24T23:59:55Z", "value": 950},
                {"timestamp": "2026-09-25T00:00:05Z", "value": 100},  # 区间0冲突
                {"timestamp": "2026-09-25T00:00:06Z", "value": 900},  # 区间1也冲突
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["days"] == []
    assert [c["index"] for c in body["conflicts"]] == [0, 1]
    assert all(c["reason"] == "no_feasible_allocation" for c in body["conflicts"])


def test_daily_range_timezone_offset_normalized_to_utc():
    """带 +08:00 偏移的时间戳换算为 UTC 后切日。"""
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": 1,
            "readings": [
                {"timestamp": "2026-09-25T07:59:55+08:00", "value": 995},
                {"timestamp": "2026-09-25T00:00:05Z", "value": 0},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    # 即 UTC 2026-09-24T23:59:55 -> 2026-09-25T00:00:05
    assert [d["date"] for d in body["days"]] == ["2026-09-24", "2026-09-25"]
    assert [(d["min_usage"], d["max_usage"]) for d in body["days"]] == [
        (0, 5),
        (0, 5),
    ]


def test_daily_range_reuses_reading_validation():
    """新接口复用读数/模数/速率校验：非法输入一律 422。"""
    base = {"modulus": 1000, "max_rate_per_second": 10}
    # 读数越界
    resp = _daily_range(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 1000},
            ],
        }
    )
    assert resp.status_code == 422
    # 时间戳不递增
    resp = _daily_range(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:01Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422
    # 缺时区
    resp = _daily_range(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422
    # 亚秒
    resp = _daily_range(
        {
            **base,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00.5Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422
    # 只有 1 条读数
    resp = _daily_range(
        {
            **base,
            "readings": [{"timestamp": "2026-09-24T00:00:00Z", "value": 100}],
        }
    )
    assert resp.status_code == 422
    # 负速率
    resp = _daily_range(
        {
            "modulus": 1000,
            "max_rate_per_second": -1,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:01Z", "value": 200},
            ],
        }
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# daily-range 核心界的精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------

def test_daily_segment_bounds_exact_by_brute_force():
    """随机小参数下，闭式边际界与枚举全部 (k, 子段分配) 的结果一致。"""

    def brute(r1, r2, secs, modulus, rate):
        caps = [math.floor(rate * s) for s in secs]
        total_cap = sum(caps)
        diff = r2 - r1
        lows = [None] * len(secs)
        highs = [None] * len(secs)
        feasible = False
        k = 0
        while diff + k * modulus <= total_cap:
            delta = diff + k * modulus
            if delta >= 0:
                feasible = True
                spans = [range(min(caps[j], delta) + 1) for j in range(len(secs))]
                for xs in itertools.product(*spans):
                    if sum(xs) == delta:
                        for j, x in enumerate(xs):
                            lows[j] = x if lows[j] is None else min(lows[j], x)
                            highs[j] = x if highs[j] is None else max(highs[j], x)
            k += 1
        return feasible, caps, lows, highs

    rng = random.Random(20260925)
    checked = 0
    for _ in range(4000):
        modulus = rng.choice([3, 5, 7, 10])
        rate = Fraction(rng.randint(0, 20), rng.randint(1, 3))
        start = datetime(2026, 9, 24, tzinfo=timezone.utc) + timedelta(
            seconds=rng.randint(0, 86399)
        )
        end = start + timedelta(seconds=rng.randint(1, 30))
        r1, r2 = rng.randrange(modulus), rng.randrange(modulus)
        result = solve_daily_segment(0, r1, r2, start, end, modulus, rate)
        secs = [s.seconds for s in result.subsegments]
        if sum(math.floor(rate * s) for s in secs) > 25:
            continue  # 控制枚举规模
        checked += 1
        feasible, caps, lows, highs = brute(r1, r2, secs, modulus, rate)
        assert result.feasible == feasible
        if feasible:
            assert [s.capacity for s in result.subsegments] == caps
            assert list(result.min_usages) == lows
            assert list(result.max_usages) == highs
    assert checked > 200  # 确认确实跑了足够多的用例


# ---------------------------------------------------------------------------
# 换表账单复核 /v1/usage/meter-change/reconcile
# ---------------------------------------------------------------------------

def _meter_payload(total_usage, old_mod=1000, new_mod=1500, rate=10, secs=1000,
                   old_values=(900, 50), new_values=(1200, 300),
                   junction="2026-09-24T00:16:40Z",
                   old_start="2026-09-24T00:00:00Z",
                   new_end="2026-09-24T00:33:20Z"):
    return {
        "total_usage": total_usage,
        "old_meter": {
            "modulus": old_mod,
            "max_rate_per_second": rate,
            "start": {"timestamp": old_start, "value": old_values[0]},
            "end": {"timestamp": junction, "value": old_values[1]},
        },
        "new_meter": {
            "modulus": new_mod,
            "max_rate_per_second": rate,
            "start": {"timestamp": junction, "value": new_values[0]},
            "end": {"timestamp": new_end, "value": new_values[1]},
        },
    }


def test_meter_change_unique_both_segments():
    """B=2250：旧表 1 圈(150)、新表 2 圈(2100)，两段同时唯一。"""
    resp = _meter_change(_meter_payload(2250))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["unique"] is True
    assert body["conflicts"] == []
    old, new = body["segments"]
    assert old["meter"] == "old" and new["meter"] == "new"
    assert (old["wraps"], old["usage"]) == (1, 150)
    assert (new["wraps"], new["usage"]) == (2, 2100)
    assert old["min_usage"] is None and old["max_usage"] is None
    assert new["min_usage"] is None and new["max_usage"] is None
    assert old["elapsed_seconds"] == 1000
    assert old["modulus"] == 1000 and new["modulus"] == 1500


def test_meter_change_ambiguous_ranges_sum_to_bill():
    """B=4750：两表模数不同（1000/1500，gcd=500），各段仍有多解。

    旧表可行圈数 1..7（步长 3），用量 [150,6150]∩账单 -> [1150,4150]；
    新表可行圈数 1..3（步长 2），用量 [600,3600]；任一可行组合两段
    用量之和恒为 4750。
    """
    resp = _meter_change(_meter_payload(4750))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["unique"] is False
    old, new = body["segments"]
    assert old["wraps"] is None and old["usage"] is None
    assert new["wraps"] is None and new["usage"] is None
    assert (old["min_usage"], old["max_usage"]) == (1150, 4150)
    assert (new["min_usage"], new["max_usage"]) == (600, 3600)
    assert (old["min_wraps"], old["max_wraps"], old["wraps_step"]) == (2, 5, 3)
    assert (new["min_wraps"], new["max_wraps"], new["wraps_step"]) == (1, 3, 2)
    # 端点组合严格对账
    assert old["min_usage"] + new["max_usage"] == 4750
    assert old["max_usage"] + new["min_usage"] == 4750


def test_meter_change_equal_moduli_matches_single_modulus():
    """两表模数相同时，与统一模数的总量对账同余规则一致。"""
    resp = _meter_change(
        _meter_payload(2250, old_mod=1000, new_mod=1000, new_values=(200, 300))
    )
    body = resp.json()
    assert body["status"] == "ok"
    old, new = body["segments"]
    # 旧 d=-850、新 d=100，B=2250 => k1+k2=3；边界内多解
    assert (old["min_usage"], old["max_usage"]) == (150, 2150)
    assert (new["min_usage"], new["max_usage"]) == (100, 2100)
    assert old["min_usage"] + new["max_usage"] == 2250


def test_meter_change_incompatible_modulus():
    """B 不被 gcd(1000,1500)=500 整除：total_incompatible_modulus。"""
    resp = _meter_change(_meter_payload(4751))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["segments"] == []
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_incompatible_modulus"
    assert conflict["modulus_gcd"] == 500
    assert conflict["old_modulus"] == 1000
    assert conflict["new_modulus"] == 1500
    # d_old + d_new = -850 - 900 = -1750 ≡ 250 (mod 500)
    assert conflict["expected_remainder"] == 250
    assert conflict["feasible_min_total"] == 750
    assert conflict["feasible_max_total"] == 18750


def test_meter_change_out_of_range_above_box():
    """B 与 gcd 相容但远超可行区间：total_out_of_range。

    相容值须 B ≡ d1+d2 = -1750 ≡ 250 (mod 500)；19250 相容但大于
    可行上界 18750。
    """
    resp = _meter_change(_meter_payload(19250))
    body = resp.json()
    conflict = body["conflicts"][0]
    assert body["status"] == "conflict"
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_out_of_range"
    assert (conflict["feasible_min_total"], conflict["feasible_max_total"]) == (
        750,
        18750,
    )
    assert conflict["expected_remainder"] is None


def test_meter_change_in_box_but_no_integer_pair():
    """B=1250 落在 [750,18750] 内且被 500 整除，但联立无整数组合。

    方程 1000k1 + 1500k2 = 3000 即 2k1+3k2=6，解族 k1∈{0,3}（k2={2,0}），
    两段均要求 k>=1，故全部落在可行区间之外。
    """
    resp = _meter_change(_meter_payload(1250))
    body = resp.json()
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "total_out_of_range"
    # detail 需明确指出"数值在区间内但无整数组合"这一可区分情形
    assert "落在上述区间内" in conflict["detail"]


def test_meter_change_old_segment_no_solution():
    """旧表段本身超过速率上限：分段冲突，不做总量联立。"""
    payload = _meter_payload(2250, rate=10, secs=1)
    payload["old_meter"]["end"] = {
        "timestamp": "2026-09-24T00:00:01Z",
        "value": 50,
    }
    payload["new_meter"]["start"] = {
        "timestamp": "2026-09-24T00:00:01Z",
        "value": 1200,
    }
    payload["new_meter"]["end"] = {
        "timestamp": "2026-09-24T00:16:41Z",
        "value": 300,
    }
    resp = _meter_change(payload)
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["segments"] == []
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "segment"
    assert conflict["meter"] == "old"
    assert conflict["reason"] == "no_solution"
    assert conflict["max_wraps"] < conflict["min_wraps"]


def test_meter_change_both_segments_no_solution_listed():
    """两段都无解时一并列出。"""
    payload = _meter_payload(2250, rate=1, secs=1)
    payload["old_meter"]["end"] = {
        "timestamp": "2026-09-24T00:00:01Z",
        "value": 50,
    }
    payload["new_meter"]["start"] = {
        "timestamp": "2026-09-24T00:00:01Z",
        "value": 1200,
    }
    payload["new_meter"]["end"] = {
        "timestamp": "2026-09-24T00:00:02Z",
        "value": 300,
    }
    resp = _meter_change(payload)
    body = resp.json()
    assert body["status"] == "conflict"
    assert [c["meter"] for c in body["conflicts"]] == ["old", "new"]
    assert all(c["reason"] == "no_solution" for c in body["conflicts"])


def test_meter_change_junction_must_be_same_instant():
    """旧表终读与新表初读不同刻：422。"""
    payload = _meter_payload(2250)
    payload["new_meter"]["start"]["timestamp"] = "2026-09-24T00:16:41Z"
    resp = _meter_change(payload)
    assert resp.status_code == 422


def test_meter_change_timezone_junction_normalized():
    """交界刻以不同时区表达但同一时刻：换算 UTC 后通过，结果不受影响。"""
    payload = _meter_payload(2250)
    # 2026-09-24T08:16:40+08:00 == 2026-09-24T00:16:40Z
    payload["old_meter"]["end"]["timestamp"] = "2026-09-24T08:16:40+08:00"
    resp = _meter_change(payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_meter_change_reuses_reading_validation():
    """复用现有校验：读数越界、时间不递增、缺时区、亚秒、负速率均 422。"""
    base = _meter_payload(2250)

    def expect_422(mut):
        import copy

        payload = copy.deepcopy(base)
        mut(payload)
        assert _meter_change(payload).status_code == 422

    # 旧表读数越界
    expect_422(lambda p: p["old_meter"]["start"].update(value=1000))
    # 新表读数越界（M=1500）
    expect_422(lambda p: p["new_meter"]["end"].update(value=1500))
    # 旧表内部时间不递增
    expect_422(
        lambda p: p["old_meter"]["start"].update(
            timestamp="2026-09-24T00:16:40Z"
        )
    )
    # 缺时区
    expect_422(
        lambda p: p["old_meter"]["start"].update(
            timestamp="2026-09-24T00:00:00"
        )
    )
    # 亚秒
    expect_422(
        lambda p: p["old_meter"]["start"].update(
            timestamp="2026-09-24T00:00:00.5Z"
        )
    )
    # 负速率（各自表独立校验）
    expect_422(lambda p: p["new_meter"].update(max_rate_per_second=-1))
    # 模数 < 1
    expect_422(lambda p: p["old_meter"].update(modulus=0))
    # 账单总量为负
    expect_422(lambda p: p.update(total_usage=-1))


# ---------------------------------------------------------------------------
# reconcile_meter_total 精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------

def test_reconcile_meter_total_exact_by_brute_force():
    """随机小参数下，丢番图通解区间与枚举全部 (k1,k2) 的结果一致。"""
    rng = random.Random(20260925)
    checked = 0
    for _ in range(4000):
        m1 = rng.choice([2, 3, 4, 5, 6, 7])
        m2 = rng.choice([2, 3, 4, 5, 6, 7, 9])
        d1 = rng.randrange(m1) - rng.randrange(m1)
        d2 = rng.randrange(m2) - rng.randrange(m2)
        lo1 = 0 if d1 >= 0 else math.ceil(-d1 / m1)
        lo2 = 0 if d2 >= 0 else math.ceil(-d2 / m2)
        hi1 = lo1 + rng.randint(0, 5)
        hi2 = lo2 + rng.randint(0, 5)

        feasible_B = sorted(
            {
                d1 + k1 * m1 + d2 + k2 * m2
                for k1 in range(lo1, hi1 + 1)
                for k2 in range(lo2, hi2 + 1)
            }
        )
        # 抽测可行 B（含全部最小/最大值）与若干不可行 B
        targets = set(feasible_B)
        for _ in range(3):
            if feasible_B:
                base_b = rng.choice(feasible_B)
                targets.add(base_b + rng.choice([-3, -2, -1, 1, 2, 3]))
        targets.update(
            [feasible_B[0] - 1, feasible_B[-1] + 1] if feasible_B else [d1 + d2]
        )

        for B in targets:
            r = reconcile_meter_total(
                d1, (lo1, hi1), m1, d2, (lo2, hi2), m2, B
            )
            pairs = [
                (k1, k2)
                for k1 in range(lo1, hi1 + 1)
                for k2 in range(lo2, hi2 + 1)
                if d1 + k1 * m1 + d2 + k2 * m2 == B
            ]
            if pairs:
                checked += 1
                assert r.status == "ok", (m1, m2, d1, d2, B, r.conflicts)
                k1s = sorted({p[0] for p in pairs})
                k2s = sorted({p[1] for p in pairs})
                os_, ns = r.old_segment, r.new_segment
                assert (os_.min_wraps, os_.max_wraps) == (k1s[0], k1s[-1])
                assert (ns.min_wraps, ns.max_wraps) == (k2s[0], k2s[-1])
                assert os_.unique == ns.unique == (len(pairs) == 1)
                if len(k1s) > 1:
                    assert os_.wraps_step == k1s[1] - k1s[0]
                    assert ns.wraps_step == k2s[1] - k2s[0]
                # 每个端点都严格对账
                assert (
                    d1 + os_.min_wraps * m1 + d2 + ns.max_wraps * m2 == B
                )
                assert (
                    d1 + os_.max_wraps * m1 + d2 + ns.min_wraps * m2 == B
                )
            else:
                g = math.gcd(m1, m2)
                expect = (
                    "total_incompatible_modulus"
                    if (B - d1 - d2) % g != 0
                    else "total_out_of_range"
                )
                assert r.status == "conflict"
                assert r.conflicts == (expect,), (m1, m2, d1, d2, B)
                assert r.old_segment is None and r.new_segment is None
    assert checked > 500


# ---------------------------------------------------------------------------
# 漏采读数推断 /v1/usage/missing-reading
# ---------------------------------------------------------------------------

def _missing_payload(total_usage, rate=10, modulus=1000,
                     start_value=100, end_value=200,
                     start_ts="2026-09-24T00:00:00Z",
                     missing_ts="2026-09-24T00:00:40Z",
                     end_ts="2026-09-24T00:01:40Z"):
    """默认：左侧 40s、右侧 60s；rate=10 时容量 C_L=400、C_R=600。"""
    return {
        "modulus": modulus,
        "max_rate_per_second": rate,
        "total_usage": total_usage,
        "start": {"timestamp": start_ts, "value": start_value},
        "end": {"timestamp": end_ts, "value": end_value},
        "missing_timestamp": missing_ts,
    }


def test_missing_reading_single_interval():
    """diff=100、B=100（K=0）：左侧 x∈[0,100]，候选为单闭区间 [100,200]。"""
    resp = _missing_reading(_missing_payload(100))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["conflicts"] == []
    assert body["total_wraps"] == 0
    assert (body["left_seconds"], body["right_seconds"]) == (40, 60)
    assert (body["left_capacity"], body["right_capacity"]) == (400, 600)
    assert (body["left_usage_min"], body["left_usage_max"]) == (0, 100)
    assert body["candidate_intervals"] == [
        {"min_reading": 100, "max_reading": 200}
    ]
    assert body["candidate_count"] == 101
    assert body["unique"] is False


def test_missing_reading_unique_candidate():
    """rate=1 时 C_L=40、C_R=60：B=100 迫使左侧恰为 40，候选唯一为 140。"""
    resp = _missing_reading(_missing_payload(100, rate=1))
    body = resp.json()
    assert body["status"] == "ok"
    assert (body["left_usage_min"], body["left_usage_max"]) == (40, 40)
    assert body["candidate_intervals"] == [
        {"min_reading": 140, "max_reading": 140}
    ]
    assert body["candidate_count"] == 1
    assert body["unique"] is True


def test_missing_reading_wrap_splits_intervals():
    """候选弧跨零点：拆为 [900,999] 与 [0,200] 两个不重叠闭区间。"""
    resp = _missing_reading(_missing_payload(300, start_value=900, end_value=200))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 1  # (300 - (-700)) / 1000
    assert (body["left_usage_min"], body["left_usage_max"]) == (0, 300)
    assert body["candidate_intervals"] == [
        {"min_reading": 900, "max_reading": 999},
        {"min_reading": 0, "max_reading": 200},
    ]
    assert body["candidate_count"] == 301
    assert body["unique"] is False


def test_missing_reading_full_modulus_domain():
    """候选数不少于模数：覆盖全部剩余类，返回全域 [0, M-1]。"""
    resp = _missing_reading(
        _missing_payload(52, rate=100, modulus=10, start_value=3, end_value=5)
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 5
    assert (body["left_usage_min"], body["left_usage_max"]) == (0, 52)
    assert body["candidate_intervals"] == [{"min_reading": 0, "max_reading": 9}]
    assert body["candidate_count"] == 10
    assert body["unique"] is False


def test_missing_reading_modulus_one_always_unique():
    """模数为 1：任何可行账单下候选均为全域 [0,0]，唯一。"""
    resp = _missing_reading(
        _missing_payload(0, modulus=1, start_value=0, end_value=0)
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["candidate_intervals"] == [{"min_reading": 0, "max_reading": 0}]
    assert body["candidate_count"] == 1
    assert body["unique"] is True


def test_missing_reading_zero_usage_flat_readings():
    """起止读数相同且账单为 0：左侧用量恰为 0，候选即首读数。"""
    resp = _missing_reading(_missing_payload(0, start_value=500, end_value=500))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 0
    assert (body["left_usage_min"], body["left_usage_max"]) == (0, 0)
    assert body["candidate_intervals"] == [
        {"min_reading": 500, "max_reading": 500}
    ]
    assert body["unique"] is True


def test_missing_reading_fractional_rate_floors_each_side():
    """0.5/s 两侧分别取整：C_L=floor(20.5)=20、C_R=floor(30.5)=30。

    整段 floor(0.5×102)=51 > 50 = C_L+C_R，验证按两侧各自取整。
    """
    payload = _missing_payload(
        50, rate=0.5, modulus=100, start_value=0, end_value=50,
        missing_ts="2026-09-24T00:00:41Z",
        end_ts="2026-09-24T00:01:42Z",
    )
    resp = _missing_reading(payload)
    body = resp.json()
    assert body["status"] == "ok"
    assert (body["left_capacity"], body["right_capacity"]) == (20, 30)
    # x ∈ [max(0, 50-30), min(20, 50)] = [20, 20]
    assert (body["left_usage_min"], body["left_usage_max"]) == (20, 20)
    assert body["candidate_intervals"] == [
        {"min_reading": 20, "max_reading": 20}
    ]
    assert body["unique"] is True


def test_missing_reading_incompatible_modulus():
    """B 与首尾读数差模 M 不同余：total_incompatible_modulus。"""
    resp = _missing_reading(_missing_payload(101))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["candidate_intervals"] == []
    assert body["candidate_count"] == 0
    assert body["unique"] is False
    assert body["total_wraps"] is None
    assert body["left_usage_min"] is None and body["left_usage_max"] is None
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "total_incompatible_modulus"
    assert conflict["expected_remainder"] == 100
    assert conflict["feasible_max_total"] == 1000  # 400 + 600


def test_missing_reading_insufficient_capacity():
    """账单同余但超出两侧容量之和：insufficient_capacity，与不相容可区分。"""
    resp = _missing_reading(_missing_payload(1100, rate=1))
    body = resp.json()
    assert body["status"] == "conflict"
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "insufficient_capacity"
    assert conflict["expected_remainder"] is None
    assert (conflict["left_capacity"], conflict["right_capacity"]) == (40, 60)
    assert conflict["feasible_max_total"] == 100
    assert body["total_wraps"] == 1  # 同余成立，K 已知
    assert body["candidate_intervals"] == []


def test_missing_reading_timezone_offset_normalized():
    """漏采时刻带 +08:00 偏移：换算 UTC 后参与切分，结果一致。"""
    payload = _missing_payload(100)
    payload["missing_timestamp"] = "2026-09-24T08:00:40+08:00"
    body = _missing_reading(payload).json()
    assert body["status"] == "ok"
    assert body["left_seconds"] == 40
    assert body["candidate_intervals"] == [
        {"min_reading": 100, "max_reading": 200}
    ]


def test_missing_reading_reuses_validation():
    """复用现有校验：漏采时刻越界/缺时区/亚秒、读数越界、负账单等均 422。"""
    base = _missing_payload(100)

    def expect_422(mut):
        import copy

        payload = copy.deepcopy(base)
        mut(payload)
        assert _missing_reading(payload).status_code == 422

    # 漏采时刻不严格介于起止之间：等于首读、等于末读、在账期之外
    expect_422(lambda p: p.update(missing_timestamp="2026-09-24T00:00:00Z"))
    expect_422(lambda p: p.update(missing_timestamp="2026-09-24T00:01:40Z"))
    expect_422(lambda p: p.update(missing_timestamp="2026-09-24T00:02:00Z"))
    # 漏采时刻缺时区、含亚秒
    expect_422(lambda p: p.update(missing_timestamp="2026-09-24T00:00:40"))
    expect_422(lambda p: p.update(missing_timestamp="2026-09-24T00:00:40.5Z"))
    # 读数超出模数
    expect_422(lambda p: p["start"].update(value=1000))
    expect_422(lambda p: p["end"].update(value=1000))
    # 起止读数缺时区、时间不递增
    expect_422(lambda p: p["start"].update(timestamp="2026-09-24T00:00:00"))
    expect_422(lambda p: p["end"].update(timestamp="2026-09-24T00:00:00Z"))
    # 账单为负、非整数
    expect_422(lambda p: p.update(total_usage=-1))
    expect_422(lambda p: p.update(total_usage=12.5))
    # 负速率、模数 < 1
    expect_422(lambda p: p.update(max_rate_per_second=-1))
    expect_422(lambda p: p.update(modulus=0))


# ---------------------------------------------------------------------------
# solve_missing_reading 精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------

def test_missing_reading_exact_by_brute_force():
    """随机小参数下，候选区间/个数/唯一性与枚举全部可行左侧用量一致。"""
    rng = random.Random(20260925)
    checked = 0
    for _ in range(4000):
        modulus = rng.choice([2, 3, 5, 7, 10, 16])
        rate = Fraction(rng.randint(0, 20), rng.randint(1, 3))
        left_seconds = rng.randint(1, 20)
        right_seconds = rng.randint(1, 20)
        r1, r2 = rng.randrange(modulus), rng.randrange(modulus)
        diff = r2 - r1
        c_l = math.floor(rate * left_seconds)
        c_r = math.floor(rate * right_seconds)

        b = diff + rng.randint(0, 3) * modulus
        if rng.random() < 0.3:
            b += rng.choice([-1, 1])  # 破坏同余（模数均 >= 2）
        if b < 0:
            continue

        result = solve_missing_reading(
            r1, r2, left_seconds, right_seconds, modulus, rate, b
        )
        if (b - diff) % modulus != 0:
            assert result.status == "conflict"
            assert result.conflicts == ("total_incompatible_modulus",)
            assert result.total_wraps is None
            continue
        if b > c_l + c_r:
            assert result.status == "conflict"
            assert result.conflicts == ("insufficient_capacity",)
            assert result.total_wraps == (b - diff) // modulus
            continue

        assert result.status == "ok"
        # 暴力枚举可行左侧用量 x：0 <= x <= C_L 且 0 <= B - x <= C_R
        feasible_x = [x for x in range(c_l + 1) if 0 <= b - x <= c_r]
        assert feasible_x
        assert (result.left_usage_min, result.left_usage_max) == (
            feasible_x[0],
            feasible_x[-1],
        )
        expected = sorted({(r1 + x) % modulus for x in feasible_x})
        assert result.candidate_count == len(expected)
        assert result.unique == (len(expected) == 1)
        # 区间并集与期望集合一致、互不重叠、且均落在 [0, M-1] 内
        covered = []
        for lo, hi in result.intervals:
            assert 0 <= lo <= hi <= modulus - 1
            covered.extend(range(lo, hi + 1))
        assert len(covered) == len(set(covered))
        assert sorted(covered) == expected
        checked += 1
    assert checked > 500


# ---------------------------------------------------------------------------
# 最小统一速率上限 /v1/usage/minimum-rate
# ---------------------------------------------------------------------------

def test_minimum_rate_unique_segments():
    """两段、账单 5000：K=5、T=4，R*=29/10，两段圈数被唯一确定为 2、3。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 5000,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 5
    assert body["min_rate_per_second"] == {"numerator": 29, "denominator": 10}
    assert body["unique"] is True
    assert body["conflicts"] == []
    s1, s2 = body["segments"]
    assert (s1["wraps"], s1["usage"]) == (2, 2100)
    assert (s2["wraps"], s2["usage"]) == (3, 2900)
    assert s1["min_usage"] is None and s1["max_usage"] is None
    assert s2["min_usage"] is None and s2["max_usage"] is None
    # 各段速率 2100/1000 与 2900/1000，峰值恰为 29/10
    assert Fraction(s1["usage"], s1["elapsed_seconds"]) == Fraction(21, 10)
    assert Fraction(s2["usage"], s2["elapsed_seconds"]) == Fraction(29, 10)


def test_minimum_rate_ambiguous_ranges_not_fabricated():
    """平读数三段（1000s/500s）、账单 2000：R*=2，两段均为范围，不虚构用量。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 2000,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 100},
                {"timestamp": "2026-09-24T00:25:00Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 2
    assert body["min_rate_per_second"] == {"numerator": 2, "denominator": 1}
    assert body["unique"] is False
    s1, s2 = body["segments"]
    assert (s1["min_wraps"], s1["max_wraps"]) == (1, 2)
    assert (s1["min_usage"], s1["max_usage"]) == (1000, 2000)
    assert s1["wraps"] is None and s1["usage"] is None
    assert (s2["min_wraps"], s2["max_wraps"]) == (0, 1)
    assert (s2["min_usage"], s2["max_usage"]) == (0, 1000)
    # 可行组合 (k1,k2) ∈ {(1,1),(2,0)}，任一组合两段用量之和恒为 2000
    assert s1["min_usage"] + s2["max_usage"] == 2000
    assert s1["max_usage"] + s2["min_usage"] == 2000


def test_minimum_rate_single_segment_reduced_fraction():
    """单段：速率即 账单/历时，按约分返回（500/60 -> 25/3）。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 500,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:01:00Z", "value": 600},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["min_rate_per_second"] == {"numerator": 25, "denominator": 3}
    assert body["unique"] is True
    seg = body["segments"][0]
    assert (seg["wraps"], seg["usage"]) == (0, 500)


def test_minimum_rate_zero_extra_wraps():
    """账单恰为最小可行总量（T=0）：R* 为各段最小用量的峰值速率，全部唯一。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 200,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 900},
                {"timestamp": "2026-09-24T00:01:40Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 1
    assert body["min_rate_per_second"] == {"numerator": 2, "denominator": 1}
    seg = body["segments"][0]
    assert (seg["wraps"], seg["usage"]) == (1, 200)


def test_minimum_rate_modulus_one():
    """模数为 1：读数恒为 0，账单 5 按历时 2s/3s 分配，R*=1，各段唯一。"""
    resp = _minimum_rate(
        {
            "modulus": 1,
            "total_usage": 5,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 0},
                {"timestamp": "2026-09-24T00:00:02Z", "value": 0},
                {"timestamp": "2026-09-24T00:00:05Z", "value": 0},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["min_rate_per_second"] == {"numerator": 1, "denominator": 1}
    assert [s["usage"] for s in body["segments"]] == [2, 3]
    assert body["unique"] is True


def test_minimum_rate_incompatible_modulus():
    """账单与首尾读数差模 M 不同余：total_incompatible_modulus。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 5500,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
                {"timestamp": "2026-09-24T00:16:40Z", "value": 200},
                {"timestamp": "2026-09-24T00:33:20Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "conflict"
    assert body["segments"] == []
    assert body["min_rate_per_second"] is None
    assert body["total_wraps"] is None
    assert body["unique"] is False
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_incompatible_modulus"
    assert conflict["expected_remainder"] == 0
    assert conflict["feasible_min_total"] == 1000
    assert conflict["feasible_max_total"] is None


def test_minimum_rate_bill_below_feasible_minimum():
    """账单同余但低于最小可行总量 1200：total_out_of_range，上界为 null。"""
    resp = _minimum_rate(
        {
            "modulus": 1000,
            "total_usage": 200,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 900},
                {"timestamp": "2026-09-24T00:00:10Z", "value": 100},
                {"timestamp": "2026-09-24T00:00:20Z", "value": 900},
                {"timestamp": "2026-09-24T00:00:30Z", "value": 100},
            ],
        }
    )
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["segments"] == []
    assert body["min_rate_per_second"] is None
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "total_out_of_range"
    assert conflict["feasible_min_total"] == 1200
    assert conflict["feasible_max_total"] is None
    assert conflict["expected_remainder"] is None
    assert body["total_wraps"] == 1  # 同余成立，K 已知


def test_minimum_rate_reuses_reading_validation():
    """复用现有读数校验：越界/不递增/缺时区/亚秒/读数不足/负账单等均 422。"""
    base = {
        "modulus": 1000,
        "total_usage": 100,
        "readings": [
            {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
            {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
        ],
    }

    def expect_422(mut):
        import copy

        payload = copy.deepcopy(base)
        mut(payload)
        assert _minimum_rate(payload).status_code == 422

    # 读数越界
    expect_422(lambda p: p["readings"][1].update(value=1000))
    # 时间戳不递增
    expect_422(lambda p: p["readings"][1].update(timestamp="2026-09-24T00:00:00Z"))
    # 缺时区
    expect_422(lambda p: p["readings"][0].update(timestamp="2026-09-24T00:00:00"))
    # 亚秒
    expect_422(lambda p: p["readings"][0].update(timestamp="2026-09-24T00:00:00.5Z"))
    # 读数不足 2 条
    expect_422(lambda p: p.update(readings=p["readings"][:1]))
    # 账单为负、非整数
    expect_422(lambda p: p.update(total_usage=-1))
    expect_422(lambda p: p.update(total_usage=12.5))
    # 模数 < 1
    expect_422(lambda p: p.update(modulus=0))


# ---------------------------------------------------------------------------
# solve_minimum_rate 精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------

def _compositions(total, parts):
    """把 total 拆成 parts 个非负整数之和的全部组合。"""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def test_minimum_rate_exact_by_brute_force():
    """随机小参数下，最小速率与各段圈数范围与枚举全部圈数组合一致。"""
    rng = random.Random(20260925)
    checked = 0
    for _ in range(3000):
        modulus = rng.choice([2, 3, 5, 7, 10])
        n = rng.randint(1, 4)
        values = [rng.randrange(modulus) for _ in range(n + 1)]
        elapsed = [rng.randint(1, 8) for _ in range(n)]
        diffs = [values[i + 1] - values[i] for i in range(n)]
        k_mins = [0 if d >= 0 else 1 for d in diffs]
        bases = [d + k * modulus for d, k in zip(diffs, k_mins)]
        extra = rng.randint(0, 5)
        total = sum(bases) + extra * modulus  # 同余且可行的账单

        result = solve_minimum_rate(diffs, elapsed, total, modulus)
        assert result.status == "ok"
        assert result.total_wraps == extra + sum(k_mins)

        # 暴力枚举全部圈数组合 t_i >= 0、Σt = extra，取峰值速率最小值
        best = None
        k_sets = [set() for _ in range(n)]
        for ts in _compositions(extra, n):
            peak = max(
                Fraction(bases[i] + ts[i] * modulus, elapsed[i])
                for i in range(n)
            )
            if best is None or peak < best:
                best = peak
                k_sets = [set() for _ in range(n)]
            if peak == best:
                for i in range(n):
                    k_sets[i].add(k_mins[i] + ts[i])

        assert result.min_rate == best
        # 返回值已约分
        assert result.min_rate.denominator > 0
        assert math.gcd(result.min_rate.numerator, result.min_rate.denominator) == 1
        for i, seg in enumerate(result.segments):
            assert (seg.min_wraps, seg.max_wraps) == (min(k_sets[i]), max(k_sets[i]))
            assert seg.unique == (len(k_sets[i]) == 1)
            if seg.unique:
                assert seg.usage(modulus) == diffs[i] + seg.min_wraps * modulus
        checked += 1
    assert checked > 500


def test_minimum_rate_conflicts_by_brute_force():
    """不相容与低于最小可行总量的账单均给出可区分的冲突原因。"""
    rng = random.Random(20260925)
    for _ in range(2000):
        modulus = rng.choice([2, 3, 5, 7, 10])
        n = rng.randint(1, 3)
        values = [rng.randrange(modulus) for _ in range(n + 1)]
        elapsed = [rng.randint(1, 8) for _ in range(n)]
        diffs = [values[i + 1] - values[i] for i in range(n)]
        k_mins = [0 if d >= 0 else 1 for d in diffs]
        bases = [d + k * modulus for d, k in zip(diffs, k_mins)]
        min_total = sum(bases)

        # 破坏同余：modulus >= 2 时 +1 必不同余
        bad = min_total + rng.randint(0, 3) * modulus + 1
        r = solve_minimum_rate(diffs, elapsed, bad, modulus)
        assert r.status == "conflict"
        assert r.conflicts == ("total_incompatible_modulus",)
        assert r.total_wraps is None
        assert r.min_rate is None
        assert r.feasible_min_total == min_total

        # 同余但低于最小可行总量
        if min_total - modulus >= 0:
            low = min_total - modulus
            r = solve_minimum_rate(diffs, elapsed, low, modulus)
            assert r.status == "conflict"
            assert r.conflicts == ("total_out_of_range",)
            assert r.total_wraps == (low - sum(diffs)) // modulus
            assert r.min_rate is None


# ---------------------------------------------------------------------------
# 双表同段交叉核验 /v1/usage/dual-meter/compare
# ---------------------------------------------------------------------------

def _dual_payload(a_rate=10, b_rate=10, a_mod=1000, b_mod=1500,
                  a_values=(900, 100), b_values=(1200, 400),
                  start="2026-09-24T00:00:00Z", end="2026-09-24T00:16:40Z"):
    """默认共同时段 1000s：表 A 差 -800（可行 [200,9200]），表 B 差 -800（[700,9700]）。"""
    return {
        "meter_a": {
            "modulus": a_mod,
            "max_rate_per_second": a_rate,
            "start": {"timestamp": start, "value": a_values[0]},
            "end": {"timestamp": end, "value": a_values[1]},
        },
        "meter_b": {
            "modulus": b_mod,
            "max_rate_per_second": b_rate,
            "start": {"timestamp": start, "value": b_values[0]},
            "end": {"timestamp": end, "value": b_values[1]},
        },
    }


def test_dual_meter_multiple_candidates():
    """共同候选 2200/5200/8200（步长 lcm=3000）：最小/最大/候选数由端点给出。"""
    resp = _dual_meter(_dual_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["conflicts"] == []
    assert body["elapsed_seconds"] == 1000
    assert body["modulus_gcd"] == 500
    assert body["usage_step"] == 3000
    assert (body["min_usage"], body["max_usage"]) == (2200, 8200)
    assert body["candidate_count"] == 3
    assert body["unique"] is False
    # 候选为等差数列 min + i*step，不逐个枚举
    assert [
        body["min_usage"] + i * body["usage_step"]
        for i in range(body["candidate_count"])
    ] == [2200, 5200, 8200]
    a, b = body["meters"]
    assert a["meter"] == "a" and b["meter"] == "b"
    assert (a["min_wraps"], a["max_wraps"]) == (1, 10)
    assert (a["min_usage"], a["max_usage"]) == (200, 9200)
    assert (b["min_wraps"], b["max_wraps"]) == (1, 7)
    assert (b["min_usage"], b["max_usage"]) == (700, 9700)


def test_dual_meter_unique_candidate():
    """表 B 速率 5 收窄其可行范围为 [700,3700]：交集内仅剩候选 2200。"""
    resp = _dual_meter(_dual_payload(b_rate=5))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["candidate_count"] == 1
    assert body["unique"] is True
    assert body["min_usage"] == body["max_usage"] == 2200


def test_dual_meter_same_modulus():
    """两表模数相同（退化为统一模数）：步长即模数，候选 200..9200 共 10 个。"""
    resp = _dual_meter(_dual_payload(b_mod=1000, b_values=(200, 400)))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["modulus_gcd"] == 1000
    assert body["usage_step"] == 1000
    assert (body["min_usage"], body["max_usage"]) == (200, 9200)
    assert body["candidate_count"] == 10
    assert body["unique"] is False


def test_dual_meter_zero_usage_consistent():
    """两表平读数且速率为 0：共同用量唯一为 0。"""
    resp = _dual_meter(
        _dual_payload(a_rate=0, b_rate=0, a_values=(42, 42), b_values=(7, 7))
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert (body["min_usage"], body["max_usage"]) == (0, 0)
    assert body["candidate_count"] == 1
    assert body["unique"] is True


def test_dual_meter_incompatible_modulus():
    """读数差 -800 与 -700 模 gcd=500 不同余：incompatible_modulus。"""
    resp = _dual_meter(_dual_payload(b_values=(1200, 500)))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["meters"] == []
    assert body["min_usage"] is None and body["max_usage"] is None
    assert body["candidate_count"] == 0
    assert body["unique"] is False
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "common"
    assert conflict["reason"] == "incompatible_modulus"
    assert conflict["modulus_gcd"] == 500
    assert conflict["expected_remainder"] == 200  # -800 mod 500
    assert (conflict["a_usage_min"], conflict["a_usage_max"]) == (200, 9200)
    assert (conflict["b_usage_min"], conflict["b_usage_max"]) == (800, 9800)


def test_dual_meter_range_disjoint_empty_intersection():
    """同余相容但表 A 可行 [200,200]、表 B [700,9700]：区间交集为空。"""
    resp = _dual_meter(_dual_payload(a_rate=1))
    body = resp.json()
    assert body["status"] == "conflict"
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "common"
    assert conflict["reason"] == "range_disjoint"
    assert conflict["expected_remainder"] is None
    assert conflict["intersect_min"] > conflict["intersect_max"]
    assert "交集为空" in conflict["detail"]


def test_dual_meter_range_disjoint_lattice_miss():
    """区间相交 [700,700] 但无同余点（最近候选 2200）：仍为 range_disjoint。"""
    resp = _dual_meter(_dual_payload(b_rate=2))
    body = resp.json()
    assert body["status"] == "conflict"
    conflict = body["conflicts"][0]
    assert conflict["reason"] == "range_disjoint"
    assert (conflict["intersect_min"], conflict["intersect_max"]) == (700, 700)
    assert "2200" in conflict["detail"]
    # 与区间本身不交的情形 detail 可区分
    assert "交集为空" not in conflict["detail"]


def test_dual_meter_segment_no_solution():
    """表 A 段本身无解（非负增量已超速率上限）：分段冲突，不做联立。"""
    resp = _dual_meter(_dual_payload(a_rate=0.1))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["meters"] == []
    assert len(body["conflicts"]) == 1
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "segment"
    assert conflict["meter"] == "a"
    assert conflict["reason"] == "no_solution"
    assert conflict["max_wraps"] < conflict["min_wraps"]


def test_dual_meter_both_segments_no_solution_listed():
    """两表段都无解时一并列出。"""
    resp = _dual_meter(_dual_payload(a_rate=0.1, b_rate=0.1))
    body = resp.json()
    assert body["status"] == "conflict"
    assert [c["meter"] for c in body["conflicts"]] == ["a", "b"]
    assert all(c["reason"] == "no_solution" for c in body["conflicts"])


def test_dual_meter_period_must_match():
    """两表起止时刻不分别相同：422。"""
    payload = _dual_payload()
    payload["meter_b"]["start"]["timestamp"] = "2026-09-24T00:00:01Z"
    assert _dual_meter(payload).status_code == 422

    payload = _dual_payload()
    payload["meter_b"]["end"]["timestamp"] = "2026-09-24T00:16:41Z"
    assert _dual_meter(payload).status_code == 422


def test_dual_meter_timezone_normalized():
    """表 B 起止时刻以 +08:00 表达同一时刻：换算 UTC 后通过，结果一致。"""
    payload = _dual_payload()
    payload["meter_b"]["start"]["timestamp"] = "2026-09-24T08:00:00+08:00"
    payload["meter_b"]["end"]["timestamp"] = "2026-09-24T08:16:40+08:00"
    resp = _dual_meter(payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert (body["min_usage"], body["max_usage"]) == (2200, 8200)
    assert body["candidate_count"] == 3


def test_dual_meter_reuses_validation():
    """复用现有校验：读数越界、时间不递增、缺时区、亚秒、负速率等均 422。"""
    base = _dual_payload()

    def expect_422(mut):
        import copy

        payload = copy.deepcopy(base)
        mut(payload)
        assert _dual_meter(payload).status_code == 422

    # 表 A 读数越界
    expect_422(lambda p: p["meter_a"]["start"].update(value=1000))
    # 表 B 读数越界（M=1500）
    expect_422(lambda p: p["meter_b"]["end"].update(value=1500))
    # 段内时间不递增
    expect_422(
        lambda p: p["meter_a"]["start"].update(timestamp="2026-09-24T00:16:40Z")
    )
    # 缺时区
    expect_422(
        lambda p: p["meter_b"]["start"].update(timestamp="2026-09-24T00:00:00")
    )
    # 亚秒
    expect_422(
        lambda p: p["meter_a"]["end"].update(timestamp="2026-09-24T00:16:40.5Z")
    )
    # 负速率（各自表独立校验）
    expect_422(lambda p: p["meter_b"].update(max_rate_per_second=-1))
    # 模数 < 1
    expect_422(lambda p: p["meter_a"].update(modulus=0))


# ---------------------------------------------------------------------------
# compare_dual_meter 精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------

def test_compare_dual_meter_exact_by_brute_force():
    """随机小参数下，共同候选区间/个数/唯一性与枚举全部 (k1,k2) 一致。"""
    rng = random.Random(20260925)
    checked = 0
    for _ in range(4000):
        m1 = rng.choice([2, 3, 4, 5, 6, 7])
        m2 = rng.choice([2, 3, 4, 5, 6, 7, 9])
        d1 = rng.randrange(m1) - rng.randrange(m1)
        d2 = rng.randrange(m2) - rng.randrange(m2)
        lo1 = 0 if d1 >= 0 else math.ceil(-d1 / m1)
        lo2 = 0 if d2 >= 0 else math.ceil(-d2 / m2)
        hi1 = lo1 + rng.randint(0, 5)
        hi2 = lo2 + rng.randint(0, 5)

        result = compare_dual_meter(d1, (lo1, hi1), m1, d2, (lo2, hi2), m2)

        # 暴力：两表各自可行用量集合的交集即共同候选
        common = sorted(
            {d1 + k1 * m1 for k1 in range(lo1, hi1 + 1)}
            & {d2 + k2 * m2 for k2 in range(lo2, hi2 + 1)}
        )
        g = math.gcd(m1, m2)
        assert result.gcd == g
        assert result.step == m1 // g * m2
        assert (result.a_usage_min, result.a_usage_max) == (
            d1 + lo1 * m1,
            d1 + hi1 * m1,
        )
        assert (result.b_usage_min, result.b_usage_max) == (
            d2 + lo2 * m2,
            d2 + hi2 * m2,
        )
        if not common:
            assert result.status == "conflict"
            assert result.candidate_count == 0
            assert result.min_usage is None and result.max_usage is None
            assert result.unique is False
            # 两种冲突原因可区分：同余不相容 vs 范围无交集
            if (d1 - d2) % g != 0:
                assert result.conflicts == ("incompatible_modulus",)
                assert result.residue is None
            else:
                assert result.conflicts == ("range_disjoint",)
                assert result.residue is not None
        else:
            checked += 1
            assert result.status == "ok"
            assert result.conflicts == ()
            assert (result.min_usage, result.max_usage) == (
                common[0],
                common[-1],
            )
            assert result.candidate_count == len(common)
            assert result.unique == (len(common) == 1)
            # 候选构成公差为 lcm 的等差数列
            assert all(
                u == common[0] + i * result.step
                for i, u in enumerate(common)
            )
            assert common[0] % result.step == result.residue
    assert checked > 500


# ---------------------------------------------------------------------------
# 停水时段账单复核 /v1/usage/outage-reconcile
# ---------------------------------------------------------------------------


def _outage_payload(
    total_usage,
    rate=30,
    values=(0, 500, 100),
    times=("2026-09-24T00:00:00Z", "2026-09-24T00:00:30Z", "2026-09-24T00:01:00Z"),
    outages=None,
    modulus=1000,
):
    if outages is None:
        outages = [
            {"start": "2026-09-24T00:00:10Z", "end": "2026-09-24T00:00:20Z"},
            {"start": "2026-09-24T00:00:40Z", "end": "2026-09-24T00:00:50Z"},
        ]
    return {
        "modulus": modulus,
        "max_rate_per_second": rate,
        "total_usage": total_usage,
        "readings": [
            {"timestamp": t, "value": v} for t, v in zip(times, values)
        ],
        "outages": outages,
    }


def test_outage_reconcile_all_unique():
    """两段各有 10s 停水：供水片段 10s+10s，速率 30 时容量各 600，圈数均唯一。"""
    resp = _outage(_outage_payload(total_usage=1100, rate=30))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["conflicts"] == []
    assert body["total_wraps"] == 1
    assert len(body["segments"]) == 2
    for i, seg in enumerate(body["segments"]):
        assert seg["elapsed_seconds"] == 30
        assert seg["outage_seconds"] == 10
        assert seg["supply_seconds"] == 20
        assert seg["total_capacity"] == 600  # floor(30×10) + floor(30×10)
        assert seg["unique"] is True
        assert [f["seconds"] for f in seg["water_fragments"]] == [10, 10]
        assert [f["capacity"] for f in seg["water_fragments"]] == [300, 300]
        assert seg["usage"] == (500 if i == 0 else 600)
        assert seg["wraps"] == (0 if i == 0 else 1)
        assert seg["min_usage"] is None and seg["max_usage"] is None
    # 唯一用量之和恒等于账单
    assert sum(s["usage"] for s in body["segments"]) == 1100
    # 合并后的停水时段按开始时刻升序回显
    assert [(o["start_timestamp"], o["end_timestamp"], o["seconds"]) for o in body["outages"]] == [
        ("2026-09-24T00:00:10Z", "2026-09-24T00:00:20Z", 10),
        ("2026-09-24T00:00:40Z", "2026-09-24T00:00:50Z", 10),
    ]


def test_outage_reconcile_ambiguous_ranges():
    """容量足够大时账单只收紧圈数范围：不唯一的段只给范围，且此消彼长和为账单。"""
    resp = _outage(_outage_payload(total_usage=2100, rate=200))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["total_wraps"] == 2
    seg0, seg1 = body["segments"]
    assert seg0["total_capacity"] == seg1["total_capacity"] == 4000
    # seg0: diff=500, k∈[0,3]；seg1: diff=-400, k∈[1,3]；K=2 收紧后
    assert (seg0["min_wraps"], seg0["max_wraps"]) == (0, 1)
    assert (seg1["min_wraps"], seg1["max_wraps"]) == (1, 2)
    assert seg0["unique"] is False and seg1["unique"] is False
    assert seg0["wraps"] is None and seg0["usage"] is None
    assert (seg0["min_usage"], seg0["max_usage"]) == (500, 1500)
    assert (seg1["min_usage"], seg1["max_usage"]) == (600, 1600)
    # 每个可行组合两段之和恒为账单（交叉端点相等）
    assert seg0["min_usage"] + seg1["max_usage"] == 2100
    assert seg0["max_usage"] + seg1["min_usage"] == 2100


def test_outage_separate_floor_no_merge_across_outage():
    """停水两侧分别 floor：0.5/s × 两个 3s 片段容量各为 1（和为 2），用量 3 无解；
    不停水时整段 floor(0.5×8)=4 可容纳 3——不能跨停水时段合并取整（若合并取整，
    6 个供水秒容量为 floor(3)=3 即不冲突）。另校验零时长停水时段本身被请求
    校验拒绝（422）。"""
    # 零时长停水（端点相同）非法
    bad = {
        "modulus": 1000,
        "max_rate_per_second": 0.5,
        "total_usage": 3,
        "readings": [
            {"timestamp": "2026-09-24T00:00:00Z", "value": 0},
            {"timestamp": "2026-09-24T00:00:08Z", "value": 3},
        ],
        "outages": [
            {"start": "2026-09-24T00:00:03Z", "end": "2026-09-24T00:00:03Z"}
        ],
    }
    assert _outage(bad).status_code == 422

    # 停水 2s（00:03~00:05）：供水片段 3s+3s -> 容量 1+1=2 < 3
    payload = dict(bad)
    payload["outages"] = [
        {"start": "2026-09-24T00:00:03Z", "end": "2026-09-24T00:00:05Z"}
    ]
    body = _outage(payload).json()
    assert body["status"] == "conflict"
    conflict = body["conflicts"][0]
    assert conflict["kind"] == "segment"
    assert conflict["reason"] == "no_solution"
    assert conflict["total_capacity"] == 2
    assert conflict["supply_seconds"] == 6
    assert [f["seconds"] for f in conflict["water_fragments"]] == [3, 3]
    assert [f["capacity"] for f in conflict["water_fragments"]] == [1, 1]

    # 无停水：整段 8s 容量 floor(4)=4，用量 3 可行
    body = _outage(dict(payload, outages=[])).json()
    assert body["status"] == "ok"
    assert body["segments"][0]["total_capacity"] == 4
    assert body["segments"][0]["usage"] == 3
    assert body["outages"] == []


def test_outage_reconcile_segment_no_solution():
    """扣除停水后容量 200，连最小非负增量都放不下：分段冲突，不做总量联立。"""
    resp = _outage(_outage_payload(total_usage=1200, rate=10))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["total_wraps"] is None
    assert body["segments"] == []
    assert [c["index"] for c in body["conflicts"]] == [0, 1]
    c0, c1 = body["conflicts"]
    assert c0["kind"] == "segment" and c0["reason"] == "no_solution"
    assert c0["total_capacity"] == 200  # floor(10×10)×2
    assert c0["min_wraps"] == 0 and c0["max_wraps"] == -1
    assert c1["min_wraps"] == 1 and c1["max_wraps"] == 0
    assert "不能跨停水时段合并取整" in c0["detail"]
    # 冲突时仍回显裁剪合并后的停水时段与各供水片段
    assert len(body["outages"]) == 2
    assert [f["seconds"] for f in c0["water_fragments"]] == [10, 10]


def test_outage_reconcile_incompatible_modulus():
    """停水不改变模 M 同余：账单余数不符为 total_incompatible_modulus。"""
    resp = _outage(_outage_payload(total_usage=2150, rate=200))
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["segments"] == []
    (conflict,) = body["conflicts"]
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_incompatible_modulus"
    assert conflict["expected_remainder"] == 100  # 首尾读数差 100
    assert conflict["total_wraps"] is None
    # 停水后的可行总量区间也一并给出
    assert (conflict["feasible_min_total"], conflict["feasible_max_total"]) == (
        1100, 7100,
    )


def test_outage_reconcile_out_of_range():
    """账单同余但超出停水后可行总量：total_out_of_range，给出停水后的可行盒。"""
    # 高于上界
    body = _outage(_outage_payload(total_usage=8100, rate=200)).json()
    assert body["status"] == "conflict"
    (conflict,) = body["conflicts"]
    assert conflict["kind"] == "total"
    assert conflict["reason"] == "total_out_of_range"
    assert conflict["total_wraps"] == 8  # 同余成立，K 已知
    assert (conflict["feasible_min_total"], conflict["feasible_max_total"]) == (
        1100, 7100,
    )
    # 低于下界：最小圈数和 0+1=1 -> 1100；账单 100 对应 K=0
    body = _outage(_outage_payload(total_usage=100, rate=200)).json()
    assert body["status"] == "conflict"
    (conflict,) = body["conflicts"]
    assert conflict["reason"] == "total_out_of_range"
    assert conflict["total_wraps"] == 0


def test_outage_clip_and_merge_windows():
    """账期外裁剪、重叠/相接合并、完全在外丢弃，结果按开始时刻升序回显。"""
    body = _outage(
        {
            "modulus": 1000,
            "max_rate_per_second": 0,
            "total_usage": 0,
            "readings": [
                {"timestamp": "2026-09-24T00:00:00Z", "value": 10},
                {"timestamp": "2026-09-24T00:01:00Z", "value": 10},
            ],
            "outages": [
                # 起点在账期外：裁剪为 [00:00, 00:10)
                {"start": "2026-09-23T00:00:00Z", "end": "2026-09-24T00:00:10Z"},
                # 与上一时段端点相接：合并
                {"start": "2026-09-24T00:00:10Z", "end": "2026-09-24T00:00:20Z"},
                # 重叠：并入
                {"start": "2026-09-24T00:00:15Z", "end": "2026-09-24T00:00:25Z"},
                # 终点在账期外：裁剪为 [00:00:40, 01:00)
                {"start": "2026-09-24T00:00:40Z", "end": "2026-09-25T00:00:00Z"},
                # 完全在账期外：丢弃
                {"start": "2026-09-26T00:00:00Z", "end": "2026-09-26T01:00:00Z"},
            ],
        }
    ).json()
    assert body["status"] == "ok"
    assert [(o["start_timestamp"], o["end_timestamp"], o["seconds"]) for o in body["outages"]] == [
        ("2026-09-24T00:00:00Z", "2026-09-24T00:00:25Z", 25),
        ("2026-09-24T00:00:40Z", "2026-09-24T00:01:00Z", 20),
    ]
    (seg,) = body["segments"]
    assert seg["outage_seconds"] == 45
    assert seg["supply_seconds"] == 15
    assert [(f["start_timestamp"], f["end_timestamp"], f["seconds"]) for f in seg["water_fragments"]] == [
        ("2026-09-24T00:00:25Z", "2026-09-24T00:00:40Z", 15)
    ]


def test_outage_half_open_boundaries():
    """左闭右开：与读数端点相接的停水不产生重复扣除，供水片段秒数之和恰为供水秒数。"""
    # 停水 [00:00:00, 00:00:10) 从区间起点相接；另一个 [00:00:20, 00:00:30) 与末点相接
    body = _outage(
        _outage_payload(
            total_usage=1000,
            rate=100,
            values=(0, 0),
            times=("2026-09-24T00:00:00Z", "2026-09-24T00:00:30Z"),
            outages=[
                {"start": "2026-09-24T00:00:00Z", "end": "2026-09-24T00:00:10Z"},
                {"start": "2026-09-24T00:00:20Z", "end": "2026-09-24T00:00:30Z"},
            ],
        )
    ).json()
    assert body["status"] == "ok"
    (seg,) = body["segments"]
    assert seg["outage_seconds"] == 20
    assert seg["supply_seconds"] == 10
    assert [(f["start_timestamp"], f["end_timestamp"]) for f in seg["water_fragments"]] == [
        ("2026-09-24T00:00:10Z", "2026-09-24T00:00:20Z")
    ]
    assert sum(f["seconds"] for f in seg["water_fragments"]) == seg["supply_seconds"]


def test_outage_no_outages_matches_reconcile():
    """不给停水时段时，结果与 /v1/usage/reconcile 完全一致。"""
    payload = _outage_payload(total_usage=5000, rate=500, outages=[])
    payload["readings"] = [
        {"timestamp": "2026-09-24T00:00:00Z", "value": 100},
        {"timestamp": "2026-09-24T00:00:10Z", "value": 200},
        {"timestamp": "2026-09-24T00:00:20Z", "value": 100},
    ]
    outage_body = _outage(payload).json()
    reconcile_body = _reconcile(
        {k: v for k, v in payload.items() if k != "outages"}
    ).json()
    assert outage_body["status"] == reconcile_body["status"] == "ok"
    assert outage_body["total_wraps"] == reconcile_body["total_wraps"]
    for a, b in zip(outage_body["segments"], reconcile_body["segments"]):
        assert a["min_wraps"] == b["min_wraps"]
        assert a["max_wraps"] == b["max_wraps"]
        assert a["wraps"] == b["wraps"]
        assert a["usage"] == b["usage"]
        assert a["min_usage"] == b["min_usage"]
        assert a["max_usage"] == b["max_usage"]
        assert a["outage_seconds"] == 0
        assert a["supply_seconds"] == a["elapsed_seconds"]
        assert a["total_capacity"] == 5000  # floor(500×10)
        # 无停水时只有一个覆盖全段的供水片段
        assert len(a["water_fragments"]) == 1
        assert a["water_fragments"][0]["seconds"] == a["elapsed_seconds"]


def test_outage_full_interval_outage():
    """整段停水：无供水片段、容量 0；平读数唯一用量 0，回绕读数则无解。"""
    flat = _outage(
        _outage_payload(
            total_usage=0, rate=100, values=(5, 5),
            times=("2026-09-24T00:00:00Z", "2026-09-24T00:00:10Z"),
            outages=[
                {"start": "2026-09-24T00:00:00Z", "end": "2026-09-24T00:00:10Z"}
            ],
        )
    ).json()
    assert flat["status"] == "ok"
    (seg,) = flat["segments"]
    assert seg["supply_seconds"] == 0
    assert seg["total_capacity"] == 0
    assert seg["water_fragments"] == []
    assert seg["unique"] is True and seg["usage"] == 0

    wrap = _outage(
        _outage_payload(
            total_usage=900, rate=100, values=(950, 850),
            times=("2026-09-24T00:00:00Z", "2026-09-24T00:00:10Z"),
            outages=[
                {"start": "2026-09-24T00:00:00Z", "end": "2026-09-24T00:00:10Z"}
            ],
        )
    ).json()
    assert wrap["status"] == "conflict"
    (conflict,) = wrap["conflicts"]
    assert conflict["reason"] == "no_solution"
    assert conflict["total_capacity"] == 0
    assert conflict["water_fragments"] == []
    assert (conflict["min_wraps"], conflict["max_wraps"]) == (1, 0)


def test_outage_window_timezone_normalized():
    """停水时段以 +08:00 表达同一时刻：换算 UTC 后裁剪合并，结果一致。"""
    payload = _outage_payload(total_usage=1100, rate=30)
    payload["outages"] = [
        {"start": "2026-09-24T08:00:10+08:00", "end": "2026-09-24T08:00:20+08:00"},
        {"start": "2026-09-24T08:00:40+08:00", "end": "2026-09-24T08:00:50+08:00"},
    ]
    body = _outage(payload).json()
    assert body["status"] == "ok"
    assert [o["start_timestamp"] for o in body["outages"]] == [
        "2026-09-24T00:00:10Z",
        "2026-09-24T00:00:40Z",
    ]
    assert all(s["supply_seconds"] == 20 for s in body["segments"])


def test_outage_reuses_validation():
    """停水接口复用读数/模数/速率校验，停水时段本身也校验：非法输入一律 422。"""
    base = _outage_payload(total_usage=1100)

    # 读数越界
    bad = copy.deepcopy(base)
    bad["readings"][0]["value"] = 1000
    assert _outage(bad).status_code == 422
    # 时间戳不递增
    bad = copy.deepcopy(base)
    bad["readings"][1]["timestamp"] = bad["readings"][0]["timestamp"]
    assert _outage(bad).status_code == 422
    # 读数缺时区
    bad = copy.deepcopy(base)
    bad["readings"][0]["timestamp"] = "2026-09-24T00:00:00"
    assert _outage(bad).status_code == 422
    # 亚秒
    bad = copy.deepcopy(base)
    bad["readings"][0]["timestamp"] = "2026-09-24T00:00:00.5Z"
    assert _outage(bad).status_code == 422
    # 负速率
    assert _outage({**base, "max_rate_per_second": -1}).status_code == 422
    # 账单为负
    assert _outage({**base, "total_usage": -1}).status_code == 422
    # 停水时段缺时区 / 亚秒 / 反向 / 零时长
    for mut_window in (
        lambda w: w.update(start="2026-09-24T00:00:10"),
        lambda w: w.update(end="2026-09-24T00:00:20.5Z"),
        lambda w: (
            w.update(start="2026-09-24T00:00:20Z"),
            w.update(end="2026-09-24T00:00:10Z"),
        ),
        lambda w: w.update(end=w["start"]),
    ):
        bad = copy.deepcopy(base)
        mut_window(bad["outages"][0])
        assert _outage(bad).status_code == 422


# ---------------------------------------------------------------------------
# outage 裁剪合并 / 逐段判定的精确性：与全枚举暴力结果比对（固定随机种子）
# ---------------------------------------------------------------------------


def test_outage_segment_exact_by_brute_force():
    """随机停水切分下，容量、片段秒数与可行圈数区间和逐片段整数分配枚举一致。"""
    rng = random.Random(20260925)
    checked = 0
    for _ in range(3000):
        modulus = rng.choice([3, 5, 7, 10])
        rate = Fraction(rng.randint(0, 10), rng.randint(1, 3))
        total_secs = rng.randint(1, 24)
        start = datetime(2026, 9, 24, tzinfo=timezone.utc)
        end = start + timedelta(seconds=total_secs)

        # 随机生成若干停水区间（允许账期外、重叠、相接）
        outages = []
        for _w in range(rng.randint(0, 4)):
            w_start = start + timedelta(seconds=rng.randint(-3, total_secs - 1))
            w_end = w_start + timedelta(seconds=rng.randint(1, 6))
            outages.append((w_start, w_end))
        merged = clip_merge_intervals(outages, start, end)

        # 账期内供水片段
        cursor, expected_secs = start, []
        for o_start, o_end in merged:
            cut_s, cut_e = max(o_start, cursor), min(o_end, end)
            if cut_s < cut_e:
                if cursor < cut_s:
                    expected_secs.append(int((cut_s - cursor).total_seconds()))
                cursor = cut_e
        if cursor < end:
            expected_secs.append(int((end - cursor).total_seconds()))

        caps = [math.floor(rate * s) for s in expected_secs]
        if sum(caps) > 25:
            continue  # 控制枚举规模

        r1, r2 = rng.randrange(modulus), rng.randrange(modulus)
        res = solve_outage_segment(
            0, r1, r2, start, end, modulus, rate, merged
        )
        checked += 1

        assert [f.seconds for f in res.fragments] == expected_secs
        assert [f.capacity for f in res.fragments] == caps
        assert res.supply_seconds == sum(expected_secs)
        assert res.outage_seconds == total_secs - sum(expected_secs)
        assert res.elapsed_seconds == total_secs

        # 暴力：枚举全部 k 与逐片段非负整数分配
        diff = r2 - r1
        feasible_ks = set()
        k = 0
        while True:
            delta = diff + k * modulus
            if delta > sum(caps):
                break
            if delta >= 0:
                spans = [range(min(cj, delta) + 1) for cj in caps]
                if any(sum(xs) == delta for xs in itertools.product(*spans)):
                    feasible_ks.add(k)
            k += 1
        if feasible_ks:
            assert res.feasible is True
            assert (res.min_wraps, res.max_wraps) == (
                min(feasible_ks), max(feasible_ks)
            )
        else:
            assert res.feasible is False
            assert res.max_wraps < res.min_wraps
    assert checked > 800


def test_outage_reconcile_exact_by_brute_force():
    """随机多段+账单下，收紧后各段圈数范围与枚举满足各段容量且总和为账单的组合一致。"""
    rng = random.Random(20260926)
    checked = 0
    for _ in range(1500):
        n = rng.randint(2, 3)
        modulus = rng.choice([3, 5, 7])
        rate = Fraction(rng.randint(0, 6), rng.randint(1, 2))
        t0 = datetime(2026, 9, 24, tzinfo=timezone.utc)

        readings = [(t0, rng.randrange(modulus))]
        seg_bounds, diffs = [], []
        cursor = t0
        outages = []
        for _i in range(n):
            secs = rng.randint(1, 8)
            cursor = cursor + timedelta(seconds=secs)
            next_value = rng.randrange(modulus)
            readings.append((cursor, next_value))

            # 段内随机停水（整数秒边界）：每个相邻间隔随机作为停水/供水
            frag_secs = []
            ncut = rng.randint(0, 2)
            cuts = sorted(rng.sample(range(1, secs), min(ncut, secs - 1))) if secs > 1 else []
            marks = [readings[-2][0]] + [
                readings[-2][0] + timedelta(seconds=x) for x in cuts
            ] + [cursor]
            for j in range(len(marks) - 1):
                gap = int((marks[j + 1] - marks[j]).total_seconds())
                if rng.randrange(2) == 1:
                    outages.append((marks[j], marks[j + 1]))
                else:
                    frag_secs.append(gap)
            cap = sum(math.floor(rate * s) for s in frag_secs)
            diff = next_value - readings[-2][1]
            kmin = 0 if diff >= 0 else (-diff + modulus - 1) // modulus
            kmax = (cap - diff) // modulus
            seg_bounds.append((kmin, kmax))
            diffs.append(diff)

        if any(hi < lo for lo, hi in seg_bounds):
            continue  # 只测全部段本身可行的情形（单段无案由另一测试覆盖）

        merged = clip_merge_intervals(
            outages, readings[0][0], readings[-1][0]
        )
        solved = [
            solve_outage_segment(
                i, readings[i][1], readings[i + 1][1],
                readings[i][0], readings[i + 1][0],
                modulus, rate, merged,
            )
            for i in range(n)
        ]
        assert all(s.feasible for s in solved)

        feasible_totals = set()
        # 账单同余类：以首尾读数差为余数
        remainder = (readings[-1][1] - readings[0][1]) % modulus
        wrap_ranges = [range(lo, hi + 1) for lo, hi in seg_bounds]
        for ks in itertools.product(*wrap_ranges):
            B = sum(d + k * modulus for d, k in zip(diffs, ks))
            feasible_totals.add(B)
        if not feasible_totals:
            continue

        b_lo, b_hi = min(feasible_totals), max(feasible_totals)
        # 取三个账单：下界、上界、中间同余类值
        for B in {b_lo, b_hi, remainder + modulus * rng.randint(0, 6)}:
            res = reconcile_total(diffs, seg_bounds, B, modulus)
            if B < b_lo or B > b_hi:
                assert res.status == "conflict"
                assert res.conflicts == ("total_out_of_range",)
                continue
            if B % modulus != remainder:
                assert res.status == "conflict"
                assert res.conflicts == ("total_incompatible_modulus",)
                continue
            assert res.status == "ok"
            checked += 1
            # 暴力：枚举总和恰为 B 的圈数组合，逐段求收紧后的可达范围
            combos = [
                ks for ks in itertools.product(*wrap_ranges)
                if sum(d + k * modulus for d, k in zip(diffs, ks)) == B
            ]
            assert combos
            for i, ns in enumerate(res.narrowed):
                ks_i = sorted({ks[i] for ks in combos})
                assert (ns.min_wraps, ns.max_wraps) == (ks_i[0], ks_i[-1])
                assert ns.unique == (ks_i[0] == ks_i[-1])
    assert checked > 300
