"""累计水表读数用量复核 API。"""

from __future__ import annotations

import math
from datetime import date as date_cls, datetime, timezone
from decimal import Decimal
from fractions import Fraction
from typing import List, Literal, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core import (
    REASON_INCOMPATIBLE_MODULUS,
    REASON_INSUFFICIENT_CAPACITY,
    REASON_NO_FEASIBLE_ALLOCATION,
    REASON_RANGE_DISJOINT,
    REASON_TOTAL_INCOMPATIBLE_MODULUS,
    REASON_TOTAL_OUT_OF_RANGE,
    STATUS_AMBIGUOUS,
    STATUS_CONFLICT,
    STATUS_NO_SOLUTION,
    STATUS_OK,
    clip_merge_intervals,
    compare_dual_meter,
    reconcile_meter_total,
    reconcile_total,
    solve_daily_segment,
    solve_minimum_rate,
    solve_missing_reading,
    solve_outage_segment,
    solve_segment,
)

app = FastAPI(
    title="Water Meter Usage Verifier",
    version="1.7.0",
    description=(
        "累计水表读数用量复核：判定相邻读数间的整圈回绕次数是否唯一，"
        "唯一时给出分段与总用量；另支持账单总量对账，以账单总用量约束"
        "整条序列的回绕次数总和，仅确定圈数已唯一的分段用量；"
        "还支持按 UTC 自然日给出覆盖时段内每日最小/最大可行整数用量；"
        "并支持换表账单复核：跨旧表/新表两段（模数可不同）以账单总用量"
        "精确联立两段回绕次数，不枚举圈数组合；"
        "以及漏采读数推断：以账单总用量与漏采时刻两侧容量约束候选读数，"
        "给出精确不重叠的候选读数闭区间，不猜测读数；"
        "另支持最小统一速率上限：无需给定速率，按账单总用量反求使各段"
        "用量/历时均不超过它的最小统一速率门槛，以约分后的分子/分母"
        "精确返回，并给出该门槛下各段可行圈数范围与是否唯一；"
        "另支持双表同段交叉核验：两块表同时计量同一用量（起止时刻分别"
        "相同），按两表读数差的模数同余联立共同用量并与两表可行用量范围"
        "求交，返回共同用量最小值、最大值、候选数与唯一性，同余不相容与"
        "范围无交集给出不同原因，不逐个枚举回绕圈数；"
        "还支持停水时段账单复核：停水时段（左闭右开）先裁剪到读数账期、"
        "合并重叠或相接时段，各相邻读数段扣除停水后把剩余连续供水片段"
        "分别 floor(速率×秒数) 取整求和作为段容量（不跨停水合并取整），"
        "再以账单总用量联立收紧各段可行圈数，区分单段无解、账单同余"
        "冲突与总量超界。"
    ),
)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

def _normalize_utc_second(v: datetime) -> datetime:
    """秒精度带时区时间戳归一化为 UTC；Reading 与漏采时刻共用同一校验。"""
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timestamp 必须携带时区信息（UTC 时间戳）")
    if v.microsecond != 0:
        raise ValueError("timestamp 必须为秒精度，不允许亚秒部分")
    return v.astimezone(timezone.utc)


class Reading(BaseModel):
    """一条秒精度 UTC 时间戳及其对应读数。"""

    timestamp: datetime = Field(
        description="秒精度时间戳，须带时区（建议 UTC，如 2026-09-24T00:00:00Z）"
    )
    value: int = Field(ge=0, description="计数器读数，须满足 0 <= value < modulus")

    @field_validator("timestamp")
    @classmethod
    def _normalize_timestamp(cls, v: datetime) -> datetime:
        return _normalize_utc_second(v)


class _BaseReadingsRequest(BaseModel):
    """模数与读数序列的共用校验（不含速率）。"""

    modulus: int = Field(ge=1, description="计数器模数（回绕周期）")
    readings: List[Reading] = Field(
        min_length=2, description="按时间排列的读数序列，至少 2 条"
    )

    @model_validator(mode="after")
    def _check_readings(self) -> "_BaseReadingsRequest":
        for r in self.readings:
            if r.value >= self.modulus:
                raise ValueError(
                    f"读数 {r.value} 超出模数范围 [0, {self.modulus})"
                )
        for prev, cur in zip(self.readings, self.readings[1:]):
            if cur.timestamp <= prev.timestamp:
                raise ValueError(
                    "时间戳必须严格递增："
                    f"{prev.timestamp.isoformat()} 之后出现 {cur.timestamp.isoformat()}"
                )
        return self


class _BaseUsageRequest(_BaseReadingsRequest):
    """读数复核 / 账单对账共用的读数与校验参数。"""

    max_rate_per_second: Decimal = Field(
        ge=0, description="每秒最大计数增量，支持小数（按精确分数计算）"
    )


class VerifyRequest(_BaseUsageRequest):
    """分段用量复核请求。"""


class ReconcileRequest(_BaseUsageRequest):
    """账单总量对账请求。"""

    total_usage: int = Field(
        ge=0, description="账单总用量（非负整数）"
    )


class DailyRangeRequest(_BaseUsageRequest):
    """按 UTC 自然日的每日用量上下界请求（读数/模数/速率校验与其他接口一致）。"""


class MinimumRateRequest(_BaseReadingsRequest):
    """最小统一速率上限请求：模数、读数序列与账单总用量（无需给定速率上限）。

    读数、时间戳、模数的校验与 `/v1/usage/verify` 完全一致（复用同一校验）。
    """

    total_usage: int = Field(
        ge=0, description="账单总用量（非负整数）"
    )


# ---------------------------------------------------------------------------
# 停水时段账单复核请求模型
# ---------------------------------------------------------------------------


class OutageWindow(BaseModel):
    """一个左闭右开的停水时段 [start, end)，端点均为秒精度带时区时间戳。"""

    start: datetime = Field(
        description="停水开始时刻（含），秒精度且须带时区，如 2026-09-24T00:00:10Z"
    )
    end: datetime = Field(
        description="停水结束时刻（不含），秒精度且须带时区；时段可部分或全部"
        "落在读数账期之外，服务端先裁剪到账期再合并重叠或相接时段"
    )

    @field_validator("start", "end")
    @classmethod
    def _normalize_outage_timestamp(cls, v: datetime) -> datetime:
        return _normalize_utc_second(v)

    @model_validator(mode="after")
    def _check_window(self) -> "OutageWindow":
        if self.end <= self.start:
            raise ValueError(
                "停水时段必须为左闭右开的非空区间："
                f"start {self.start.isoformat()} 必须早于 end {self.end.isoformat()}"
            )
        return self


class OutageReconcileRequest(_BaseUsageRequest):
    """停水时段账单复核请求：读数、模数、速率、账单总量与若干停水时段。

    读数、时间戳、模数与速率校验与 `/v1/usage/verify` 完全一致（复用同一
    校验）。停水时段允许重叠、相接或部分/全部落在账期之外，服务端统一
    裁剪到首末读数账期后合并，调用方无需预先整理。
    """

    total_usage: int = Field(ge=0, description="账单总用量（非负整数）")
    outages: List[OutageWindow] = Field(
        default_factory=list,
        description="若干左闭右开停水时段；先裁剪到读数账期，再合并重叠或相接"
        "（端点相同）的时段；裁剪后为空（完全在账期外）的时段被丢弃",
    )


# ---------------------------------------------------------------------------
# 换表账单复核请求模型
# ---------------------------------------------------------------------------


class MeterBlock(BaseModel):
    """单块表（旧表或新表）的参数与其段内首尾读数。"""

    modulus: int = Field(ge=1, description="该表计数器模数（回绕周期）")
    max_rate_per_second: Decimal = Field(
        ge=0, description="该表每秒最大计数增量，支持小数（按精确分数计算）"
    )
    start: Reading = Field(description="段内首读数（旧表初读 / 新表初读）")
    end: Reading = Field(description="段内终读数（旧表终读 / 新表终读）")

    @model_validator(mode="after")
    def _check_block(self) -> "MeterBlock":
        for r in (self.start, self.end):
            if r.value >= self.modulus:
                raise ValueError(
                    f"读数 {r.value} 超出模数范围 [0, {self.modulus})"
                )
        if self.end.timestamp <= self.start.timestamp:
            raise ValueError(
                "时间戳必须严格递增："
                f"{self.start.timestamp.isoformat()} 之后出现 "
                f"{self.end.timestamp.isoformat()}"
            )
        return self


class MeterChangeRequest(BaseModel):
    """换表账单复核请求：跨旧表、新表两段，模数与速率可各自不同。"""

    total_usage: int = Field(ge=0, description="跨两表的账单总用量（非负整数）")
    old_meter: MeterBlock = Field(description="旧表参数与起止读数")
    new_meter: MeterBlock = Field(description="新表参数与起止读数")

    @model_validator(mode="after")
    def _check_junction(self) -> "MeterChangeRequest":
        # 换表刻：旧表终读与新表初读必须为同一时刻（读数本身可不同）
        if self.old_meter.end.timestamp != self.new_meter.start.timestamp:
            raise ValueError(
                "旧表终读与新表初读必须为同一时刻（换表刻）：旧表终读 "
                f"{self.old_meter.end.timestamp.isoformat()} 与新表初读 "
                f"{self.new_meter.start.timestamp.isoformat()} 不一致"
            )
        return self


# ---------------------------------------------------------------------------
# 漏采读数推断请求模型
# ---------------------------------------------------------------------------


class MissingReadingRequest(BaseModel):
    """漏采读数推断请求：同一水表的起止读数、漏采时刻与账单总用量。"""

    modulus: int = Field(ge=1, description="计数器模数（回绕周期）")
    max_rate_per_second: Decimal = Field(
        ge=0, description="每秒最大计数增量，支持小数（按精确分数计算）"
    )
    total_usage: int = Field(ge=0, description="该账期账单总用量（非负整数）")
    start: Reading = Field(description="账期首读数")
    end: Reading = Field(description="账期末读数")
    missing_timestamp: datetime = Field(
        description="漏采时刻，须为秒精度带时区时间戳，且严格介于起止时刻之间"
    )

    @field_validator("missing_timestamp")
    @classmethod
    def _normalize_missing_timestamp(cls, v: datetime) -> datetime:
        return _normalize_utc_second(v)

    @model_validator(mode="after")
    def _check_request(self) -> "MissingReadingRequest":
        for r in (self.start, self.end):
            if r.value >= self.modulus:
                raise ValueError(
                    f"读数 {r.value} 超出模数范围 [0, {self.modulus})"
                )
        if not (
            self.start.timestamp < self.missing_timestamp < self.end.timestamp
        ):
            raise ValueError(
                "漏采时刻必须严格介于起止时刻之间：首读数 "
                f"{self.start.timestamp.isoformat()}、漏采时刻 "
                f"{self.missing_timestamp.isoformat()}、末读数 "
                f"{self.end.timestamp.isoformat()}"
            )
        return self


# ---------------------------------------------------------------------------
# 复核接口响应模型
# ---------------------------------------------------------------------------

class SegmentOut(BaseModel):
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    wraps: int = Field(description="判定的整圈回绕次数")
    usage: int = Field(description="该区间用量（计数增量）")


class SegmentErrorOut(BaseModel):
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    reason: Literal["no_solution", "ambiguous"]
    detail: str
    min_wraps: int = Field(description="满足非负增量约束的最小回绕次数")
    max_wraps: int = Field(description="满足速率上限约束的最大回绕次数")


class VerifyResponse(BaseModel):
    status: Literal["ok", "error"]
    modulus: int
    segments: List[SegmentOut] = Field(description="回绕次数唯一的区间")
    errors: List[SegmentErrorOut] = Field(description="无解或多解的出错区间")
    total_usage: Optional[int] = Field(
        description="全部区间唯一时的总用量；存在出错区间时为 null"
    )


# ---------------------------------------------------------------------------
# 账单对账接口响应模型
# ---------------------------------------------------------------------------

class SegmentRangeOut(BaseModel):
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    min_wraps: int = Field(description="账单总量约束后仍可行的最小回绕次数")
    max_wraps: int = Field(description="账单总量约束后仍可行的最大回绕次数")
    wraps: Optional[int] = Field(
        default=None, description="圈数唯一时的回绕次数；仍有多解时为 null"
    )
    usage: Optional[int] = Field(
        default=None, description="圈数唯一时该段的确定用量；仍有多解时为 null"
    )
    min_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的下界；圈数唯一时为 null"
    )
    max_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的上界；圈数唯一时为 null"
    )


class ReconcileSegmentErrorOut(BaseModel):
    kind: Literal["segment"] = Field(description="冲突层级：分段")
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    reason: Literal["no_solution"]
    detail: str
    min_wraps: int
    max_wraps: int


class ReconcileTotalErrorOut(BaseModel):
    kind: Literal["total"] = Field(description="冲突层级：账单总量")
    reason: Literal[
        "total_incompatible_modulus", "total_out_of_range"
    ]
    detail: str
    feasible_min_total: int = Field(description="可行账单总量的下界（含）")
    feasible_max_total: int = Field(description="可行账单总量的上界（含）")
    expected_remainder: Optional[int] = Field(
        default=None, description="与模数相容时账单总量应有的模 M 余数"
    )
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；与模数不相容时为 null",
    )


class ReconcileResponse(BaseModel):
    status: Literal["ok", "conflict"]
    modulus: int
    total_usage: int = Field(description="请求给出的账单总用量")
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；冲突时可能为 null",
    )
    segments: List[SegmentRangeOut] = Field(
        description="账单总量约束后的各段结果：圈数唯一者给出确定用量，其余给出范围"
    )
    conflicts: List[
        Union[ReconcileSegmentErrorOut, ReconcileTotalErrorOut]
    ] = Field(description="冲突明细：某段原本无解，或账单总量与模数不相容/超界")


# ---------------------------------------------------------------------------
# 停水时段账单复核接口响应模型
# ---------------------------------------------------------------------------


class WaterFragmentOut(BaseModel):
    """一个相邻读数区间内、扣除停水后剩余的连续供水片段 [start, end)。"""

    start_timestamp: datetime
    end_timestamp: datetime
    seconds: int = Field(description="该供水片段秒数")
    capacity: int = Field(description="该片段整数容量 floor(速率×秒数)")


class OutageWindowOut(BaseModel):
    """裁剪到读数账期并合并后的一个停水时段 [start, end)。"""

    start_timestamp: datetime
    end_timestamp: datetime
    seconds: int = Field(description="该停水时段秒数（裁剪到账期后）")


class OutageSegmentRangeOut(BaseModel):
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int = Field(description="区间总时长（含停水），秒")
    outage_seconds: int = Field(description="落在该区间内的停水秒数")
    supply_seconds: int = Field(
        description="扣除停水后的供水秒数（各供水片段秒数之和）"
    )
    total_capacity: int = Field(
        description="各供水片段分别 floor(速率×秒数) 后的整数容量之和"
    )
    water_fragments: List[WaterFragmentOut] = Field(
        description="扣除停水后剩余的连续供水片段（左闭右开），按时间升序；"
        "各片段分别取整，不跨停水时段合并"
    )
    min_wraps: int = Field(description="账单总量约束后仍可行的最小回绕次数")
    max_wraps: int = Field(description="账单总量约束后仍可行的最大回绕次数")
    wraps: Optional[int] = Field(
        default=None, description="圈数唯一时的回绕次数；仍有多解时为 null"
    )
    usage: Optional[int] = Field(
        default=None, description="圈数唯一时该段的确定用量；仍有多解时为 null"
    )
    min_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的下界；圈数唯一时为 null"
    )
    max_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的上界；圈数唯一时为 null"
    )
    unique: bool = Field(description="该段回绕圈数是否已被账单总量唯一确定")


class OutageSegmentErrorOut(BaseModel):
    kind: Literal["segment"] = Field(description="冲突层级：分段")
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int = Field(description="区间总时长（含停水），秒")
    outage_seconds: int = Field(description="落在该区间内的停水秒数")
    supply_seconds: int = Field(description="扣除停水后的供水秒数")
    reason: Literal["no_solution"]
    detail: str
    min_wraps: int = Field(description="满足非负增量约束的最小回绕次数")
    max_wraps: int = Field(
        description="满足供水片段总容量约束的最大回绕次数；小于 min_wraps 表示无解"
    )
    total_capacity: int = Field(
        description="各供水片段 floor(速率×秒数) 容量之和；停水导致其可能小于"
        "floor(速率×供水总秒数)"
    )
    water_fragments: List[WaterFragmentOut] = Field(
        description="扣除停水后剩余的连续供水片段（左闭右开）"
    )


class OutageTotalErrorOut(BaseModel):
    kind: Literal["total"] = Field(description="冲突层级：账单总量")
    reason: Literal[
        "total_incompatible_modulus", "total_out_of_range"
    ]
    detail: str
    feasible_min_total: int = Field(
        description="扣除停水后各段最小可行用量之和，即可行账单总量下界（含）"
    )
    feasible_max_total: int = Field(
        description="扣除停水后各段最大可行用量之和，即可行账单总量上界（含）"
    )
    expected_remainder: Optional[int] = Field(
        default=None, description="与模数相容时账单总量应有的模 M 余数"
    )
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；与模数不相容时为 null",
    )


class OutageReconcileResponse(BaseModel):
    status: Literal["ok", "conflict"]
    modulus: int
    total_usage: int = Field(description="请求给出的账单总用量")
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；单段无解或与模数不相容时为 null",
    )
    outages: List[OutageWindowOut] = Field(
        description="裁剪到读数账期并合并重叠/相接时段后的停水时段，按开始时刻升序"
    )
    segments: List[OutageSegmentRangeOut] = Field(
        description="账单总量约束后的各段结果：供水秒数、容量、圈数/用量上下界与唯一性；冲突时为空"
    )
    conflicts: List[
        Union[OutageSegmentErrorOut, OutageTotalErrorOut]
    ] = Field(
        description="冲突明细：某段扣除停水后无解（no_solution），或账单总量"
        "与模数不相容（total_incompatible_modulus）/ 超出停水后可行总量"
        "（total_out_of_range）"
    )


# ---------------------------------------------------------------------------
# 每日用量上下界接口响应模型
# ---------------------------------------------------------------------------

# 每日边界的独立性说明：逐日边际极值，不保证跨日同时达到
DAILY_BOUNDARY_NOTE = (
    "每个 UTC 自然日的 min_usage/max_usage 均可由某个可行的回绕次数及"
    "子段整数分配分别达到；不同日期（以及同一读数区间内各子段）的上界"
    "不保证存在同一个分配同时成立，请勿将各日上界相加后作为整体上界使用"
)


class DailyUsageOut(BaseModel):
    date: date_cls = Field(description="UTC 自然日")
    min_usage: int = Field(
        description="当日所有落在该日的子段用量下界之和（非负整数，可达到）"
    )
    max_usage: int = Field(
        description="当日所有落在该日的子段用量上界之和（非负整数，可达到）"
    )
    segment_count: int = Field(
        description="当日覆盖到的读数区间子段数（同一区间至多贡献一个子段）"
    )


class DailyRangeConflictOut(BaseModel):
    index: int
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    reason: Literal["no_feasible_allocation"]
    detail: str
    min_wraps: int = Field(description="满足非负增量约束的最小回绕次数")
    max_wraps: int = Field(
        description="满足跨日子段总容量约束的最大回绕次数；小于 min_wraps 表示无解"
    )
    total_capacity: int = Field(
        description="区间内各 UTC 日子段容量 floor(速率×秒数) 之和"
    )


class DailyRangeResponse(BaseModel):
    status: Literal["ok", "conflict"]
    modulus: int
    days: List[DailyUsageOut] = Field(
        description="覆盖时段内各 UTC 自然日的最小/最大可行整数用量；冲突时为空"
    )
    boundary_note: str = Field(
        description="每日边界可分别达到、不保证各日上界同时成立的说明"
    )
    conflicts: List[DailyRangeConflictOut] = Field(
        description="不存在可行子段分配的读数区间及冲突原因；非空时 days 为空"
    )


# ---------------------------------------------------------------------------
# 换表账单复核接口响应模型
# ---------------------------------------------------------------------------


class MeterSegmentRangeOut(BaseModel):
    meter: Literal["old", "new"] = Field(description="所属表：旧表 / 新表")
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    modulus: int = Field(description="该表模数")
    min_wraps: int = Field(description="账单总量联立后仍可行的最小回绕次数")
    max_wraps: int = Field(description="账单总量联立后仍可行的最大回绕次数")
    wraps_step: int = Field(
        description="可行回绕次数序列的公差（模 gcd 归约后的对侧模数）"
    )
    wraps: Optional[int] = Field(
        default=None, description="圈数唯一时的回绕次数；仍有多解时为 null"
    )
    usage: Optional[int] = Field(
        default=None, description="圈数唯一时该段的确定用量；仍有多解时为 null"
    )
    min_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的下界；圈数唯一时为 null"
    )
    max_usage: Optional[int] = Field(
        default=None, description="仍有多解时该段用量的上界；圈数唯一时为 null"
    )


class MeterSegmentErrorOut(BaseModel):
    kind: Literal["segment"] = Field(description="冲突层级：分段")
    meter: Literal["old", "new"]
    start_timestamp: datetime
    end_timestamp: datetime
    start_reading: int
    end_reading: int
    elapsed_seconds: int
    modulus: int
    reason: Literal["no_solution"]
    detail: str
    min_wraps: int
    max_wraps: int


class MeterTotalErrorOut(BaseModel):
    kind: Literal["total"] = Field(description="冲突层级：账单总量")
    reason: Literal[
        "total_incompatible_modulus", "total_out_of_range"
    ]
    detail: str
    old_modulus: int
    new_modulus: int
    modulus_gcd: int = Field(description="两表模数的最大公约数 gcd(M_old, M_new)")
    feasible_min_total: int = Field(
        description="不联立账单时两段可行用量之和的下界（含）"
    )
    feasible_max_total: int = Field(
        description="不联立账单时两段可行用量之和的上界（含）"
    )
    expected_remainder: Optional[int] = Field(
        default=None,
        description="与两表模数相容时账单总量应有的模 gcd 余数",
    )


class MeterChangeResponse(BaseModel):
    status: Literal["ok", "conflict"]
    total_usage: int = Field(description="请求给出的跨两表账单总用量")
    unique: bool = Field(
        description="两段用量是否都被账单总量唯一确定（同时唯一）"
    )
    segments: List[MeterSegmentRangeOut] = Field(
        description="联立后的两段结果，按旧表（meter=old）、新表（meter=new）顺序；冲突时为空"
    )
    conflicts: List[
        Union[MeterSegmentErrorOut, MeterTotalErrorOut]
    ] = Field(description="冲突明细：某段原本无解，或账单总量不相容/超界")


# ---------------------------------------------------------------------------
# 漏采读数推断接口响应模型
# ---------------------------------------------------------------------------


class CandidateIntervalOut(BaseModel):
    """候选漏采读数的一个闭区间（端点含在内，均在 [0, modulus-1] 内）。"""

    min_reading: int = Field(description="候选读数闭区间下界（含）")
    max_reading: int = Field(description="候选读数闭区间上界（含）")


class MissingReadingConflictOut(BaseModel):
    reason: Literal["total_incompatible_modulus", "insufficient_capacity"]
    detail: str
    expected_remainder: Optional[int] = Field(
        default=None,
        description="账单与模数不相容时，账单应有的模 M 余数；容量不足时为 null",
    )
    left_capacity: int = Field(description="漏采时刻左侧容量 floor(速率×秒数)")
    right_capacity: int = Field(description="漏采时刻右侧容量 floor(速率×秒数)")
    feasible_max_total: int = Field(
        description="两侧容量之和，即该账期可行账单总量的上界（含）"
    )


class MissingReadingResponse(BaseModel):
    status: Literal["ok", "conflict"]
    modulus: int
    total_usage: int = Field(description="请求给出的账单总用量")
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；账单与模数不相容时为 null",
    )
    start_timestamp: datetime
    end_timestamp: datetime
    missing_timestamp: datetime
    start_reading: int
    end_reading: int
    left_seconds: int = Field(description="首读数到漏采时刻的秒数")
    right_seconds: int = Field(description="漏采时刻到末读数的秒数")
    left_capacity: int = Field(description="左侧容量 floor(速率×左秒数)")
    right_capacity: int = Field(description="右侧容量 floor(速率×右秒数)")
    left_usage_min: Optional[int] = Field(
        default=None, description="左侧可行用量闭区间下界（含）；冲突时为 null"
    )
    left_usage_max: Optional[int] = Field(
        default=None, description="左侧可行用量闭区间上界（含）；冲突时为 null"
    )
    candidate_intervals: List[CandidateIntervalOut] = Field(
        description="候选漏采读数的精确、不重叠闭区间（按左侧用量递增的环序；"
        "跨零点拆为两段，覆盖全模数时为全域 [0, modulus-1]）；冲突时为空"
    )
    candidate_count: int = Field(description="候选读数个数；冲突时为 0")
    unique: bool = Field(
        description="候选读数是否唯一（candidate_count == 1）"
    )
    conflicts: List[MissingReadingConflictOut] = Field(
        description="冲突明细：账单总量与模数不相容，或两侧容量不足"
    )


# ---------------------------------------------------------------------------
# 最小统一速率上限接口响应模型
# ---------------------------------------------------------------------------


class RateFractionOut(BaseModel):
    """约分后的精确速率（分子/分母，无浮点近似）。"""

    numerator: int = Field(description="约分后的分子")
    denominator: int = Field(description="约分后的分母（正整数）")


class MinimumRateTotalErrorOut(BaseModel):
    kind: Literal["total"] = Field(description="冲突层级：账单总量")
    reason: Literal[
        "total_incompatible_modulus", "total_out_of_range"
    ]
    detail: str
    feasible_min_total: int = Field(
        description="可行账单总量的下界（含）：各段取满足非负增量的最小圈数时的总量"
    )
    feasible_max_total: Optional[int] = Field(
        default=None,
        description="可行账单总量上界；速率不设上限，故恒为 null（无上限）",
    )
    expected_remainder: Optional[int] = Field(
        default=None, description="与模数相容时账单总量应有的模 M 余数"
    )
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；与模数不相容时为 null",
    )


class MinimumRateResponse(BaseModel):
    status: Literal["ok", "conflict"]
    modulus: int
    total_usage: int = Field(description="请求给出的账单总用量")
    total_wraps: Optional[int] = Field(
        default=None,
        description="账单总量隐含的总回绕次数 K；与模数不相容时为 null",
    )
    min_rate_per_second: Optional[RateFractionOut] = Field(
        default=None,
        description="最小统一速率上限（约分后的分子/分母）；冲突时为 null",
    )
    unique: bool = Field(
        description="该门槛下各段圈数是否全部被唯一确定"
    )
    segments: List[SegmentRangeOut] = Field(
        description="最小速率门槛与账单总量约束后的各段结果："
        "圈数唯一者给出确定用量，其余给出范围；冲突时为空"
    )
    conflicts: List[MinimumRateTotalErrorOut] = Field(
        description="冲突明细：账单总量与模数不相容，或低于可行最小总量"
    )


# ---------------------------------------------------------------------------
# 公共逻辑
# ---------------------------------------------------------------------------

def _no_solution_detail(diff: int, modulus: int, sol, elapsed: int) -> str:
    min_delta = diff + sol.min_wraps * modulus
    return (
        f"无解：最可能增量 {min_delta} 已超出该区间 "
        f"{elapsed}s 内允许的最大增量 {sol.max_allowed}"
        f"（k 需同时满足 k>={sol.min_wraps} 与 k<={sol.max_wraps}，区间为空）"
    )


def _solve_all(req: _BaseUsageRequest):
    """对每段执行现有读数/模数/速率校验，返回 (段上下文, 判定结果) 列表。"""
    max_rate = Fraction(req.max_rate_per_second)
    results = []
    for i, (a, b) in enumerate(zip(req.readings, req.readings[1:])):
        elapsed = int((b.timestamp - a.timestamp).total_seconds())
        sol = solve_segment(i, a.value, b.value, elapsed, req.modulus, max_rate)
        results.append((a, b, elapsed, sol))
    return results


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/usage/verify", response_model=VerifyResponse)
def verify_usage(req: VerifyRequest) -> VerifyResponse:
    segments: List[SegmentOut] = []
    errors: List[SegmentErrorOut] = []

    for a, b, elapsed, sol in _solve_all(req):
        if sol.status == STATUS_OK:
            segments.append(
                SegmentOut(
                    index=sol.index,
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    start_reading=a.value,
                    end_reading=b.value,
                    elapsed_seconds=elapsed,
                    wraps=sol.wraps,
                    usage=sol.usage,
                )
            )
            continue

        diff = b.value - a.value
        if sol.status == STATUS_NO_SOLUTION:
            detail = _no_solution_detail(diff, req.modulus, sol, elapsed)
        else:  # STATUS_AMBIGUOUS
            detail = (
                f"多解：回绕次数 k 在 [{sol.min_wraps}, {sol.max_wraps}] 内均满足"
                f"非负增量与速率上限约束，无法唯一确定用量，不做猜测"
            )
        errors.append(
            SegmentErrorOut(
                index=sol.index,
                start_timestamp=a.timestamp,
                end_timestamp=b.timestamp,
                start_reading=a.value,
                end_reading=b.value,
                elapsed_seconds=elapsed,
                reason=sol.status,  # type: ignore[arg-type]
                detail=detail,
                min_wraps=sol.min_wraps,
                max_wraps=sol.max_wraps,
            )
        )

    has_error = bool(errors)
    return VerifyResponse(
        status="error" if has_error else "ok",
        modulus=req.modulus,
        segments=segments,
        errors=errors,
        total_usage=None if has_error else sum(s.usage for s in segments),
    )


@app.post("/v1/usage/reconcile", response_model=ReconcileResponse)
def reconcile_usage(req: ReconcileRequest) -> ReconcileResponse:
    """账单总量对账：复用逐段读数/模数/速率校验，再以账单总量收紧圈数范围。"""
    solved = _solve_all(req)

    # 1) 某段原本无解：直接构成冲突，不进行总量推断
    segment_conflicts: List[ReconcileSegmentErrorOut] = []
    feasible = []
    for a, b, elapsed, sol in solved:
        if sol.status == STATUS_NO_SOLUTION:
            segment_conflicts.append(
                ReconcileSegmentErrorOut(
                    kind="segment",
                    index=sol.index,
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    start_reading=a.value,
                    end_reading=b.value,
                    elapsed_seconds=elapsed,
                    reason=STATUS_NO_SOLUTION,
                    detail=_no_solution_detail(b.value - a.value, req.modulus, sol, elapsed),
                    min_wraps=sol.min_wraps,
                    max_wraps=sol.max_wraps,
                )
            )
        else:
            feasible.append((a, b, elapsed, sol))

    if segment_conflicts:
        return ReconcileResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            total_usage=req.total_usage,
            total_wraps=None,
            segments=[],
            conflicts=segment_conflicts,
        )

    # 2) 全部段本身可行（唯一或多解），以整数区间运算做总量对账
    diffs = [b.value - a.value for a, b, _, _ in feasible]
    bounds = [(sol.min_wraps, sol.max_wraps) for *_, sol in feasible]
    result = reconcile_total(diffs, bounds, req.total_usage, req.modulus)

    if result.status == STATUS_CONFLICT:
        end_minus_start = sum(diffs)
        expected_remainder = end_minus_start % req.modulus
        total_conflicts = []
        for reason in result.conflicts:
            if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS:
                detail = (
                    f"账单总量与模数不相容：总量 B 必须满足 "
                    f"B ≡ {expected_remainder} (mod {req.modulus})"
                    f"（首尾读数差为 {end_minus_start}），收到 "
                    f"{req.total_usage}（余数 {req.total_usage % req.modulus}），"
                    f"总回绕次数不是整数"
                )
            else:  # REASON_TOTAL_OUT_OF_RANGE
                detail = (
                    f"账单总量超出可行范围：各段回绕次数均为非负整数且满足"
                    f"速率上限时，总用量只能在 [{result.feasible_min_total}, "
                    f"{result.feasible_max_total}]（模数 {req.modulus}）之内，"
                    f"收到 {req.total_usage}"
                )
            total_conflicts.append(
                ReconcileTotalErrorOut(
                    kind="total",
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    feasible_min_total=result.feasible_min_total,
                    feasible_max_total=result.feasible_max_total,
                    expected_remainder=(
                        expected_remainder
                        if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS
                        else None
                    ),
                    total_wraps=result.total_wraps,
                )
            )
        return ReconcileResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            total_usage=req.total_usage,
            total_wraps=result.total_wraps,
            segments=[],
            conflicts=total_conflicts,
        )

    # 3) 可行：圈数已唯一的段确定用量，其余仅给范围，不猜测
    segments: List[SegmentRangeOut] = []
    for (a, b, elapsed, _), ns in zip(feasible, result.narrowed):
        min_usage, max_usage = ns.usage_bounds(req.modulus)
        segments.append(
            SegmentRangeOut(
                index=ns.index,
                start_timestamp=a.timestamp,
                end_timestamp=b.timestamp,
                start_reading=a.value,
                end_reading=b.value,
                elapsed_seconds=elapsed,
                min_wraps=ns.min_wraps,
                max_wraps=ns.max_wraps,
                wraps=ns.wraps,
                usage=ns.usage(req.modulus),
                min_usage=None if ns.unique else min_usage,
                max_usage=None if ns.unique else max_usage,
            )
        )

    return ReconcileResponse(
        status=STATUS_OK,
        modulus=req.modulus,
        total_usage=req.total_usage,
        total_wraps=result.total_wraps,
        segments=segments,
        conflicts=[],
    )


@app.post("/v1/usage/outage-reconcile", response_model=OutageReconcileResponse)
def reconcile_usage_with_outages(req: OutageReconcileRequest) -> OutageReconcileResponse:
    """停水时段账单复核：扣除停水后按供水片段分别取整，再以账单总量联立。

    复用现有秒级时间戳、读数范围、模数与速率上限校验。停水时段按左闭
    右开处理：先统一裁剪到读数账期 ``[首读数时刻, 末读数时刻)``，再合并
    重叠或相接（端点相同）的时段。对每对相邻读数，从其区间内扣除停水
    时段，把剩余的连续供水片段分别按 ``floor(速率 × 秒数)`` 取整后求和，
    作为该段整数用量容量——**不跨停水时段合并取整**。该容量与非负增量
    一起夹出可行回绕圈数区间；任一区间无解即以分段冲突返回，不做总量
    联立。各段均可行后，账单总用量经与
    ``/v1/usage/reconcile`` 相同的整数区间运算收紧各段圈数（账单同余
    冲突、总量超界原因可区分）。
    """
    max_rate = Fraction(req.max_rate_per_second)
    period_start = req.readings[0].timestamp
    period_end = req.readings[-1].timestamp

    # 1) 停水时段裁剪到账期并合并重叠/相接（左闭右开）
    raw_outages = [(w.start, w.end) for w in req.outages]
    merged = clip_merge_intervals(raw_outages, period_start, period_end)
    outage_out = [
        OutageWindowOut(
            start_timestamp=s,
            end_timestamp=e,
            seconds=int((e - s).total_seconds()),
        )
        for s, e in merged
    ]

    def _fragments_out(fragments) -> List[WaterFragmentOut]:
        return [
            WaterFragmentOut(
                start_timestamp=f.start,
                end_timestamp=f.end,
                seconds=f.seconds,
                capacity=f.capacity,
            )
            for f in fragments
        ]

    # 2) 逐段扣除停水、按供水片段分别取整求容量，夹出可行圈数区间
    solved = []
    segment_conflicts: List[OutageSegmentErrorOut] = []
    for i, (a, b) in enumerate(zip(req.readings, req.readings[1:])):
        seg_result = solve_outage_segment(
            i,
            a.value,
            b.value,
            a.timestamp,
            b.timestamp,
            req.modulus,
            max_rate,
            merged,
        )
        diff = b.value - a.value
        if not seg_result.feasible:
            min_delta = diff + seg_result.min_wraps * req.modulus
            detail = (
                f"无解：扣除停水后该区间供水 {seg_result.supply_seconds}s，"
                f"{len(seg_result.fragments)} 个连续供水片段分别 "
                f"floor(速率×秒数) 的容量之和仅为 {seg_result.total_capacity}，"
                f"而满足非负增量的最小总增量已为 {min_delta}"
                f"（读数差 {diff} + {seg_result.min_wraps}×{req.modulus}）；"
                f"回绕次数需同时满足 k>={seg_result.min_wraps} 与 "
                f"k<={seg_result.max_wraps}，区间为空（各片段分别取整，"
                f"不能跨停水时段合并取整）"
            )
            segment_conflicts.append(
                OutageSegmentErrorOut(
                    kind="segment",
                    index=i,
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    start_reading=a.value,
                    end_reading=b.value,
                    elapsed_seconds=seg_result.elapsed_seconds,
                    outage_seconds=seg_result.outage_seconds,
                    supply_seconds=seg_result.supply_seconds,
                    reason=STATUS_NO_SOLUTION,
                    detail=detail,
                    min_wraps=seg_result.min_wraps,
                    max_wraps=seg_result.max_wraps,
                    total_capacity=seg_result.total_capacity,
                    water_fragments=_fragments_out(seg_result.fragments),
                )
            )
        else:
            solved.append((a, b, seg_result))

    if segment_conflicts:
        return OutageReconcileResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            total_usage=req.total_usage,
            total_wraps=None,
            outages=outage_out,
            segments=[],
            conflicts=segment_conflicts,
        )

    # 3) 各段扣除停水后均可行：以账单总用量整数区间联立收紧（复用对账逻辑）
    diffs = [b.value - a.value for a, b, _ in solved]
    bounds = [(r.min_wraps, r.max_wraps) for *_, r in solved]
    result = reconcile_total(diffs, bounds, req.total_usage, req.modulus)

    if result.status == STATUS_CONFLICT:
        end_minus_start = sum(diffs)
        expected_remainder = end_minus_start % req.modulus
        total_conflicts: List[OutageTotalErrorOut] = []
        for reason in result.conflicts:
            if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS:
                detail = (
                    f"账单总量与模数不相容：总量 B 必须满足 "
                    f"B ≡ {expected_remainder} (mod {req.modulus})"
                    f"（首尾读数差为 {end_minus_start}），收到 "
                    f"{req.total_usage}（余数 {req.total_usage % req.modulus}），"
                    f"总回绕次数不是整数（停水只影响各段容量，不改变模 M 同余）"
                )
            else:  # REASON_TOTAL_OUT_OF_RANGE
                detail = (
                    f"账单总量超出停水后的可行范围：扣除停水后各段回绕次数均为"
                    f"非负整数且不超过各自供水片段容量之和时，总用量只能在 "
                    f"[{result.feasible_min_total}, {result.feasible_max_total}]"
                    f"（模数 {req.modulus}）之内，收到 {req.total_usage}"
                )
            total_conflicts.append(
                OutageTotalErrorOut(
                    kind="total",
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    feasible_min_total=result.feasible_min_total,
                    feasible_max_total=result.feasible_max_total,
                    expected_remainder=(
                        expected_remainder
                        if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS
                        else None
                    ),
                    total_wraps=result.total_wraps,
                )
            )
        return OutageReconcileResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            total_usage=req.total_usage,
            total_wraps=result.total_wraps,
            outages=outage_out,
            segments=[],
            conflicts=total_conflicts,
        )

    # 4) 可行：圈数唯一者给出确定用量，其余仅给范围，不猜测
    segments: List[OutageSegmentRangeOut] = []
    for (a, b, seg), ns in zip(solved, result.narrowed):
        min_usage, max_usage = ns.usage_bounds(req.modulus)
        segments.append(
            OutageSegmentRangeOut(
                index=ns.index,
                start_timestamp=a.timestamp,
                end_timestamp=b.timestamp,
                start_reading=a.value,
                end_reading=b.value,
                elapsed_seconds=seg.elapsed_seconds,
                outage_seconds=seg.outage_seconds,
                supply_seconds=seg.supply_seconds,
                total_capacity=seg.total_capacity,
                water_fragments=_fragments_out(seg.fragments),
                min_wraps=ns.min_wraps,
                max_wraps=ns.max_wraps,
                wraps=ns.wraps,
                usage=ns.usage(req.modulus),
                min_usage=None if ns.unique else min_usage,
                max_usage=None if ns.unique else max_usage,
                unique=ns.unique,
            )
        )

    return OutageReconcileResponse(
        status=STATUS_OK,
        modulus=req.modulus,
        total_usage=req.total_usage,
        total_wraps=result.total_wraps,
        outages=outage_out,
        segments=segments,
        conflicts=[],
    )


@app.post("/v1/usage/daily-range", response_model=DailyRangeResponse)
def daily_usage_range(req: DailyRangeRequest) -> DailyRangeResponse:
    """按 UTC 自然日给出覆盖时段内每日最小/最大可行整数用量。

    复用读数、模数、最大每秒速率的请求校验；相邻读数跨 UTC 午夜时按
    午夜切成子段，子段容量为 ``floor(max_rate_per_second × 秒数)``，
    区间总增量 ``读数差 + k×模数`` 须能以非负整数分配到各子段。
    """
    max_rate = Fraction(req.max_rate_per_second)
    conflicts: List[DailyRangeConflictOut] = []
    # day -> [下界之和, 上界之和, 子段数]
    day_acc: "dict[date_cls, list[int]]" = {}

    for i, (a, b) in enumerate(zip(req.readings, req.readings[1:])):
        result = solve_daily_segment(
            i, a.value, b.value, a.timestamp, b.timestamp, req.modulus, max_rate
        )
        elapsed = int((b.timestamp - a.timestamp).total_seconds())

        if not result.feasible:
            diff = b.value - a.value
            min_delta = diff + result.min_wraps * req.modulus
            detail = (
                f"无可行分配：该区间按 UTC 午夜切成 {len(result.subsegments)} 段，"
                f"各子段容量 floor(速率×秒数) 之和为 {result.total_capacity}，"
                f"而满足非负增量的最小总增量已为 {min_delta}"
                f"（读数差 {diff} + {result.min_wraps}×{req.modulus}）；"
                f"回绕次数需同时满足 k>={result.min_wraps} 与 k<={result.max_wraps}，"
                f"区间为空，无法把整数增量分配到各日子段"
            )
            conflicts.append(
                DailyRangeConflictOut(
                    index=i,
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    start_reading=a.value,
                    end_reading=b.value,
                    elapsed_seconds=elapsed,
                    reason=REASON_NO_FEASIBLE_ALLOCATION,  # type: ignore[arg-type]
                    detail=detail,
                    min_wraps=result.min_wraps,
                    max_wraps=result.max_wraps,
                    total_capacity=result.total_capacity,
                )
            )
            continue

        for seg, low, high in zip(
            result.subsegments, result.min_usages, result.max_usages
        ):
            acc = day_acc.setdefault(seg.day, [0, 0, 0])
            acc[0] += low
            acc[1] += high
            acc[2] += 1

    if conflicts:
        return DailyRangeResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            days=[],
            boundary_note=DAILY_BOUNDARY_NOTE,
            conflicts=conflicts,
        )

    days = [
        DailyUsageOut(
            date=day,
            min_usage=acc[0],
            max_usage=acc[1],
            segment_count=acc[2],
        )
        for day, acc in sorted(day_acc.items())
    ]
    return DailyRangeResponse(
        status=STATUS_OK,
        modulus=req.modulus,
        days=days,
        boundary_note=DAILY_BOUNDARY_NOTE,
        conflicts=[],
    )


# ---------------------------------------------------------------------------
# 换表账单复核
# ---------------------------------------------------------------------------


def _solve_meter_block(meter: str, block: MeterBlock):
    """对一块表的段执行现有读数/模数/速率判定，返回 (上下文, 判定结果)。"""
    a, b = block.start, block.end
    elapsed = int((b.timestamp - a.timestamp).total_seconds())
    max_rate = Fraction(block.max_rate_per_second)
    sol = solve_segment(
        0 if meter == "old" else 1,
        a.value,
        b.value,
        elapsed,
        block.modulus,
        max_rate,
    )
    return a, b, elapsed, sol


@app.post("/v1/usage/meter-change/reconcile", response_model=MeterChangeResponse)
def reconcile_meter_change_usage(req: MeterChangeRequest) -> MeterChangeResponse:
    """换表账单复核：旧表/新表两段（模数可不同）以账单总量精确联立。

    复用现有秒级时间戳、读数范围与速率上限校验；另要求旧表终读与新表
    初读同刻。两段先各自求满足非负增量与速率上限的回绕整数区间，再对
    线性丢番图方程 ``M_old*k1 + M_new*k2 = B - d1 - d2`` 求单参数整数
    通解并由两段区间夹出参数范围，不枚举圈数组合。
    """
    solved = {
        "old": _solve_meter_block("old", req.old_meter),
        "new": _solve_meter_block("new", req.new_meter),
    }

    # 1) 某段原本无解：以分段冲突返回，不进行总量联立
    blocks = {"old": req.old_meter, "new": req.new_meter}
    segment_conflicts: List[MeterSegmentErrorOut] = []
    feasible: dict = {}
    for meter in ("old", "new"):
        block = blocks[meter]
        a, b, elapsed, sol = solved[meter]
        if sol.status == STATUS_NO_SOLUTION:
            segment_conflicts.append(
                MeterSegmentErrorOut(
                    kind="segment",
                    meter=meter,  # type: ignore[arg-type]
                    start_timestamp=a.timestamp,
                    end_timestamp=b.timestamp,
                    start_reading=a.value,
                    end_reading=b.value,
                    elapsed_seconds=elapsed,
                    modulus=block.modulus,
                    reason=STATUS_NO_SOLUTION,
                    detail=_no_solution_detail(
                        b.value - a.value, block.modulus, sol, elapsed
                    ),
                    min_wraps=sol.min_wraps,
                    max_wraps=sol.max_wraps,
                )
            )
        else:
            feasible[meter] = (block, a, b, elapsed, sol)

    if segment_conflicts:
        return MeterChangeResponse(
            status=STATUS_CONFLICT,
            total_usage=req.total_usage,
            unique=False,
            segments=[],
            conflicts=segment_conflicts,
        )

    # 2) 两段本身可行（唯一或多解），模数可不同，精确联立总量
    old_block, old_start, old_end, _, old_sol = feasible["old"]
    new_block, new_start, new_end, _, new_sol = feasible["new"]
    result = reconcile_meter_total(
        old_diff=old_end.value - old_start.value,
        old_bounds=(old_sol.min_wraps, old_sol.max_wraps),
        old_modulus=old_block.modulus,
        new_diff=new_end.value - new_start.value,
        new_bounds=(new_sol.min_wraps, new_sol.max_wraps),
        new_modulus=new_block.modulus,
        total_usage=req.total_usage,
    )

    if result.status == STATUS_CONFLICT:
        conflicts: List[MeterTotalErrorOut] = []
        for reason in result.conflicts:
            if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS:
                detail = (
                    f"账单总量与两表模数不相容：总量 B 必须满足 "
                    f"B ≡ d_old + d_new (mod gcd)，其中 gcd("
                    f"{old_block.modulus}, {new_block.modulus}) = {result.gcd}；"
                    f"收到 {req.total_usage}，除以 {result.gcd} 不整除，"
                    f"两段回绕次数无法同时取整数"
                )
                expected_remainder = (
                    old_end.value
                    - old_start.value
                    + new_end.value
                    - new_start.value
                ) % result.gcd
            else:  # REASON_TOTAL_OUT_OF_RANGE
                detail = (
                    f"账单总量联立后无可行整数解：两段回绕次数各自为非负整数"
                    f"且满足速率上限时，分段用量之和只能在 "
                    f"[{result.feasible_min_total}, {result.feasible_max_total}] "
                    f"内取值（旧表模数 {old_block.modulus}、新表模数 "
                    f"{new_block.modulus}，gcd {result.gcd}）；收到 "
                    f"{req.total_usage}。"
                )
                if (
                    result.feasible_min_total
                    <= req.total_usage
                    <= result.feasible_max_total
                ):
                    detail += (
                        "总量虽落在上述区间内，但不存在同时满足两表圈数区间的"
                        "整数组合（模 gcd 相容但联立通解落在可行范围之外）"
                    )
                expected_remainder = None
            conflicts.append(
                MeterTotalErrorOut(
                    kind="total",
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    old_modulus=old_block.modulus,
                    new_modulus=new_block.modulus,
                    modulus_gcd=result.gcd,
                    feasible_min_total=result.feasible_min_total,
                    feasible_max_total=result.feasible_max_total,
                    expected_remainder=expected_remainder,
                )
            )
        return MeterChangeResponse(
            status=STATUS_CONFLICT,
            total_usage=req.total_usage,
            unique=False,
            segments=[],
            conflicts=conflicts,
        )

    # 3) 可行：圈数唯一者给出确定用量，其余仅给范围，不猜测
    segments: List[MeterSegmentRangeOut] = []
    for meter, ns, ctx in (
        ("old", result.old_segment, feasible["old"]),
        ("new", result.new_segment, feasible["new"]),
    ):
        block, a, b, elapsed, _ = ctx
        min_usage, max_usage = ns.usage_bounds()
        segments.append(
            MeterSegmentRangeOut(
                meter=meter,  # type: ignore[arg-type]
                start_timestamp=a.timestamp,
                end_timestamp=b.timestamp,
                start_reading=a.value,
                end_reading=b.value,
                elapsed_seconds=elapsed,
                modulus=block.modulus,
                min_wraps=ns.min_wraps,
                max_wraps=ns.max_wraps,
                wraps_step=ns.wraps_step,
                wraps=ns.wraps,
                usage=min_usage if ns.unique else None,
                min_usage=None if ns.unique else min_usage,
                max_usage=None if ns.unique else max_usage,
            )
        )

    both_unique = result.old_segment.unique and result.new_segment.unique
    return MeterChangeResponse(
        status=STATUS_OK,
        total_usage=req.total_usage,
        unique=both_unique,
        segments=segments,
        conflicts=[],
    )


# ---------------------------------------------------------------------------
# 漏采读数推断
# ---------------------------------------------------------------------------


@app.post("/v1/usage/missing-reading", response_model=MissingReadingResponse)
def infer_missing_reading(req: MissingReadingRequest) -> MissingReadingResponse:
    """漏采读数推断：以账单总量与漏采时刻两侧容量约束候选读数，不猜测。

    复用现有秒级时间戳、读数范围、模数与速率校验。先校验账单总量与首尾
    读数差的模 M 同余（同余且账单非负时隐含总回绕次数自动非负），再按
    漏采时刻两侧各自的 ``floor(速率×秒数)`` 整数容量求左侧可行用量闭
    区间；容量不足即冲突。可行时将首读数加左侧用量取模，候选在模 M 环
    上为连续弧：跨零点拆段、覆盖全模数时返回全域，并给出候选个数与
    是否唯一。
    """
    left_seconds = int(
        (req.missing_timestamp - req.start.timestamp).total_seconds()
    )
    right_seconds = int(
        (req.end.timestamp - req.missing_timestamp).total_seconds()
    )
    result = solve_missing_reading(
        req.start.value,
        req.end.value,
        left_seconds,
        right_seconds,
        req.modulus,
        Fraction(req.max_rate_per_second),
        req.total_usage,
    )

    conflicts: List[MissingReadingConflictOut] = []
    for reason in result.conflicts:
        if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS:
            expected = result.diff % req.modulus
            detail = (
                f"账单总量与首尾读数不相容：总量 B 必须满足 "
                f"B ≡ {expected} (mod {req.modulus})"
                f"（末读数 − 首读数 = {result.diff}），收到 "
                f"{req.total_usage}（余数 {req.total_usage % req.modulus}），"
                f"总回绕次数不是整数"
            )
        else:  # REASON_INSUFFICIENT_CAPACITY
            expected = None
            detail = (
                f"两侧容量不足：账单总量 {req.total_usage} 超出漏采时刻两侧"
                f"容量之和 {result.left_capacity} + {result.right_capacity} = "
                f"{result.left_capacity + result.right_capacity}"
                f"（左侧 {left_seconds}s、右侧 {right_seconds}s 各自的 "
                f"floor(速率×秒数)），不存在可行的左右侧非负整数用量分配"
            )
        conflicts.append(
            MissingReadingConflictOut(
                reason=reason,  # type: ignore[arg-type]
                detail=detail,
                expected_remainder=expected,
                left_capacity=result.left_capacity,
                right_capacity=result.right_capacity,
                feasible_max_total=result.left_capacity + result.right_capacity,
            )
        )

    return MissingReadingResponse(
        status=STATUS_CONFLICT if conflicts else STATUS_OK,
        modulus=req.modulus,
        total_usage=req.total_usage,
        total_wraps=result.total_wraps,
        start_timestamp=req.start.timestamp,
        end_timestamp=req.end.timestamp,
        missing_timestamp=req.missing_timestamp,
        start_reading=req.start.value,
        end_reading=req.end.value,
        left_seconds=left_seconds,
        right_seconds=right_seconds,
        left_capacity=result.left_capacity,
        right_capacity=result.right_capacity,
        left_usage_min=result.left_usage_min,
        left_usage_max=result.left_usage_max,
        candidate_intervals=[
            CandidateIntervalOut(min_reading=lo, max_reading=hi)
            for lo, hi in result.intervals
        ],
        candidate_count=result.candidate_count,
        unique=result.unique,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# 最小统一速率上限
# ---------------------------------------------------------------------------


@app.post("/v1/usage/minimum-rate", response_model=MinimumRateResponse)
def minimum_uniform_rate(req: MinimumRateRequest) -> MinimumRateResponse:
    """最小统一速率上限：按账单总用量反求各段速率上限的最小统一门槛。

    复用现有秒级时间戳、读数范围与模数校验（无速率上限输入）。先校验账单
    总量与首尾读数差的模 M 同余（不相容即明确冲突）；相容时，在各段非负
    整数用量且总和等于账单的约束下，求使「段用量 ÷ 段历时」均不超过它的
    最小统一速率上限，以约分后的分子/分母精确返回（Fraction 精确二分 +
    整数区间运算，不用浮点近似，也不枚举全部圈数组合），并给出该门槛下
    每段可行圈数范围与是否唯一；范围不唯一时不虚构单段用量。
    """
    pairs = list(zip(req.readings, req.readings[1:]))
    diffs = [b.value - a.value for a, b in pairs]
    elapsed = [int((b.timestamp - a.timestamp).total_seconds()) for a, b in pairs]
    result = solve_minimum_rate(diffs, elapsed, req.total_usage, req.modulus)

    if result.status == STATUS_CONFLICT:
        end_minus_start = sum(diffs)
        conflicts: List[MinimumRateTotalErrorOut] = []
        for reason in result.conflicts:
            if reason == REASON_TOTAL_INCOMPATIBLE_MODULUS:
                expected: Optional[int] = end_minus_start % req.modulus
                detail = (
                    f"账单总量与模数不相容：总量 B 必须满足 "
                    f"B ≡ {expected} (mod {req.modulus})"
                    f"（首尾读数差为 {end_minus_start}），收到 "
                    f"{req.total_usage}（余数 {req.total_usage % req.modulus}），"
                    f"总回绕次数不是整数"
                )
            else:  # REASON_TOTAL_OUT_OF_RANGE
                expected = None
                detail = (
                    f"账单总量低于可行最小总量：各段取满足非负增量的最小圈数时，"
                    f"总用量已为 {result.feasible_min_total}（模数 {req.modulus}），"
                    f"收到 {req.total_usage}；速率上限不设约束时可行总量没有上界，"
                    f"但不能低于该最小值"
                )
            conflicts.append(
                MinimumRateTotalErrorOut(
                    kind="total",
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    feasible_min_total=result.feasible_min_total,
                    feasible_max_total=None,
                    expected_remainder=expected,
                    total_wraps=result.total_wraps,
                )
            )
        return MinimumRateResponse(
            status=STATUS_CONFLICT,
            modulus=req.modulus,
            total_usage=req.total_usage,
            total_wraps=result.total_wraps,
            min_rate_per_second=None,
            unique=False,
            segments=[],
            conflicts=conflicts,
        )

    # 可行：圈数已唯一的段确定用量，其余仅给范围，不猜测
    segments: List[SegmentRangeOut] = []
    for (a, b), sec, ns in zip(pairs, elapsed, result.segments):
        min_usage, max_usage = ns.usage_bounds(req.modulus)
        segments.append(
            SegmentRangeOut(
                index=ns.index,
                start_timestamp=a.timestamp,
                end_timestamp=b.timestamp,
                start_reading=a.value,
                end_reading=b.value,
                elapsed_seconds=sec,
                min_wraps=ns.min_wraps,
                max_wraps=ns.max_wraps,
                wraps=ns.wraps,
                usage=ns.usage(req.modulus),
                min_usage=None if ns.unique else min_usage,
                max_usage=None if ns.unique else max_usage,
            )
        )

    rate = result.min_rate
    assert rate is not None  # STATUS_OK 时必有最小速率
    return MinimumRateResponse(
        status=STATUS_OK,
        modulus=req.modulus,
        total_usage=req.total_usage,
        total_wraps=result.total_wraps,
        min_rate_per_second=RateFractionOut(
            numerator=rate.numerator, denominator=rate.denominator
        ),
        unique=all(ns.unique for ns in result.segments),
        segments=segments,
        conflicts=[],
    )


# ---------------------------------------------------------------------------
# 双表同段交叉核验
# ---------------------------------------------------------------------------


class DualMeterCompareRequest(BaseModel):
    """双表同段交叉核验请求：两块表在同一时段计量同一用量。"""

    meter_a: MeterBlock = Field(description="表 A 的模数、速率上限与起止读数")
    meter_b: MeterBlock = Field(description="表 B 的模数、速率上限与起止读数")

    @model_validator(mode="after")
    def _check_same_period(self) -> "DualMeterCompareRequest":
        # 共同计量：两表起止时刻必须分别相同（读数本身可不同）
        if self.meter_a.start.timestamp != self.meter_b.start.timestamp:
            raise ValueError(
                "两表首读时刻必须相同（共同计量起点）：表 A 首读 "
                f"{self.meter_a.start.timestamp.isoformat()} 与表 B 首读 "
                f"{self.meter_b.start.timestamp.isoformat()} 不一致"
            )
        if self.meter_a.end.timestamp != self.meter_b.end.timestamp:
            raise ValueError(
                "两表末读时刻必须相同（共同计量终点）：表 A 末读 "
                f"{self.meter_a.end.timestamp.isoformat()} 与表 B 末读 "
                f"{self.meter_b.end.timestamp.isoformat()} 不一致"
            )
        return self


class DualMeterSegmentOut(BaseModel):
    meter: Literal["a", "b"] = Field(description="表标识：表 A / 表 B")
    modulus: int = Field(description="该表模数")
    start_reading: int
    end_reading: int
    min_wraps: int = Field(description="该表满足非负增量与速率上限的最小回绕次数")
    max_wraps: int = Field(description="该表满足非负增量与速率上限的最大回绕次数")
    min_usage: int = Field(description="该表独立可行用量下界（含）")
    max_usage: int = Field(description="该表独立可行用量上界（含）")


class DualMeterSegmentErrorOut(BaseModel):
    kind: Literal["segment"] = Field(description="冲突层级：分段")
    meter: Literal["a", "b"]
    modulus: int
    start_reading: int
    end_reading: int
    reason: Literal["no_solution"]
    detail: str
    min_wraps: int = Field(description="满足非负增量约束的最小回绕次数")
    max_wraps: int = Field(
        description="满足速率上限约束的最大回绕次数；小于 min_wraps 表示无解"
    )


class DualMeterCommonErrorOut(BaseModel):
    kind: Literal["common"] = Field(description="冲突层级：共同用量")
    reason: Literal["incompatible_modulus", "range_disjoint"]
    detail: str
    modulus_gcd: int = Field(description="两表模数的最大公约数 gcd(M_a, M_b)")
    usage_step: int = Field(description="共同候选用量的模周期 lcm(M_a, M_b)")
    expected_remainder: Optional[int] = Field(
        default=None,
        description="同余相容时两表读数差应共有的模 gcd 余数（以表 A 为准）；"
        "范围无交集时为 null",
    )
    a_usage_min: int = Field(description="表 A 独立可行用量下界（含）")
    a_usage_max: int = Field(description="表 A 独立可行用量上界（含）")
    b_usage_min: int = Field(description="表 B 独立可行用量下界（含）")
    b_usage_max: int = Field(description="表 B 独立可行用量上界（含）")
    intersect_min: int = Field(description="两表可行用量范围的区间交下界")
    intersect_max: int = Field(
        description="两表可行用量范围的区间交上界；小于下界即交集为空"
    )


class DualMeterCompareResponse(BaseModel):
    status: Literal["ok", "conflict"]
    start_timestamp: datetime = Field(description="共同计量起点（两表首读时刻相同）")
    end_timestamp: datetime = Field(description="共同计量终点（两表末读时刻相同）")
    elapsed_seconds: int = Field(description="共同计量时段秒数")
    modulus_gcd: int = Field(description="两表模数的最大公约数 gcd(M_a, M_b)")
    usage_step: int = Field(description="相邻共同候选用量之差 lcm(M_a, M_b)")
    min_usage: Optional[int] = Field(
        default=None, description="共同用量最小值；冲突时为 null"
    )
    max_usage: Optional[int] = Field(
        default=None, description="共同用量最大值；冲突时为 null"
    )
    candidate_count: int = Field(
        description="共同用量候选数（等差数列项数）；冲突时为 0"
    )
    unique: bool = Field(description="共同用量是否唯一（candidate_count == 1）")
    meters: List[DualMeterSegmentOut] = Field(
        description="两表各自的可行圈数与用量范围，按表 A、表 B 顺序；冲突时为空"
    )
    conflicts: List[Union[DualMeterSegmentErrorOut, DualMeterCommonErrorOut]] = Field(
        description="冲突明细：某表段原本无解，或同余不相容 / 范围无交集"
    )


@app.post("/v1/usage/dual-meter/compare", response_model=DualMeterCompareResponse)
def compare_dual_meter_usage(req: DualMeterCompareRequest) -> DualMeterCompareResponse:
    """双表同段交叉核验：判断两块同时计量同一用量的水表是否可能一致。

    复用现有秒级时间戳、读数范围、模数与速率校验；另要求两表起止时刻
    分别相同（允许不同时区表达，换算 UTC 后比较）。各表先各自求满足非负
    增量与速率上限的回绕整数区间（用量非负且不超速），再按两表读数差的
    模数同余联立共同用量（中国剩余定理），并与两表可行用量范围求交，
    返回共同用量的最小值、最大值、候选数与唯一性；同余不相容与范围无
    交集给出不同冲突原因。全程整数区间与同余运算，不逐个枚举回绕圈数。
    """
    solved = {
        "a": _solve_meter_block("old", req.meter_a),
        "b": _solve_meter_block("new", req.meter_b),
    }
    blocks = {"a": req.meter_a, "b": req.meter_b}
    elapsed = int(
        (req.meter_a.end.timestamp - req.meter_a.start.timestamp).total_seconds()
    )
    modulus_gcd = math.gcd(req.meter_a.modulus, req.meter_b.modulus)
    usage_step = req.meter_a.modulus // modulus_gcd * req.meter_b.modulus

    # 1) 某表段本身无解（用量无法同时非负且不超速）：分段冲突，不做联立
    segment_conflicts: List[DualMeterSegmentErrorOut] = []
    feasible: dict = {}
    for label in ("a", "b"):
        block = blocks[label]
        r_start, r_end, _, sol = solved[label]
        if sol.status == STATUS_NO_SOLUTION:
            segment_conflicts.append(
                DualMeterSegmentErrorOut(
                    kind="segment",
                    meter=label,  # type: ignore[arg-type]
                    modulus=block.modulus,
                    start_reading=r_start.value,
                    end_reading=r_end.value,
                    reason=STATUS_NO_SOLUTION,
                    detail=_no_solution_detail(
                        r_end.value - r_start.value, block.modulus, sol, elapsed
                    ),
                    min_wraps=sol.min_wraps,
                    max_wraps=sol.max_wraps,
                )
            )
        else:
            feasible[label] = (block, r_start, r_end, sol)

    if segment_conflicts:
        return DualMeterCompareResponse(
            status=STATUS_CONFLICT,
            start_timestamp=req.meter_a.start.timestamp,
            end_timestamp=req.meter_a.end.timestamp,
            elapsed_seconds=elapsed,
            modulus_gcd=modulus_gcd,
            usage_step=usage_step,
            min_usage=None,
            max_usage=None,
            candidate_count=0,
            unique=False,
            meters=[],
            conflicts=segment_conflicts,
        )

    # 2) 两表各自可行：按读数差同余联立共同用量，并与两表可行范围求交
    a_block, a_start, a_end, a_sol = feasible["a"]
    b_block, b_start, b_end, b_sol = feasible["b"]
    a_diff = a_end.value - a_start.value
    b_diff = b_end.value - b_start.value
    result = compare_dual_meter(
        a_diff=a_diff,
        a_bounds=(a_sol.min_wraps, a_sol.max_wraps),
        a_modulus=a_block.modulus,
        b_diff=b_diff,
        b_bounds=(b_sol.min_wraps, b_sol.max_wraps),
        b_modulus=b_block.modulus,
    )

    if result.status == STATUS_CONFLICT:
        common_conflicts: List[DualMeterCommonErrorOut] = []
        for reason in result.conflicts:
            if reason == REASON_INCOMPATIBLE_MODULUS:
                expected: Optional[int] = a_diff % result.gcd
                detail = (
                    f"两表读数差同余不相容：共同用量须同时满足 "
                    f"u ≡ {a_diff} (mod {a_block.modulus}) 且 "
                    f"u ≡ {b_diff} (mod {b_block.modulus})，即两读数差模 "
                    f"gcd({a_block.modulus}, {b_block.modulus}) = {result.gcd} "
                    f"同余；但二者模 {result.gcd} 的余数分别为 "
                    f"{a_diff % result.gcd}、{b_diff % result.gcd}，"
                    f"不存在任何共同用量，两表不可能一致"
                )
            else:  # REASON_RANGE_DISJOINT
                expected = None
                if result.intersect_min > result.intersect_max:
                    detail = (
                        f"两表可行用量范围无交集：表 A 可行用量 "
                        f"[{result.a_usage_min}, {result.a_usage_max}]"
                        f"（模数 {a_block.modulus}），表 B 可行用量 "
                        f"[{result.b_usage_min}, {result.b_usage_max}]"
                        f"（模数 {b_block.modulus}），区间交集为空，"
                        f"不存在共同用量，两表不可能一致"
                    )
                else:
                    detail = (
                        f"两表可行用量范围之交 [{result.intersect_min}, "
                        f"{result.intersect_max}] 内无共同用量：共同用量须满足 "
                        f"u ≡ {result.residue} (mod {result.step})，"
                        f"不小于交集下界的最小候选 {result.nearest_candidate} "
                        f"已超出交集上界 {result.intersect_max}，两表不可能一致"
                    )
            common_conflicts.append(
                DualMeterCommonErrorOut(
                    kind="common",
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    modulus_gcd=result.gcd,
                    usage_step=result.step,
                    expected_remainder=expected,
                    a_usage_min=result.a_usage_min,
                    a_usage_max=result.a_usage_max,
                    b_usage_min=result.b_usage_min,
                    b_usage_max=result.b_usage_max,
                    intersect_min=result.intersect_min,
                    intersect_max=result.intersect_max,
                )
            )
        return DualMeterCompareResponse(
            status=STATUS_CONFLICT,
            start_timestamp=req.meter_a.start.timestamp,
            end_timestamp=req.meter_a.end.timestamp,
            elapsed_seconds=elapsed,
            modulus_gcd=modulus_gcd,
            usage_step=usage_step,
            min_usage=None,
            max_usage=None,
            candidate_count=0,
            unique=False,
            meters=[],
            conflicts=common_conflicts,
        )

    # 3) 可行：共同候选为 min_usage + i×usage_step 的等差数列，不逐个枚举
    meters = [
        DualMeterSegmentOut(
            meter="a",
            modulus=a_block.modulus,
            start_reading=a_start.value,
            end_reading=a_end.value,
            min_wraps=a_sol.min_wraps,
            max_wraps=a_sol.max_wraps,
            min_usage=result.a_usage_min,
            max_usage=result.a_usage_max,
        ),
        DualMeterSegmentOut(
            meter="b",
            modulus=b_block.modulus,
            start_reading=b_start.value,
            end_reading=b_end.value,
            min_wraps=b_sol.min_wraps,
            max_wraps=b_sol.max_wraps,
            min_usage=result.b_usage_min,
            max_usage=result.b_usage_max,
        ),
    ]
    return DualMeterCompareResponse(
        status=STATUS_OK,
        start_timestamp=req.meter_a.start.timestamp,
        end_timestamp=req.meter_a.end.timestamp,
        elapsed_seconds=elapsed,
        modulus_gcd=modulus_gcd,
        usage_step=usage_step,
        min_usage=result.min_usage,
        max_usage=result.max_usage,
        candidate_count=result.candidate_count,
        unique=result.unique,
        meters=meters,
        conflicts=[],
    )
