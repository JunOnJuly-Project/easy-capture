"""ui/segment_table 순수 로직 단위 테스트 (PySide6 비의존).

대상 모듈: easy_capture.ui.segment_table (미구현 — import 실패 RED 정상)

TDD RED 단계:
  - 프로덕션 코드 없으므로 전 케이스 xfail(RED) 예상
  - PySide6 import 없이 stdlib·SpeedSegment·normalize_segments만 사용
  - 기존 379개 테스트 무회귀: 이 파일만 신규 추가, 기존 파일 무수정

테스트 범위:
  1. PRESET_FACTORS — 배속 프리셋 상수 계약
  2. rows_to_segments — 테이블 행 데이터 → SpeedSegment 튜플 변환
  3. dynamic_fast_cap — GIF 패스트 상한 동적 계산 (base_fps 의존)

WHY try/except + xfail:
  모듈이 없으면 import 자체가 ModuleNotFoundError를 던진다.
  격리해 기존 테스트 컬렉션 차단을 막으면서,
  _HAS_SEGMENT_TABLE=False 시 각 테스트 첫 줄 _require_segment_table()가
  pytest.xfail()을 호출해 XFAIL(=예상된 RED) 양상을 명확히 노출한다.
  구현 완료 시 xfail → GREEN 자동 전환(strict=False).
"""
from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# 모듈 격리 — 미구현 시 XFAIL RED (기존 테스트 차단 없음)
# ---------------------------------------------------------------------------
try:
    from easy_capture.ui.segment_table import (
        PRESET_FACTORS,
        dynamic_fast_cap,
        rows_to_segments,
    )
    _HAS_SEGMENT_TABLE = True
except ModuleNotFoundError:
    PRESET_FACTORS = None  # type: ignore[assignment]
    rows_to_segments = None  # type: ignore[assignment]
    dynamic_fast_cap = None  # type: ignore[assignment]
    _HAS_SEGMENT_TABLE = False

# SpeedSegment·normalize_segments는 이미 구현되어 있음 (timeremap.py)
try:
    from easy_capture.core.timing.timeremap import (
        SpeedSegment,
        normalize_segments,
    )
    _HAS_TIMEREMAP = True
except ModuleNotFoundError:
    SpeedSegment = None  # type: ignore[assignment]
    normalize_segments = None  # type: ignore[assignment]
    _HAS_TIMEREMAP = False

_MSG_NOT_IMPL = "easy_capture.ui.segment_table 미구현 — RED 예상"


def _require_segment_table() -> None:
    """테스트 본문 첫 줄에 호출 — 미구현이면 xfail로 RED 표시.

    WHY: pytest.xfail(strict=False)로 "예상된 실패"를 명확히 표현한다.
         구현 완료 시 자동으로 XPASS → 데코레이터 제거만 하면 GREEN.
    """
    if not _HAS_SEGMENT_TABLE:
        pytest.xfail(_MSG_NOT_IMPL)


# ---------------------------------------------------------------------------
# 테스트 상수 — 매직넘버 금지
# ---------------------------------------------------------------------------
# 기대 프리셋 배속값 (계획서 Story 5 Task 5-1)
EXPECTED_PRESET_SLOW_QUARTER: float = 0.25   # 4배 슬로우
EXPECTED_PRESET_SLOW_THIRD: float = 0.33     # 3배 슬로우 (≈1/3)
EXPECTED_PRESET_SLOW_HALF: float = 0.5       # 2배 슬로우
EXPECTED_PRESET_NORMAL: float = 1.0          # 등속
EXPECTED_PRESET_FAST_DOUBLE: float = 2.0     # 2배 패스트

# dynamic_fast_cap 공식 상수
# GIF 패스트 상한 = min(절대상한, 50 / base_fps)
# WHY: duration_ms = (1000/fps) / factor ≥ 20ms
#      → factor ≤ 1000 / (fps × 20) = 50 / fps
#      50 = 1000ms ÷ GIF_DURATION_CLAMP_MS(20ms). "fps × 0.02"는 오기(역수).
DYNAMIC_CAP_NUMERATOR: float = 50.0         # 50 / base_fps = GIF 패스트 상한 분자

# 절대 상한: factor > 4.0은 SpeedSegment 검증에서 막히므로 4.0이 자연 상한
# WHY: dynamic_fast_cap이 4.0보다 클 수 없다 (timeremap FACTOR_MAX=4.0과 정합)
ABSOLUTE_CAP: float = 4.0

# 테스트용 fps 값
FPS_LOW: float = 12.0    # 저fps — min(4.0, 50/12) = 4.0 (절대상한 클램프)
FPS_NORMAL: float = 50.0  # 50fps — min(4.0, 50/50) = 1.0
FPS_HIGH: float = 120.0   # 고fps — min(4.0, 50/120) = 0.417

# 부동소수 비교 허용 오차
APPROX_REL: float = 1e-6
APPROX_ABS: float = 0.01  # 0.33 같은 무한소수에 절대 허용치 추가

# 유효 테이블 행 데이터 형식: list[tuple[int, int, float]] = (start, end, factor)
ROW_SINGLE: list[tuple[int, int, float]] = [(0, 10, 0.5)]
ROW_MULTI: list[tuple[int, int, float]] = [(0, 5, 0.5), (10, 15, 2.0), (20, 25, 1.0)]
ROW_EMPTY: list[tuple[int, int, float]] = []


# ===========================================================================
# 1. PRESET_FACTORS — 배속 프리셋 상수
# ===========================================================================
class TestPresetFactors:
    """PRESET_FACTORS: 배속 프리셋 컬렉션 계약.

    WHY: ComboBox에 표시될 프리셋이 명세와 일치해야
         사용자가 기대하는 배속 옵션을 선택할 수 있다.
    """

    def test_PRESET_FACTORS가_0_25를_포함한다(self):
        """Given: PRESET_FACTORS
        When:  멤버 확인
        Then:  0.25(4배 슬로우)가 포함된다.

        WHY: 0.25x는 연사 장면의 슈퍼 슬로우모션 핵심 프리셋.
             계획서 Task 5-1 명시.
        """
        _require_segment_table()

        assert EXPECTED_PRESET_SLOW_QUARTER in PRESET_FACTORS

    def test_PRESET_FACTORS가_0_5를_포함한다(self):
        """Given: PRESET_FACTORS
        When:  멤버 확인
        Then:  0.5(2배 슬로우)가 포함된다.

        WHY: 0.5x는 슬로우모션의 기본 프리셋으로 가장 많이 사용됨.
        """
        _require_segment_table()

        assert EXPECTED_PRESET_SLOW_HALF in PRESET_FACTORS

    def test_PRESET_FACTORS가_1_0을_포함한다(self):
        """Given: PRESET_FACTORS
        When:  멤버 확인
        Then:  1.0(등속)이 포함된다.

        WHY: 등속은 "구간 해제" 역할 — 선택 취소 없이 1.0으로 되돌릴 수 있어야 함.
        """
        _require_segment_table()

        assert EXPECTED_PRESET_NORMAL in PRESET_FACTORS

    def test_PRESET_FACTORS가_2_0을_포함한다(self):
        """Given: PRESET_FACTORS
        When:  멤버 확인
        Then:  2.0(2배 패스트)이 포함된다.

        WHY: 2.0x는 패스트포워드 핵심 프리셋.
        """
        _require_segment_table()

        assert EXPECTED_PRESET_FAST_DOUBLE in PRESET_FACTORS

    def test_PRESET_FACTORS가_0_33을_포함한다(self):
        """Given: PRESET_FACTORS
        When:  멤버 확인
        Then:  약 0.33(3배 슬로우)이 포함된다.

        WHY: 계획서 설계 명시 "0.5↔0.25 사이 0.33 존재" — 3배 슬로우 프리셋.
             부동소수 비교: 절대 허용치(0.01) 적용.
        """
        _require_segment_table()

        # 0.33 또는 1/3에 가까운 값이 포함되어야 함
        has_third = any(
            abs(f - EXPECTED_PRESET_SLOW_THIRD) <= APPROX_ABS
            for f in PRESET_FACTORS
        )
        assert has_third, (
            f"PRESET_FACTORS에 약 0.33이 없음. 현재값: {sorted(PRESET_FACTORS)}"
        )

    def test_0_33이_0_25와_0_5_사이에_위치한다(self):
        """Given: PRESET_FACTORS (정렬됨)
        When:  0.33 위치 확인
        Then:  0.25 < 0.33 < 0.5 순서 보장.

        WHY: 계획서 설계 명시 "0.5↔0.25 사이 0.33 존재" — 정렬 순서 계약.
        """
        _require_segment_table()

        sorted_factors = sorted(PRESET_FACTORS)
        idx_quarter = None
        idx_third = None
        idx_half = None

        for i, f in enumerate(sorted_factors):
            if abs(f - EXPECTED_PRESET_SLOW_QUARTER) <= APPROX_ABS:
                idx_quarter = i
            elif abs(f - EXPECTED_PRESET_SLOW_THIRD) <= APPROX_ABS:
                idx_third = i
            elif abs(f - EXPECTED_PRESET_SLOW_HALF) <= APPROX_ABS:
                idx_half = i

        assert idx_quarter is not None, "0.25 프리셋 없음"
        assert idx_third is not None, "0.33 프리셋 없음"
        assert idx_half is not None, "0.5 프리셋 없음"
        assert idx_quarter < idx_third < idx_half, (
            f"정렬 순서 오류: 0.25(idx={idx_quarter}), "
            f"0.33(idx={idx_third}), 0.5(idx={idx_half})"
        )

    def test_PRESET_FACTORS는_비어있지_않다(self):
        """Given: PRESET_FACTORS
        When:  길이 확인
        Then:  1개 이상의 프리셋 포함.
        """
        _require_segment_table()

        assert len(PRESET_FACTORS) > 0, "PRESET_FACTORS가 비어있음"

    def test_PRESET_FACTORS_모든_값은_양수이다(self):
        """Given: PRESET_FACTORS
        When:  각 값 부호 확인
        Then:  모두 > 0.

        WHY: 0/음수 배속은 SpeedSegment 검증에서 ValueError이므로
             프리셋에 포함되어선 안 된다.
        """
        _require_segment_table()

        for f in PRESET_FACTORS:
            assert f > 0, f"비양수 프리셋 발견: {f}"

    def test_PRESET_FACTORS_4x는_기본_포함이_아니다(self):
        """Given: PRESET_FACTORS
        When:  4.0 포함 여부 확인
        Then:  4.0은 기본 프리셋에 없다(고급 옵션).

        WHY: 계획서 Task 5-3 "4x는 고급 옵션화 — 기본 제외".
             GIF에서 4x는 8.3ms → 20ms 클램프로 실질 무의미, UI에서 혼란 방지.
        """
        _require_segment_table()

        assert 4.0 not in PRESET_FACTORS, (
            "4.0x가 기본 PRESET_FACTORS에 포함됨 — 고급 옵션으로 분리해야 함"
        )


# ===========================================================================
# 2. rows_to_segments — 테이블 행 데이터 → SpeedSegment 튜플 변환
# ===========================================================================
class TestRowsToSegments:
    """rows_to_segments(rows) -> tuple[SpeedSegment, ...]: 변환 계약.

    WHY: UI 테이블 행(start, end, factor)을 도메인 객체로 변환하고
         normalize_segments를 통해 겹침/역전/범위 검증까지 수행한다.
         순수 변환이므로 PySide6 비의존으로 단위 테스트 가능해야 한다.
    """

    def test_빈_행_리스트는_빈_튜플을_반환한다(self):
        """Given: rows=[] (빈 테이블)
        When:  rows_to_segments 호출
        Then:  () 빈 튜플 반환.

        WHY: 구간 없음 = 타임리맵 비적용 — 항등 경로의 시작.
             segments=() → export 무회귀 계약.
        """
        _require_segment_table()

        result = rows_to_segments(ROW_EMPTY)

        assert result == ()

    def test_단일_행이_SpeedSegment로_변환된다(self):
        """Given: rows=[(0, 10, 0.5)]
        When:  rows_to_segments 호출
        Then:  (SpeedSegment(start=0, end=10, factor=0.5),) 반환.

        WHY: 기본 변환 경로 검증 — start/end/factor가 정확히 매핑되어야 한다.
        """
        _require_segment_table()

        result = rows_to_segments([(0, 10, 0.5)])

        assert len(result) == 1
        seg = result[0]
        assert seg.start == 0
        assert seg.end == 10
        assert seg.factor == pytest.approx(0.5, rel=APPROX_REL)

    def test_다중_행이_정렬된_SpeedSegment_튜플로_반환된다(self):
        """Given: rows=[(20,25,1.0),(0,5,0.5),(10,15,2.0)] (역순)
        When:  rows_to_segments 호출
        Then:  start 기준 오름차순 정렬된 SpeedSegment 튜플 반환.

        WHY: normalize_segments가 정렬을 수행해야 한다.
             테이블 행 순서에 무관하게 정렬된 결과 보장.
        """
        _require_segment_table()

        rows_unordered: list[tuple[int, int, float]] = [
            (20, 25, 1.0),
            (0, 5, 0.5),
            (10, 15, 2.0),
        ]
        result = rows_to_segments(rows_unordered)

        assert len(result) == 3
        assert result[0].start == 0
        assert result[1].start == 10
        assert result[2].start == 20

    def test_반환값이_tuple_타입이다(self):
        """Given: 유효한 단일 행
        When:  rows_to_segments 호출
        Then:  반환값이 tuple이다(list 아님).

        WHY: VideoExportConfig.segments가 tuple을 기대하므로
             변환 결과가 tuple이어야 type 안정성이 보장된다.
        """
        _require_segment_table()

        result = rows_to_segments(ROW_SINGLE)

        assert isinstance(result, tuple), f"tuple이 아님: {type(result)}"

    def test_반환_원소가_SpeedSegment_인스턴스이다(self):
        """Given: 유효한 행 리스트
        When:  rows_to_segments 호출
        Then:  각 원소가 SpeedSegment 인스턴스이다.

        WHY: 도메인 타입 정확성 — timeremap이 SpeedSegment를 소비한다.
        """
        _require_segment_table()
        if not _HAS_TIMEREMAP:
            pytest.skip("timeremap 미구현 — SpeedSegment 타입 검사 불가")

        result = rows_to_segments(ROW_MULTI)

        for seg in result:
            assert isinstance(seg, SpeedSegment), f"SpeedSegment가 아님: {type(seg)}"

    def test_float_factor가_정확하게_변환된다(self):
        """Given: rows=[(5, 15, 0.33)]
        When:  rows_to_segments 호출
        Then:  factor == 0.33 (부동소수 정밀도 보존).

        WHY: UI ComboBox에서 float 프리셋 값을 선택 시
             부동소수가 왜곡 없이 SpeedSegment에 전달되어야 한다.
        """
        _require_segment_table()

        result = rows_to_segments([(5, 15, 0.33)])

        assert result[0].factor == pytest.approx(0.33, abs=APPROX_ABS)

    def test_겹치는_행이면_ValueError를_전파한다(self):
        """Given: rows=[(0,10,0.5),(5,15,2.0)] (겹침)
        When:  rows_to_segments 호출
        Then:  ValueError 발생 (normalize_segments 검증 전파).

        WHY: UI는 이 예외를 잡아 한국어 QMessageBox를 표시한다.
             변환 함수가 검증을 위임·전파해야 UI가 올바르게 처리할 수 있다.
        """
        _require_segment_table()

        rows_overlapping: list[tuple[int, int, float]] = [(0, 10, 0.5), (5, 15, 2.0)]

        with pytest.raises(ValueError, match="겹침|중복|overlap"):
            rows_to_segments(rows_overlapping)

    def test_역전_행이면_ValueError를_전파한다(self):
        """Given: rows=[(10, 5, 0.5)] (start > end 역전)
        When:  rows_to_segments 호출
        Then:  ValueError 발생 (normalize_segments 역전 검증 전파).

        WHY: 사용자가 실수로 start > end를 입력했을 때 UI 안내가 가능하도록
             예외가 전파되어야 한다.
        """
        _require_segment_table()

        rows_reversed: list[tuple[int, int, float]] = [(10, 5, 0.5)]

        with pytest.raises(ValueError, match="역전|시작|start"):
            rows_to_segments(rows_reversed)

    def test_범위_밖_factor면_ValueError를_전파한다(self):
        """Given: rows=[(0,10,5.0)] (factor=5.0, 상한 4.0 초과)
        When:  rows_to_segments 호출
        Then:  ValueError 발생 (normalize_segments factor 범위 검증 전파).

        WHY: 4x 초과 배속은 timeremap이 허용하지 않는다.
             UI ComboBox가 프리셋 외 값을 허용한 경우 방어.
        """
        _require_segment_table()

        rows_bad_factor: list[tuple[int, int, float]] = [(0, 10, 5.0)]

        with pytest.raises(ValueError, match="배속|factor|범위"):
            rows_to_segments(rows_bad_factor)

    def test_ValueError_메시지가_한국어를_포함한다(self):
        """Given: 역전 구간 행
        When:  rows_to_segments 호출 → ValueError
        Then:  에러 메시지에 한국어가 포함된다.

        WHY: 글로벌 CLAUDE.md 언어 정책 — 사용자 대면 에러는 한국어.
        """
        _require_segment_table()

        rows_bad: list[tuple[int, int, float]] = [(10, 2, 0.5)]

        with pytest.raises(ValueError) as exc_info:
            rows_to_segments(rows_bad)

        msg = str(exc_info.value)
        has_korean = any(ord(c) >= 0xAC00 for c in msg)
        assert has_korean, f"에러 메시지에 한국어 없음: {msg!r}"

    def test_다중_유효_행의_각_필드가_올바르게_매핑된다(self):
        """Given: rows=[(0,5,0.5),(10,15,2.0),(20,25,1.0)]
        When:  rows_to_segments 호출
        Then:  각 SpeedSegment의 start/end/factor가 행 데이터와 일치한다.

        WHY: 다중 행에서 인덱스 혼용 없이 각 행이 올바른 Segment로 변환됨 보장.
        """
        _require_segment_table()

        rows: list[tuple[int, int, float]] = [(0, 5, 0.5), (10, 15, 2.0), (20, 25, 1.0)]
        result = rows_to_segments(rows)

        # 정렬 후에도 원본 데이터가 보존되어야 함
        expected = {(0, 5, 0.5), (10, 15, 2.0), (20, 25, 1.0)}
        actual = {(seg.start, seg.end, seg.factor) for seg in result}
        assert actual == expected, f"필드 매핑 오류. 예상: {expected}, 실제: {actual}"


# ===========================================================================
# 3. dynamic_fast_cap — GIF 패스트 상한 동적 계산
# ===========================================================================
class TestDynamicFastCap:
    """dynamic_fast_cap(base_fps) -> float: GIF 패스트 상한 계산 계약.

    공식: min(ABSOLUTE_CAP, 50 / base_fps)
    WHY: duration_ms = (1000/fps) / factor ≥ 20ms
         → factor ≤ 1000 / (fps × 20) = 50 / fps
         50 = 1000ms ÷ GIF_DURATION_CLAMP_MS(20ms).
         절대 상한 4.0 클램프로 SpeedSegment FACTOR_MAX와 정합.
    """

    def test_base_12fps에서_패스트_상한이_절대상한에_클램프된다(self):
        """Given: base_fps=12.0 (저fps)
        When:  dynamic_fast_cap(12.0) 호출
        Then:  cap = min(4.0, 50/12.0) = min(4.0, 4.17) = 4.0 (절대상한 클램프).

        WHY: 12fps에서 50/12 = 4.17이 FACTOR_MAX(4.0)을 초과하므로 4.0으로 클램프.
             duration = (1000/12)/4.0 = 20.8ms ≥ 20ms 보장.
             저fps에서도 불변식이 만족됨을 검증.
        """
        _require_segment_table()

        expected_cap = min(ABSOLUTE_CAP, DYNAMIC_CAP_NUMERATOR / FPS_LOW)
        result = dynamic_fast_cap(FPS_LOW)

        assert result == pytest.approx(expected_cap, rel=APPROX_REL), (
            f"12fps cap 오류: 예상 {expected_cap}, 실제 {result}"
        )

    def test_base_50fps에서_패스트_상한이_1_0이다(self):
        """Given: base_fps=50.0
        When:  dynamic_fast_cap(50.0) 호출
        Then:  cap = min(4.0, 50/50.0) = min(4.0, 1.0) = 1.0.

        WHY: 50fps에서 50/50 = 1.0 → 1x 배속까지만 GIF 안전.
             duration = (1000/50)/1.0 = 20ms — 20ms 경계에 딱 맞는 대표 케이스.
             계획서 설계 명시 "base 50fps → 1.0".
        """
        _require_segment_table()

        expected_cap = min(ABSOLUTE_CAP, DYNAMIC_CAP_NUMERATOR / FPS_NORMAL)
        result = dynamic_fast_cap(FPS_NORMAL)

        assert result == pytest.approx(expected_cap, rel=APPROX_REL), (
            f"50fps cap 오류: 예상 {expected_cap}, 실제 {result}"
        )

    def test_base_120fps에서_상한이_낮다(self):
        """Given: base_fps=120.0 (고fps)
        When:  dynamic_fast_cap(120.0) 호출
        Then:  min(4.0, 50/120) = min(4.0, 0.417) = 0.417.

        WHY: 120fps에서는 1프레임 = 8.3ms. 불변식 factor ≤ 50/120 ≈ 0.417.
             duration = (1000/120)/0.417 = 20.0ms — 정확히 20ms 경계.
             고fps일수록 패스트 배속이 더 제한됨(짧은 프레임 시간 때문).
        """
        _require_segment_table()

        expected_cap = min(ABSOLUTE_CAP, DYNAMIC_CAP_NUMERATOR / FPS_HIGH)
        result = dynamic_fast_cap(FPS_HIGH)

        assert result == pytest.approx(expected_cap, rel=APPROX_REL), (
            f"120fps cap 오류: 예상 {expected_cap}, 실제 {result}"
        )

    def test_fps가_증가하면_cap이_단조_감소한다(self):
        """Given: base_fps = 12, 24, 30, 60, 120 (오름차순)
        When:  dynamic_fast_cap 각각 호출
        Then:  절대상한 클램프 구간 이후 cap 값이 단조 감소.

        WHY: 공식 50/fps — fps가 커질수록 cap이 작아진다.
             고fps일수록 프레임 시간이 짧아 20ms 보장을 위해 더 낮은 배속 상한이 필요.
             단조감소 보장 = "더 좋은 카메라라도 GIF 패스트는 더 제한된다" 직관 표현.
             절대상한(4.0) 클램프 구간에서는 평탄(==)이 허용된다.
        """
        _require_segment_table()

        fps_list = [12.0, 24.0, 30.0, 60.0, 120.0]
        caps = [dynamic_fast_cap(fps) for fps in fps_list]

        for i in range(len(caps) - 1):
            assert caps[i] >= caps[i + 1], (
                f"단조감소 위반: fps={fps_list[i]}→cap={caps[i]}, "
                f"fps={fps_list[i+1]}→cap={caps[i+1]}"
            )

    def test_반환값은_양수이다(self):
        """Given: 임의의 양수 base_fps
        When:  dynamic_fast_cap 호출
        Then:  반환값 > 0.

        WHY: cap=0이면 어떤 패스트 배속도 허용 안 됨 — 의미없는 상한.
        """
        _require_segment_table()

        fps_cases = [12.0, 30.0, 60.0]
        for fps in fps_cases:
            result = dynamic_fast_cap(fps)
            assert result > 0, f"fps={fps}에서 cap={result} ≤ 0"

    def test_반환값이_절대상한_4_0을_초과하지_않는다(self):
        """Given: 매우 높은 base_fps (1000.0)
        When:  dynamic_fast_cap(1000.0) 호출
        Then:  반환값 ≤ 4.0 (ABSOLUTE_CAP).

        WHY: SpeedSegment FACTOR_MAX=4.0 초과 값은 normalize_segments에서
             ValueError가 된다. cap이 4.0을 초과하면 "가능" 안내 후
             실제 export 시 오류 발생하는 UX 혼란이 생긴다.
        """
        _require_segment_table()

        very_high_fps = 1000.0
        result = dynamic_fast_cap(very_high_fps)

        assert result <= ABSOLUTE_CAP, (
            f"절대상한 초과: dynamic_fast_cap({very_high_fps}) = {result} > {ABSOLUTE_CAP}"
        )

    def test_공식_50_나누기_base_fps와_일치한다(self):
        """Given: base_fps=30.0 (일반 카메라)
        When:  dynamic_fast_cap(30.0) 호출
        Then:  min(4.0, 50/30.0) = min(4.0, 1.667) = 1.667.

        WHY: 공식 검증 케이스 — 30fps 카메라에서 1.67x까지 GIF 안전 패스트.
             duration = (1000/30) / 1.667 = 20.0ms — 정확히 20ms 경계.
             50 = 1000ms ÷ 20ms(GIF_DURATION_CLAMP_MS) 유도.
        """
        _require_segment_table()

        base_fps_30 = 30.0
        expected = min(ABSOLUTE_CAP, DYNAMIC_CAP_NUMERATOR / base_fps_30)
        result = dynamic_fast_cap(base_fps_30)

        assert result == pytest.approx(expected, rel=APPROX_REL), (
            f"30fps 공식 검증 실패: 예상 {expected}, 실제 {result}"
        )

    def test_cap이_duration_20ms_이상을_보장한다(self):
        """Given: base_fps=30.0, cap = dynamic_fast_cap(30.0)
        When:  factor=cap으로 duration 계산
        Then:  duration = 1000/fps/cap ≥ 20ms.

        WHY: 핵심 불변식 — cap의 존재 이유.
             cap을 적용한 배속에서 GIF duration이 20ms 이상이어야
             뷰어 역전(100ms 강제) 방지가 의미있다.
        """
        _require_segment_table()

        fps_cases = [12.0, 24.0, 30.0, 60.0, 120.0]
        gif_min_duration_ms = 20.0  # GIF_DURATION_CLAMP_MS

        for fps in fps_cases:
            cap = dynamic_fast_cap(fps)
            if cap <= 0:
                continue  # 방어: cap=0은 별도 테스트에서 커버
            duration = 1000.0 / fps / cap
            assert duration >= gif_min_duration_ms - APPROX_ABS, (
                f"fps={fps}, cap={cap}: duration={duration:.2f}ms < 20ms — "
                "GIF 역전 방지 보장 실패"
            )
