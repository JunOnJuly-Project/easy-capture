"""컷별 오브젝트 선택 순수 모델 (수동 교정 슬라이스).

배경:
  멀티샷 군무에서 자동 재매칭(IoU, feat=None)이 needs_correction 82.7%로
  구조적 실패. 해결책: 각 컷(샷) 시작 프레임에서 후보를 검출해 사용자에게
  보여주고, 사용자가 추적 대상을 명시 선택(CutSelection) → 그 선택으로
  컷별 재추적. 자동 재매칭은 폴백으로만 남긴다(혼합 정책).

설계 원칙:
  - 순수 core — numpy/stdlib만. torch·transformers·PySide6·PyAV·scenedetect 비의존.
  - frozen dataclass로 사용자 선택 불변 보장(TrackResult·Detection 패턴 계승).
  - 분할/검증을 순수 함수로 분리해 app 오케스트레이션 메서드를 20줄 이내로 유지.
"""
from __future__ import annotations

from dataclasses import dataclass

# 샷 인덱스 하한 — 0-기반(음수 인덱스 차단용 상수)
_SHOT_INDEX_MIN = 0


@dataclass(frozen=True)
class CutSelection:
    """한 샷의 추적 대상을 사용자가 명시 지정한 단위(불변).

    shot_index: split_into_shots 결과의 샷 인덱스(0-기반).
    point:      사용자가 클릭한 좌표 (x, y) — SAM2 재초기화 add_click 입력.
    box:        선택 대상 전신 bbox (x1, y1, x2, y2) — SAM2 box 프롬프트 입력.
                default None(하위호환). box가 있으면 add_box 우선 디스패치한다.

    WHY: frozen=True로 사용자 선택이 실수로 덮어씌워지는 버그를 차단한다.
         (샷, 클릭점)을 한 단위로 묶어 매개변수 폭증을 방지한다.
         box: box 프롬프트(detect 전신 bbox→SAM2)로 중심점 1개(point)보다
         정확한 전신 마스크를 얻기 위해 전신 bbox를 함께 보관한다(Story D).
         default None으로 기존 (shot_index, point) 생성 코드를 깨지 않는다.
    """

    shot_index: int
    point: tuple[int, int]
    box: tuple[float, float, float, float] | None = None


def index_selections_by_shot(
    selections: list[CutSelection],
    n_shots: int,
) -> dict[int, tuple[int, int]]:
    """선택 리스트를 {shot_index: point} 딕셔너리로 매핑한다(순수).

    빈 리스트면 빈 딕셔너리를 반환한다(전 샷 자동 재매칭 폴백 신호).
    매핑 전 validate_selections로 범위·중복을 검증한다.

    Args:
        selections: 사용자 선택 리스트(빈 리스트 허용).
        n_shots:    전체 샷 수(범위 검증 기준).

    Returns:
        {shot_index: point} — 오케스트레이션이 샷 인덱스로 클릭점을 O(1) 조회.

    WHY: 검증을 내부에서 호출해 매핑 결과가 항상 유효 범위임을 보장한다.
    """
    validate_selections(selections, n_shots)
    return {s.shot_index: s.point for s in selections}


def validate_selections(
    selections: list[CutSelection],
    n_shots: int,
) -> None:
    """선택 리스트의 shot_index 범위·중복을 검증한다(순수).

    위반 시 한국어 ValueError(수치 포함)를 발생시킨다. 정상/빈 입력은 통과.

    Args:
        selections: 사용자 선택 리스트(빈 리스트 허용).
        n_shots:    전체 샷 수 — 유효 범위는 0 <= shot_index < n_shots.

    Raises:
        ValueError: shot_index 범위 위반 또는 중복 시(한국어 메시지).

    WHY: 존재하지 않는 샷·중복 선택은 오케스트레이션에서 IndexError·모호한
         동작을 유발한다. 진입 시점에 명시적 예외로 차단한다.
    """
    seen: set[int] = set()
    for selection in selections:
        _raise_if_out_of_range(selection.shot_index, n_shots)
        _raise_if_duplicate(selection.shot_index, seen)
        seen.add(selection.shot_index)


def _raise_if_out_of_range(shot_index: int, n_shots: int) -> None:
    """shot_index가 [0, n_shots) 범위를 벗어나면 한국어 ValueError 발생."""
    if not _SHOT_INDEX_MIN <= shot_index < n_shots:
        raise ValueError(
            f"샷 인덱스 {shot_index}가 유효 범위(0 이상 {n_shots} 미만)를 "
            "벗어났습니다."
        )


def _raise_if_duplicate(shot_index: int, seen: set[int]) -> None:
    """이미 본 shot_index가 다시 나오면 한국어 ValueError 발생."""
    if shot_index in seen:
        raise ValueError(
            f"샷 인덱스 {shot_index}가 중복 선택되었습니다. "
            "한 샷에는 하나의 추적 대상만 지정할 수 있습니다."
        )
