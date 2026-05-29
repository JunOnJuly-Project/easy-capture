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

    shot_index:      split_into_shots 결과의 샷 인덱스(0-기반).
    point:           사용자가 클릭한 좌표 (x, y) — SAM2 재초기화 add_click 입력.
    box:             선택 대상 전신 bbox (x1, y1, x2, y2) — SAM2 box 프롬프트 입력.
                     default None(하위호환). box가 있으면 add_box 우선 디스패치한다.
    negative_points: 옆 멤버 좌표 묶음 ((x, y), ...) — '대상 아님'(SAM2 label 0).
                     default ()(하위호환). 비어 있으면 negative 없이 추적한다.

    WHY: frozen=True로 사용자 선택이 실수로 덮어씌워지는 버그를 차단한다.
         (샷, 클릭점)을 한 단위로 묶어 매개변수 폭증을 방지한다.
         box: box 프롬프트(detect 전신 bbox→SAM2)로 중심점 1개(point)보다
         정확한 전신 마스크를 얻기 위해 전신 bbox를 함께 보관한다(Story D).
         negative_points: 군무 밀착 구간에서 box+positive만으론 대상+옆사람이 한
         덩어리로 합쳐진다. negative point(label 0)로 옆 멤버 경계를 가른다.
         default None/()으로 기존 (shot_index, point[, box]) 생성 코드를 깨지 않는다.
    """

    shot_index: int
    point: tuple[int, int]
    box: tuple[float, float, float, float] | None = None
    negative_points: tuple[tuple[int, int], ...] = ()


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


def validate_negative_points(
    selection: CutSelection,
    frame_size: tuple[int, int],
) -> None:
    """selection의 negative 좌표가 프레임 안이고 positive와 다른지 검증한다(순수).

    위반 시 한국어 ValueError를 발생시킨다. 빈 negatives는 통과(무회귀).

    Args:
        selection:  검증할 사용자 선택(negative_points·point 포함).
        frame_size: 프레임 크기 (W, H) — 유효 범위는 0 <= x < W, 0 <= y < H.

    Raises:
        ValueError: negative가 프레임 밖이거나 positive(point)와 동일 좌표일 때.

    WHY: 프레임 밖 좌표는 SAM2에 무의미하고 좌표계 오류를 유발한다.
         positive와 동일 좌표는 '대상이자 대상 아님'의 모순이므로 차단한다.
    """
    width, height = frame_size
    for negative in selection.negative_points:
        _raise_if_negative_out_of_bounds(negative, width, height)
        _raise_if_negative_equals_positive(negative, selection.point)


def _raise_if_negative_out_of_bounds(
    negative: tuple[int, int],
    width: int,
    height: int,
) -> None:
    """negative 좌표가 [0, W)×[0, H) 범위를 벗어나면 한국어 ValueError 발생."""
    x, y = negative
    if not (_SHOT_INDEX_MIN <= x < width and _SHOT_INDEX_MIN <= y < height):
        raise ValueError(
            f"negative 좌표 {negative}가 프레임(0 이상 {width}×{height} 미만)을 "
            "벗어났습니다."
        )


def _raise_if_negative_equals_positive(
    negative: tuple[int, int],
    point: tuple[int, int],
) -> None:
    """negative 좌표가 positive(point)와 동일하면 한국어 ValueError 발생."""
    if tuple(negative) == tuple(point):
        raise ValueError(
            f"negative 좌표 {negative}가 대상 클릭점과 동일합니다. "
            "같은 점을 대상이자 대상 아님으로 동시에 지정할 수 없습니다."
        )


# 중심 좌표 분모 — (좌표합)/2 (매직넘버 금지)
_CENTER_DIVISOR = 2


def box_center(box: tuple[float, float, float, float]) -> tuple[int, int]:
    """박스 (x1, y1, x2, y2)의 정수 중심 (cx, cy)을 반환한다(순수).

    cx = int((x1 + x2) / 2), cy = int((y1 + y2) / 2) — float 좌표를 정수로 절단.

    Args:
        box: 전신 bbox (x1, y1, x2, y2).

    Returns:
        (cx, cy) — SAM2 add_click(point) 입력용 정수 픽셀 좌표.

    WHY: SAM2 클릭점은 정수 픽셀이어야 한다. 노트북 셀 7.5 `_box_center`의
         int(...) 절단(내림)을 그대로 보존한다(데스크톱 UI 변환 위임).
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / _CENTER_DIVISOR), int((y1 + y2) / _CENTER_DIVISOR)


def pick_box_at(
    point: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
) -> int | None:
    """점을 포함하는 후보 박스의 인덱스를 반환한다(겹침 시 최소 넓이, 밖이면 None).

    Args:
        point: 클릭 좌표 (x, y).
        boxes: 후보 박스 리스트 (x1, y1, x2, y2)들.

    Returns:
        점을 포함하는 박스 중 가장 작은 넓이의 인덱스, 없으면 None.

    WHY: 데스크톱 UI는 클릭 좌표만 안다. 겹친 박스에서 안쪽(작은 박스)을
         우선해 군무 밀착 구간의 모호성을 해소한다(히트테스트를 core가 담당).
    """
    containing = [i for i, box in enumerate(boxes) if _contains(box, point)]
    if not containing:
        return None
    return min(containing, key=lambda i: _box_area(boxes[i]))


def _contains(
    box: tuple[float, float, float, float],
    point: tuple[int, int],
) -> bool:
    """점 (x, y)이 박스 [x1, x2] × [y1, y2] 경계 안(포함)인지 판정한다."""
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_area(box: tuple[float, float, float, float]) -> float:
    """박스 넓이 (x2 - x1) × (y2 - y1)를 반환한다(겹침 시 최소 선택 기준)."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True)
class ShotChoice:
    """한 샷의 사용자 선택(인덱스만) — UI/core 경계의 입력 DTO(불변).

    target_idx:    대상 후보 인덱스(None=미선택→자동 재매칭 폴백).
    negative_idxs: 배제할 옆 멤버 후보 인덱스 묶음(default 빈 튜플).

    WHY: 데스크톱 UI는 인덱스(int)만 다룬다. core가 ShotChoice(인덱스)를
         CutSelection(좌표·box)으로 변환해 core→app 역참조를 회피한다.
    """

    target_idx: int | None
    negative_idxs: tuple[int, ...] = ()


def build_selections_from_choices(
    candidate_boxes: list[list[tuple[float, float, float, float]]],
    choices: dict[int, ShotChoice],
) -> list[CutSelection]:
    """샷별 후보 박스 + 선택(인덱스) → 검증된 CutSelection 리스트로 변환한다(순수).

    shot_index 오름차순으로 순회한다. target_idx가 None이거나 후보 수를 초과하면
    그 샷은 건너뛴다(자동 재매칭 폴백). 빌드 후 validate_selections로 검증한다.

    Args:
        candidate_boxes: 샷별 후보 박스 리스트(box만 — app DTO 비참조).
        choices:         {shot_index: ShotChoice} 사용자 선택 매핑.

    Returns:
        shot_index 오름차순 정렬된 CutSelection 리스트.

    WHY: 노트북 셀 7.5 변환 루프 이식. 딕셔너리 순서 의존을 제거해 결정적
         출력을 보장하고, 빌드 결과가 항상 범위·중복 유효함을 검증으로 보장한다.
    """
    selections = []
    for shot_index in sorted(choices):
        selection = _build_one_selection(
            shot_index, candidate_boxes[shot_index], choices[shot_index]
        )
        if selection is not None:
            selections.append(selection)
    validate_selections(selections, len(candidate_boxes))
    return selections


def _build_one_selection(
    shot_index: int,
    boxes: list[tuple[float, float, float, float]],
    choice: ShotChoice,
) -> CutSelection | None:
    """한 샷의 ShotChoice를 CutSelection으로 변환한다(미선택·범위밖이면 None).

    target_idx가 None이거나 후보 수를 초과하면 None(폴백)을 반환한다.
    negative_idxs는 자기 자신·범위 밖 인덱스를 제외하고 box 중심으로 변환한다.
    """
    target_idx = choice.target_idx
    if target_idx is None or not 0 <= target_idx < len(boxes):
        return None
    box = boxes[target_idx]
    negatives = _build_negative_points(boxes, choice.negative_idxs, target_idx)
    return CutSelection(
        shot_index=shot_index,
        point=box_center(box),
        box=box,
        negative_points=negatives,
    )


def _build_negative_points(
    boxes: list[tuple[float, float, float, float]],
    negative_idxs: tuple[int, ...],
    target_idx: int,
) -> tuple[tuple[int, int], ...]:
    """negative 후보 인덱스를 box 중심으로 변환한다(자기·범위밖 제외, 셀 7.5 규칙)."""
    return tuple(
        box_center(boxes[ni])
        for ni in negative_idxs
        if ni != target_idx and 0 <= ni < len(boxes)
    )
