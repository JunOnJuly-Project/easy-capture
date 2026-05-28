"""크롭 크기 변환 순수 함수.

슬라이더 비율(정수 %) → 실제 픽셀 크기(짝수, 하한 보장).
PySide6 비의존 — UI 없이 단위 테스트 가능하도록 분리.

WHY: box_size=(300,300) 매직넘버를 제거하고 프레임 크기 기준
     상대 비율로 박스를 결정한다. 슬라이더 값(비율)과 픽셀 환산
     로직을 분리해 테스트 가능성을 확보한다.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 슬라이더 범위 상수 — 슬라이더가 표현하는 비율 (%)
# ---------------------------------------------------------------------------
# 최소 크롭 비율: 프레임 최소변의 10%
MIN_CROP_RATIO: int = 10
# 최대 크롭 비율: 프레임 최소변의 90%
MAX_CROP_RATIO: int = 90
# 기본 크롭 비율: 프레임 최소변의 50%
DEFAULT_CROP_RATIO: int = 50

# 0-픽셀 방지를 위한 최소 변 크기 (짝수)
_MIN_PIXEL_SIZE: int = 2


def crop_ratio_to_size(
    ratio: int, frame_shape: tuple[int, int]
) -> tuple[int, int]:
    """슬라이더 비율(정수 %)을 짝수 픽셀 크기 (W, H) 로 환산한다.

    비율은 프레임 최소변(min(W, H))을 기준으로 한다.
    반환값은 항상 짝수이고 최소 하한(_MIN_PIXEL_SIZE) 이상이다.

    Args:
        ratio: 슬라이더 값 (MIN_CROP_RATIO ~ MAX_CROP_RATIO, 정수 %)
        frame_shape: (W, H) 프레임 크기

    Returns:
        (w, h) — 짝수, 하한 보장, 프레임 범위 이내

    WHY: 최소변 기준으로 환산하면 프레임 방향(가로/세로)에 무관하게
         동일 비율에서 일관된 상대 크기가 나온다. 정방형이면 w==h.
    """
    frame_w, frame_h = frame_shape
    # 최소변 기준 픽셀 환산
    min_side = min(frame_w, frame_h)
    raw = min_side * ratio / 100

    # 짝수 정렬 (내림)
    even = max(_MIN_PIXEL_SIZE, int(raw) - (int(raw) % 2))

    # 프레임 범위 클램프 (짝수 유지)
    w = min(even, frame_w - (frame_w % 2))
    h = min(even, frame_h - (frame_h % 2))

    # 최소 하한 보장
    w = max(_MIN_PIXEL_SIZE, w)
    h = max(_MIN_PIXEL_SIZE, h)

    return int(w), int(h)
