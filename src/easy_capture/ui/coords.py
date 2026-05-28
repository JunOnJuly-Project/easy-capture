"""위젯↔이미지 좌표 변환 순수 함수 모음.

PySide6 비의존(튜플/숫자만). 레터박스·스케일 보정 포함.
단위 테스트에서 PySide6 없이 검증 가능하도록 순수 함수로 분리한다.

WHY: FrameCanvas가 실제 스케일과 레터박스 오프셋을 계산한 뒤
     이 모듈에 위임하면, 좌표 변환 버그를 UI 없이 발견할 수 있다.
"""
from __future__ import annotations

from typing import NamedTuple


class ScaleInfo(NamedTuple):
    """이미지를 위젯에 표시할 때 적용된 스케일·오프셋 정보.

    scale: 이미지→위젯 배율 (동일 비율 유지)
    offset_x: 레터박스 수평 여백 (픽셀)
    offset_y: 레터박스 수직 여백 (픽셀)
    """

    scale: float
    offset_x: float
    offset_y: float


def compute_scale_info(
    widget_size: tuple[int, int],
    image_size: tuple[int, int],
) -> ScaleInfo:
    """위젯 크기와 이미지 크기로 레터박스 스케일 정보를 계산한다.

    aspect-fit(letterbox) 방식: 이미지 비율 유지, 빈 영역은 여백.
    """
    ww, wh = widget_size
    iw, ih = image_size

    if iw == 0 or ih == 0:
        return ScaleInfo(scale=1.0, offset_x=0.0, offset_y=0.0)

    scale = min(ww / iw, wh / ih)
    scaled_w = iw * scale
    scaled_h = ih * scale
    offset_x = (ww - scaled_w) / 2.0
    offset_y = (wh - scaled_h) / 2.0
    return ScaleInfo(scale=scale, offset_x=offset_x, offset_y=offset_y)


def widget_to_image(
    widget_pos: tuple[float, float],
    widget_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int] | None:
    """위젯 좌표를 이미지 픽셀 좌표로 역변환한다.

    레터박스 여백 외부 클릭이면 None을 반환한다.
    반환: 클램프된 이미지 좌표 (x, y) 정수 튜플, 또는 None.
    """
    info = compute_scale_info(widget_size, image_size)
    wx, wy = widget_pos

    if info.scale == 0:
        return None

    ix = (wx - info.offset_x) / info.scale
    iy = (wy - info.offset_y) / info.scale

    iw, ih = image_size
    if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
        return None

    return int(ix), int(iy)


def image_to_widget(
    image_pos: tuple[float, float],
    widget_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """이미지 픽셀 좌표를 위젯 좌표로 변환한다.

    오버레이 마스크 렌더링 시 이미지 좌표를 위젯에 매핑할 때 사용한다.
    """
    info = compute_scale_info(widget_size, image_size)
    ix, iy = image_pos
    wx = ix * info.scale + info.offset_x
    wy = iy * info.scale + info.offset_y
    return wx, wy
