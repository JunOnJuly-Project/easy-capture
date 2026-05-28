"""테스트 더블(가짜 의존성) 구현.

FakeBackend   — SegmentationBackend Protocol 준수. torch/transformers 비의존.
FakeFrameSource — FrameSource Protocol 준수. PyAV/ffprobe 비의존.

두 클래스 모두 결정적(deterministic) 출력을 보장해 단위 테스트에서
예측 가능한 centroid·마스크를 계산할 수 있게 한다.

변경 이력:
  - segment_call_count: segment_image 호출 횟수 카운터 추가 (crop-ux 슬라이스).
    WHY: compute_box 반복 호출 시 세그가 재실행되지 않는다는 핵심 회귀 가드를
         FakeBackend 단독으로 검증할 수 있게 한다.
"""
from __future__ import annotations

import numpy as np

# --- SegmentationBackend는 이미 존재 → import 가능 ---
from easy_capture.core.segmentation.backend import SegmentationBackend  # noqa: F401

# --- video_io는 아직 미구현 → ImportError는 TDD Red 단계에서 정상 ---
from easy_capture.infra.video_io import FrameMeta, FrameSource  # noqa: F401

# ---------------------------------------------------------------------------
# 결정적 마스크 생성 헬퍼 상수
# ---------------------------------------------------------------------------
# 클릭 포인트 주변 더미 마스크 크기 (픽셀 단위, 양방향)
_MASK_HALF_SIZE = 20


def _make_rect_mask(
    height: int,
    width: int,
    cx: int,
    cy: int,
    half: int = _MASK_HALF_SIZE,
) -> np.ndarray:
    """cx, cy 중심의 사각형 bool 마스크를 반환한다.

    프레임 경계를 넘지 않도록 클램프한다.
    centroid_of_mask(result) == (cx_실제, cy_실제) 가 예측 가능하도록
    정수 픽셀 정렬된 사각형을 사용한다.
    """
    # WHY: 홀수 크기 사각형을 쓰면 평균이 정확히 정수 좌표 중심이 된다.
    #      2*half 크기(짝수)를 쓰므로 실제 중심은 cx ± 0.5 오차가 발생할 수 있다.
    #      테스트는 "근사값이 요청 좌표에 가까운지"만 검증한다.
    mask = np.zeros((height, width), dtype=bool)
    y1 = max(0, cy - half)
    y2 = min(height, cy + half)
    x1 = max(0, cx - half)
    x2 = min(width, cx + half)
    mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------
# FakeBackend
# ---------------------------------------------------------------------------
class FakeBackend:
    """SegmentationBackend Protocol을 준수하는 테스트 더블.

    torch·transformers를 전혀 import하지 않는다.
    segment_image는 입력 프레임 크기에 맞춘 결정적 더미 마스크를 반환한다.
    - 클릭 포인트(points)가 주어지면 그 주변 사각형
    - 포인트가 없으면 프레임 중앙 사각형
    - empty_mask=True이면 픽셀이 전부 False인 빈 마스크를 반환(빈 마스크 경로 테스트용)

    속성:
        segment_call_count: segment_image 호출 횟수 스파이 카운터.
            WHY: compute_box 반복 호출 시 segment_image가 재호출되지 않는다는
                 "재세그 안 함" 핵심 회귀 가드를 테스트에서 직접 단언하기 위해.
                 기존 결정적 마스크 동작은 이 카운터 추가로 변경되지 않는다.
    """

    def __init__(self, device: str = "cpu", empty_mask: bool = False) -> None:
        """device 속성 보관 (Protocol 요구사항).

        empty_mask: True이면 항상 빈 마스크 반환 — EmptyMaskError 경로 테스트용.
        """
        self.device = device
        self._empty_mask = empty_mask
        # 호출 횟수 카운터 — 초기값 0, segment_image 호출마다 +1
        self.segment_call_count: int = 0

    def segment_image(
        self,
        frame: np.ndarray,
        points=None,
        boxes=None,
    ) -> np.ndarray:
        """결정적 bool HxW 마스크를 반환한다. 호출마다 segment_call_count를 증가한다.

        Given: RGB HxWx3 프레임 + 선택적 클릭 포인트
        When:  포인트가 있으면 첫 번째 포인트 주변, 없으면 프레임 중앙
        When:  empty_mask=True이면 전부 False인 빈 마스크
        Then:  bool dtype, 프레임과 동일한 (H, W) shape
        """
        # WHY: 카운터를 먼저 증가시켜야 empty_mask 조기 반환에서도 카운트된다.
        self.segment_call_count += 1
        h, w = frame.shape[:2]

        if self._empty_mask:
            # WHY: 빈 마스크 → EmptyMaskError 경로를 테스트하기 위한 모드
            return np.zeros((h, w), dtype=bool)

        if points is not None and len(points) > 0:
            # 첫 번째 클릭 포인트 사용 (x, y 순서)
            cx, cy = int(points[0][0]), int(points[0][1])
        else:
            # 클릭 없으면 중앙
            cx, cy = w // 2, h // 2

        return _make_rect_mask(h, w, cx, cy)

    def supports_video(self) -> bool:
        """이미지 전용 — 비디오 추적 미지원."""
        return False


# ---------------------------------------------------------------------------
# FakeFrameSource
# ---------------------------------------------------------------------------
# 고정 프레임 크기 상수
_FAKE_FRAME_HEIGHT = 360
_FAKE_FRAME_WIDTH = 640
_FAKE_FPS = None  # 이미지 소스이므로 fps 없음


class FakeFrameSource:
    """FrameSource Protocol을 준수하는 테스트 더블.

    PyAV·ffprobe를 전혀 사용하지 않는다.
    read_frame은 항상 동일한 고정 RGB 배열을 반환한다.
    픽셀값은 좌표 기반 결정적 그라디언트(테스트 pixel 검증용).
    """

    def probe(self) -> FrameMeta:
        """고정 메타 반환.

        Given: 테스트 더블 인스턴스
        When:  probe() 호출
        Then:  width=640, height=360, is_video=False, fps=None
        """
        return FrameMeta(
            width=_FAKE_FRAME_WIDTH,
            height=_FAKE_FRAME_HEIGHT,
            is_video=False,
            fps=_FAKE_FPS,
        )

    def read_frame(self, index: int = 0) -> np.ndarray:
        """RGB HxWx3 uint8 고정 배열 반환.

        Given: 호출 index (무시됨 — 항상 동일 프레임)
        When:  read_frame() 호출
        Then:  shape (360, 640, 3), dtype uint8, 결정적 픽셀값
        """
        # WHY: np.indices로 좌표 기반 그라디언트를 만들면
        #      crop_array 테스트에서 특정 픽셀값을 예측해 검증할 수 있다.
        h, w = _FAKE_FRAME_HEIGHT, _FAKE_FRAME_WIDTH
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # R채널: x 좌표 기반 (0~255)
        frame[:, :, 0] = (np.arange(w) * 255 // (w - 1)).astype(np.uint8)
        # G채널: y 좌표 기반 (0~255)
        frame[:, :, 1] = (np.arange(h) * 255 // (h - 1)).astype(np.uint8)[:, np.newaxis]
        # B채널: 고정값 128
        frame[:, :, 2] = 128
        return frame
