"""테스트 더블(가짜 의존성) 구현.

FakeBackend         — SegmentationBackend Protocol 준수. torch/transformers 비의존.
FakeFrameSource     — FrameSource Protocol 준수. PyAV/ffprobe 비의존.
FakeUpscaleBackend  — UpscaleBackend Protocol 준수. torch/transformers 비의존.

모든 클래스는 결정적(deterministic) 출력을 보장해 단위 테스트에서
예측 가능한 결과를 계산할 수 있게 한다.

변경 이력:
  - segment_call_count: segment_image 호출 횟수 카운터 추가 (crop-ux 슬라이스).
    WHY: compute_box 반복 호출 시 세그가 재실행되지 않는다는 핵심 회귀 가드를
         FakeBackend 단독으로 검증할 수 있게 한다.
  - FakeUpscaleBackend: UpscaleBackend Protocol 준수, upscale_call_count 스파이 추가
    (image-upscale 슬라이스). WHY: export가 업스케일을 정확히 필요한 만큼만
    호출하는지 단언하기 위해. torch/transformers 비의존, nearest 확대로 결정적 출력.
"""
from __future__ import annotations

import numpy as np

# --- SegmentationBackend는 이미 존재 → import 가능 ---
from easy_capture.core.segmentation.backend import SegmentationBackend  # noqa: F401

from easy_capture.infra.video_io import FrameMeta, FrameSource  # noqa: F401

# --- UpscaleBackend: 구현 전이므로 import 실패가 예상 RED 상태 ---
# WHY: try/except로 감싸는 이유 — core/upscale 패키지가 아직 없을 때
#      fakes.py import 자체가 실패해 기존 137개 테스트까지 차단되는 것을 방지한다.
#      UpscaleBackend가 없으면 FakeUpscaleBackend의 isinstance 검사 테스트만 실패하고
#      나머지 기존 테스트는 정상 통과한다.
try:
    from easy_capture.core.upscale.backend import UpscaleBackend  # noqa: F401
except ModuleNotFoundError:
    UpscaleBackend = None  # type: ignore[assignment,misc]

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


# ---------------------------------------------------------------------------
# FakeUpscaleBackend
# ---------------------------------------------------------------------------
# nearest 확대 방식 상수
# WHY: np.repeat(np.repeat(..., scale, axis=0), scale, axis=1) 방식으로
#      torch 없이 scale배 nearest 확대를 구현한다. 결과가 결정적이어서
#      저장 이미지의 크기를 정확히 예측해 단언할 수 있다.
_UPSCALE_CHANNELS = 3


class FakeUpscaleBackend:
    """UpscaleBackend Protocol을 준수하는 테스트 더블.

    torch·transformers·PySide6를 전혀 import하지 않는다.
    upscale은 numpy np.repeat 기반 nearest 확대로 결정적 출력을 반환한다.
    결과 shape: (H*scale, W*scale, 3), dtype uint8.

    속성:
        upscale_call_count: upscale 호출 횟수 스파이 카운터.
            WHY: export가 업스케일을 정확히 필요한 만큼만(선택 시 1회,
                 미선택 시 0회) 호출한다는 핵심 회귀 가드를 단언하기 위해.
                 FakeBackend.segment_call_count 패턴을 그대로 적용한다.
    """

    def __init__(self, device: str = "cpu", scale: int = 2) -> None:
        """device·scale 속성 보관 (UpscaleBackend Protocol 요구사항).

        scale: 고정 배율(2 또는 4). 테스트에서 원하는 배율을 주입한다.
        """
        self.device = device
        self.scale = scale
        # 호출 횟수 카운터 — 초기값 0, upscale 호출마다 +1
        self.upscale_call_count: int = 0

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB HxWx3 uint8 → (H*scale, W*scale, 3) uint8 nearest 확대.

        Given: RGB HxWx3 uint8 배열
        When:  upscale 호출
        Then:  (H*scale, W*scale, 3) uint8, 호출 횟수 카운터 +1

        WHY: np.repeat으로 nearest 확대해 torch 비의존으로 결정적 출력을 보장한다.
             저장 이미지 크기를 테스트에서 scale배로 정확히 예측할 수 있게 한다.
        """
        # WHY: 카운터를 먼저 증가시켜야 조기 반환이 있어도 카운트된다.
        self.upscale_call_count += 1
        # H축 repeat → W축 repeat 순서로 nearest 확대
        enlarged = np.repeat(image_rgb, self.scale, axis=0)
        enlarged = np.repeat(enlarged, self.scale, axis=1)
        return enlarged.astype(np.uint8)
