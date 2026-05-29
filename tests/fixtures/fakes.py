"""테스트 더블(가짜 의존성) 구현.

FakeBackend            — SegmentationBackend Protocol 준수. torch/transformers 비의존.
FakeFrameSource        — FrameSource Protocol 준수. PyAV/ffprobe 비의존.
FakeUpscaleBackend     — UpscaleBackend Protocol 준수. torch/transformers 비의존.
FakeVideoBackend       — VideoSegmentationBackend Protocol 준수. torch/transformers 비의존.
FakeDetectionBackend   — DetectionBackend Protocol 준수. torch/transformers 비의존.

모든 클래스는 결정적(deterministic) 출력을 보장해 단위 테스트에서
예측 가능한 결과를 계산할 수 있게 한다.

변경 이력:
  - segment_call_count: segment_image 호출 횟수 카운터 추가 (crop-ux 슬라이스).
    WHY: compute_box 반복 호출 시 세그가 재실행되지 않는다는 핵심 회귀 가드를
         FakeBackend 단독으로 검증할 수 있게 한다.
  - FakeUpscaleBackend: UpscaleBackend Protocol 준수, upscale_call_count 스파이 추가
    (image-upscale 슬라이스). WHY: export가 업스케일을 정확히 필요한 만큼만
    호출하는지 단언하기 위해. torch/transformers 비의존, nearest 확대로 결정적 출력.
  - FakeVideoBackend: VideoSegmentationBackend Protocol 준수, propagate_call_count 등
    3종 스파이 카운터 추가 (video-tracking 슬라이스).
    WHY: track 1회 후 compute_boxes 반복 호출 시 propagate가 재호출되지 않는다는
         "재추적 안 함" 핵심 회귀 가드를 단독으로 검증하기 위해.
         drift(드리프트)·empty_after(occlusion) 옵션으로 프레임별 마스크를 결정적 생성.
  - FakeDetectionBackend: DetectionBackend Protocol 준수, detect_call_count 스파이 추가
    (video-shot-retrack 슬라이스).
    WHY: track이 컷 경계마다 정확히 1회만 detect를 호출하는지 단언하기 위해.
         시나리오별 후보 리스트(통과/미달/다중/빈)를 주입해 결정적 검출을 흉내 낸다.
         torch/transformers 전혀 로드하지 않는다.
  - FakeVideoBackend.add_box: VideoSegmentationBackend Protocol의 box 프롬프트 확장
    (mask-refine 슬라이스 — box 프롬프트 경로).
    WHY: PoC는 detect 전신 bbox를 SAM2 box 프롬프트로 직접 넘겼는데 production이
         박스 중심점(point)만 넘기는 방식으로 회귀해 마스크가 부정확·과대해졌다.
         add_box(session, box) 계약을 더블로 흉내 내, app이 자동 재매칭 통과 샷에서
         중심점 변환(_box_center) 대신 detect best box를 add_box로 전달하는지
         add_box_call_count 스파이로 단언한다. box 경로는 box 영역 사각형을
         그대로 마스크로 반환해 결정적(centroid·bbox 예측 가능)이다.
  - FakeVideoBackend.add_box/add_click negatives 인자 + negatives 스파이
    (negative point 슬라이스 — negative point 경로).
    WHY: 군무 밀착 구간에서 box+positive만으론 대상+옆사람이 한 덩어리로 합쳐진다.
         negative point("이 점=옆 멤버는 대상 아님", SAM2 label 0)로 경계를 가른다.
         transformers 검증상 box+positive+negative는 SAM2 1회 호출로 조립
         (input_boxes+input_points+input_labels, clear_old_inputs=True)되므로
         Protocol은 별도 메서드가 아니라 add_box/add_click에 negatives 인자를 추가한다.
         add_box(session, box, negatives=())·add_click(session, point, negatives=())로
         확장하고, app이 그 샷의 negative 좌표를 backend로 정확히 전달하는지를
         add_negative_call_count·last_negatives 스파이로 단언한다. negatives는
         카운터/기록만 — propagate 마스크 생성엔 영향을 주지 않아도 된다(가드 전용).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# --- SegmentationBackend는 이미 존재 → import 가능 ---
from easy_capture.core.segmentation.backend import SegmentationBackend  # noqa: F401

from easy_capture.core.source.frame_source import FrameMeta, FrameSource  # noqa: F401

# --- UpscaleBackend: 구현 전이므로 import 실패가 예상 RED 상태 ---
# WHY: try/except로 감싸는 이유 — core/upscale 패키지가 아직 없을 때
#      fakes.py import 자체가 실패해 기존 137개 테스트까지 차단되는 것을 방지한다.
#      UpscaleBackend가 없으면 FakeUpscaleBackend의 isinstance 검사 테스트만 실패하고
#      나머지 기존 테스트는 정상 통과한다.
try:
    from easy_capture.core.upscale.backend import UpscaleBackend  # noqa: F401
except ModuleNotFoundError:
    UpscaleBackend = None  # type: ignore[assignment,misc]

# --- VideoSegmentationBackend: 비디오 슬라이스 구현 전이므로 try/except 격리 ---
# WHY: core/segmentation/video_backend.py가 아직 없을 때 fakes.py import 자체가
#      실패해 기존 테스트를 차단하지 않도록 동일한 패턴을 적용한다.
#      VideoSegmentationBackend가 없으면 FakeVideoBackend 관련 테스트만 실패하고
#      나머지 기존 166개 테스트는 정상 통과한다.
try:
    from easy_capture.core.segmentation.video_backend import (  # noqa: F401
        VideoSegmentationBackend,
    )
    _HAS_VIDEO_BACKEND = True
except ModuleNotFoundError:
    VideoSegmentationBackend = None  # type: ignore[assignment,misc]
    _HAS_VIDEO_BACKEND = False

# --- DetectionBackend: 샷경계 재추적 슬라이스 구현 전이므로 try/except 격리 ---
# WHY: core/segmentation/detection_backend.py가 아직 없을 때 fakes.py import 자체가
#      실패해 기존 테스트를 차단하지 않도록 동일한 패턴을 적용한다.
#      DetectionBackend·Detection이 없으면 FakeDetectionBackend 관련 테스트만 실패하고
#      나머지 기존 216개 테스트는 정상 통과한다.
try:
    from easy_capture.core.segmentation.detection_backend import (  # noqa: F401
        Detection,
        DetectionBackend,
    )
    _HAS_DETECTION_BACKEND = True
except ModuleNotFoundError:
    Detection = None  # type: ignore[assignment,misc]
    DetectionBackend = None  # type: ignore[assignment,misc]
    _HAS_DETECTION_BACKEND = False

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

    read_frames: FrameSpan 구간만큼 동일 프레임을 복사해 리스트로 반환한다.
        WHY: 비디오 슬라이스 테스트에서 FakeVideoBackend에 넘길 프레임 시퀀스를
             PyAV 없이 결정적으로 생성한다. 계획서 FrameSource Protocol 확장 준수.
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

    def read_frames(self, span: object) -> list[np.ndarray]:
        """FrameSpan 구간만큼 고정 프레임 리스트를 반환한다.

        Given: FrameSpan(start, end, step) 또는 start/end/step 속성을 가진 객체
        When:  read_frames(span) 호출
        Then:  range(start, end, step) 길이만큼 동일 프레임 복사 리스트

        WHY: 비디오 슬라이스 테스트에서 PyAV 없이 결정적 프레임 시퀀스를 공급한다.
             이미지 소스의 "단일 프레임 1개 리스트 위임" 정책(LSP 안전)을 구현한다.
             FrameSpan이 아직 미구현이면 hasattr 폴백으로 범용 처리한다.
        """
        # FrameSpan dataclass의 start/end/step 속성 사용 (구조적 덕 타이핑)
        start = getattr(span, "start", 0)
        end = getattr(span, "end", 1)
        step = getattr(span, "step", 1)
        indices = range(start, end, step)
        return [self.read_frame(i) for i in indices]


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


# ---------------------------------------------------------------------------
# FakeVideoBackend
# ---------------------------------------------------------------------------
# 비디오 더블 내부 상수
_VIDEO_MASK_HALF_SIZE = 20   # 사각형 마스크 반경 (픽셀 단위, 양방향)
_VIDEO_DEFAULT_H = 360       # 세션 기본 프레임 높이 (frames 미제공 시 폴백)
_VIDEO_DEFAULT_W = 640       # 세션 기본 프레임 너비


class _FakeSession:
    """FakeVideoBackend 내부 세션 객체(불투명 opaque 계약 준수).

    WHY: VideoSegmentationBackend Protocol은 session을 opaque object로 사용한다.
         실제 SAM2 세션 타입을 노출하지 않는 계약을 테스트 더블에서도 그대로 유지한다.
    """

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.click_point: tuple[int, int] | None = None
        # box 프롬프트 경로 — add_box로 등록된 (x1, y1, x2, y2). 미등록 시 None.
        # WHY: box 경로(box 프롬프트)와 point 경로(click)를 propagate에서 구분해
        #      box가 있으면 box 영역 사각형을, 없으면 클릭점 중심 사각형을 반환한다.
        self.box: tuple[float, float, float, float] | None = None


class FakeVideoBackend:
    """VideoSegmentationBackend Protocol을 준수하는 테스트 더블.

    torch·transformers를 전혀 import하지 않는다.
    propagate는 클릭 포인트 기반 결정적 마스크 리스트를 반환한다.

    드리프트(drift): 프레임 i마다 클릭점 중심이 (dx, dy) 씩 이동한 사각형 마스크.
        WHY: SAM2 video가 객체를 추적하며 centroid가 이동하는 현상을 흉내 낸다.
             smooth_centroids 적용 효과를 결정적으로 검증할 수 있다.

    empty_after: 이 인덱스(포함) 이후 프레임에 빈 마스크를 반환(occlusion 시뮬레이션).
        WHY: valid_flags=False → gap_policy BACKGROUND 경로를 결정적으로 검증한다.

    스파이 카운터:
        init_call_count     — init_session 호출 횟수
        add_click_call_count — add_click 호출 횟수
        add_box_call_count   — add_box 호출 횟수 (box 프롬프트 경로)
        add_negative_call_count — add_box/add_click이 비어있지 않은 negatives를
            받아 등록한 횟수 (negative point 경로)
        last_negatives      — 마지막 add_box/add_click 호출이 받은 negatives 튜플
            (negative 미지정 호출은 () 로 기록 → 무회귀 단언용)
        propagate_call_count — propagate 호출 횟수
        WHY: track 1회 후 compute_boxes를 여러 번 호출해도
             propagate_call_count == 1 임을 단언해 재추적 없음을 강제한다.
             이미지 모드 segment_call_count 가드의 비디오판.
             add_box_call_count: app이 자동 재매칭 통과 샷에서 중심점 변환 대신
             detect best box를 add_box로 넘기는지 단언하는 box 프롬프트 가드.
             add_negative_call_count·last_negatives: app이 그 샷의 negative
             좌표(옆 멤버, label 0)를 add_box/add_click의 negatives 인자로
             정확히 전달하는지 단언하는 negative point 가드. negatives 미지정
             경로는 last_negatives==()로 남아 기존 box/point 경로 무회귀를 보장한다.
    """

    device: str = "cpu"

    def __init__(
        self,
        drift: tuple[int, int] = (2, 0),
        empty_after: int | None = None,
    ) -> None:
        """drift·empty_after 설정 보관, 스파이 카운터 초기화.

        drift: (dx, dy) — 프레임 i마다 클릭 중심에 i*dx, i*dy 오프셋 적용.
        empty_after: 이 프레임 인덱스(포함) 이후 빈 마스크 반환. None이면 사용 안 함.
        """
        self._drift = drift
        self._empty_after = empty_after
        # 스파이 카운터 — 초기값 0, 각 메서드 호출마다 +1
        self.init_call_count: int = 0
        self.add_click_call_count: int = 0
        self.add_box_call_count: int = 0
        self.propagate_call_count: int = 0
        # negative point 스파이 — 비어있지 않은 negatives 등록 횟수와 마지막 값
        # WHY: add_box/add_click이 받은 negative 좌표를 기록해, app이 그 샷의
        #      negative를 정확히 전달하는지 단언한다. 초기 last_negatives=()는
        #      "아직 negative 미수신" 상태(무회귀 기본값)를 나타낸다.
        self.add_negative_call_count: int = 0
        self.last_negatives: tuple = ()

    def init_session(self, frames: list[np.ndarray]) -> _FakeSession:
        """구간 프레임 시퀀스로 세션을 초기화하고 반환한다.

        Given: RGB HxWx3 uint8 프레임 리스트
        When:  init_session 호출
        Then:  _FakeSession 반환, init_call_count +1
        """
        # WHY: 카운터를 먼저 증가시켜 조기 반환에서도 카운트된다.
        self.init_call_count += 1
        return _FakeSession(frames=frames)

    def add_click(
        self,
        session: _FakeSession,
        point: tuple[int, int],
        negatives: "Sequence[tuple[int, int]]" = (),
    ) -> None:
        """첫 프레임에 전경 클릭 포인트(+선택적 negative 좌표)를 등록한다.

        Given: _FakeSession 인스턴스, (x, y) 클릭 좌표, 선택적 negatives
        When:  add_click 호출
        Then:  session.click_point에 저장, add_click_call_count +1.
               negatives는 항상 last_negatives에 기록(빈 튜플 포함),
               비어있지 않으면 add_negative_call_count +1.

        WHY: transformers 검증상 box+positive+negative는 SAM2 1회 호출로
             조립되므로 별도 메서드가 아니라 add_click에 negatives 인자를 받는다.
             negatives는 카운터/기록만 — propagate 마스크 생성엔 영향 없어도 된다.
             label 0(negative)으로 옆 멤버를 '대상 아님'으로 표시하는 입력을 모사한다.
        """
        self.add_click_call_count += 1
        session.click_point = point
        self._record_negatives(negatives)

    def add_box(
        self,
        session: _FakeSession,
        box: tuple[float, float, float, float],
        negatives: "Sequence[tuple[int, int]]" = (),
    ) -> None:
        """첫 프레임에 전경 box 프롬프트(+선택적 negative 좌표)를 등록한다.

        Given: _FakeSession 인스턴스, (x1, y1, x2, y2) box 좌표, 선택적 negatives
        When:  add_box 호출
        Then:  session.box에 저장, add_box_call_count +1.
               negatives는 항상 last_negatives에 기록(빈 튜플 포함),
               비어있지 않으면 add_negative_call_count +1.

        WHY: PoC가 쓰던 box 프롬프트(detect 전신 bbox→SAM2)를 더블로 흉내 낸다.
             propagate에서 box가 등록돼 있으면 box 영역 사각형을 마스크로 반환해
             '중심점 1개(point)'보다 정확한 전신 마스크를 결정적으로 모사한다.
             negatives는 box+positive와 함께 SAM2 1회 호출로 조립되는 label 0
             입력을 모사한다(군무 밀착 구간 옆 멤버 경계 분리).
        """
        self.add_box_call_count += 1
        session.box = box
        self._record_negatives(negatives)

    def _record_negatives(
        self,
        negatives: "Sequence[tuple[int, int]]",
    ) -> None:
        """negatives를 last_negatives에 기록하고, 비어있지 않으면 카운터를 올린다.

        WHY: add_box/add_click 양쪽이 동일한 스파이 기록 로직을 공유하도록 추출(DRY).
             빈 입력도 last_negatives에 ()로 남겨 "negative 미전달"을 명시 단언할 수 있다.
        """
        recorded = tuple(negatives)
        self.last_negatives = recorded
        if recorded:
            self.add_negative_call_count += 1

    def propagate(self, session: _FakeSession) -> list[np.ndarray]:
        """세션을 끝까지 전파해 프레임별 bool HxW 마스크 리스트를 반환한다.

        Given: add_click(point) 또는 add_box(box)로 프롬프트가 등록된 _FakeSession
        When:  propagate 호출
        Then:  프레임 수만큼의 bool HxW 마스크 리스트 반환, propagate_call_count +1.
               box 등록 시: box 영역 사각형 마스크(중심에 drift 적용).
               box 미등록 시: 클릭점 중심 사각형(drift 적용).
               empty_after 적용: 해당 인덱스 이후 빈 마스크(occlusion 경로).

        WHY: 결정적 마스크로 centroid 시퀀스가 예측 가능해
             smooth_centroids 적용 결과를 수동 계산과 비교할 수 있다.
             box 경로는 box 영역 자체를 마스크로 반환해 point 경로(중심점 1개
             주변 고정 반경 사각형)와 결과 bbox가 구분되도록 한다.
        """
        self.propagate_call_count += 1
        if not session.frames:
            return []

        h, w = session.frames[0].shape[:2]
        base_x, base_y, half_or_box = self._resolve_prompt(session, w, h)

        dx, dy = self._drift
        masks: list[np.ndarray] = []
        for i in range(len(session.frames)):
            # empty_after 이후 인덱스 → occlusion(빈 마스크)
            is_occluded = (
                self._empty_after is not None and i >= self._empty_after
            )
            if is_occluded:
                masks.append(np.zeros((h, w), dtype=bool))
            else:
                cx = int(base_x + i * dx)
                cy = int(base_y + i * dy)
                masks.append(self._build_mask(h, w, cx, cy, half_or_box))
        return masks

    @staticmethod
    def _resolve_prompt(
        session: _FakeSession,
        w: int,
        h: int,
    ) -> tuple[int, int, tuple[int, int] | int]:
        """등록된 프롬프트(box 우선, 없으면 click)에서 중심·반경 정보를 도출한다.

        Returns:
            (base_x, base_y, half_or_box) —
            box 경로면 half_or_box=(half_w, half_h)(box 절반 크기),
            point 경로면 half_or_box=_VIDEO_MASK_HALF_SIZE(고정 반경 정수).

        WHY: propagate를 20줄 이내로 유지하면서 box/point 경로의 중심·크기를
             한 곳에서 결정해 _build_mask가 분기 없이 사각형을 그리게 한다.
        """
        if session.box is not None:
            x1, y1, x2, y2 = session.box
            base_x = int((x1 + x2) / 2)
            base_y = int((y1 + y2) / 2)
            half_w = max(1, int((x2 - x1) / 2))
            half_h = max(1, int((y2 - y1) / 2))
            return base_x, base_y, (half_w, half_h)
        base_x = session.click_point[0] if session.click_point else w // 2
        base_y = session.click_point[1] if session.click_point else h // 2
        return base_x, base_y, _VIDEO_MASK_HALF_SIZE

    @staticmethod
    def _build_mask(
        h: int,
        w: int,
        cx: int,
        cy: int,
        half_or_box: tuple[int, int] | int,
    ) -> np.ndarray:
        """중심·반경 정보로 사각형 bool 마스크를 만든다(box/point 공통)."""
        if isinstance(half_or_box, tuple):
            half_w, half_h = half_or_box
            mask = np.zeros((h, w), dtype=bool)
            y1 = max(0, cy - half_h)
            y2 = min(h, cy + half_h)
            x1 = max(0, cx - half_w)
            x2 = min(w, cx + half_w)
            mask[y1:y2, x1:x2] = True
            return mask
        return _make_rect_mask(h, w, cx, cy, half_or_box)


# ---------------------------------------------------------------------------
# FakeDetectionBackend
# ---------------------------------------------------------------------------

class FakeDetectionBackend:
    """DetectionBackend Protocol을 준수하는 테스트 더블.

    torch·transformers·Grounding DINO를 전혀 import하지 않는다.
    생성자에 시나리오별 후보 리스트를 주입해 결정적 검출을 흉내 낸다.

    시나리오 주입 방식:
      - candidates_sequence: 호출 순서대로 반환할 후보 리스트의 리스트.
        예: [[Detection(...)], [], [Detection(...), Detection(...)]]
        주입하지 않으면 candidates_fixed로 고정 반환.
      - candidates_fixed: 매 호출마다 동일한 후보 리스트 반환(기본 []).

    속성:
        detect_call_count: detect 호출 횟수 스파이 카운터.
            WHY: track이 컷 경계마다 정확히 1회만 detect를 호출하는지 단언하고,
                 compute_boxes 반복 호출 시 detect가 재호출되지 않음을 강제한다.
                 FakeVideoBackend.propagate_call_count의 검출 전용판.
    """

    device: str = "cpu"

    def __init__(
        self,
        candidates_fixed: "list | None" = None,
        candidates_sequence: "list[list] | None" = None,
    ) -> None:
        """시나리오 후보 주입.

        candidates_fixed: 매 호출마다 동일하게 반환할 Detection 리스트.
            None이면 빈 리스트([])를 항상 반환.
        candidates_sequence: 호출 순서별 후보 리스트의 시퀀스.
            주어지면 candidates_fixed를 무시하고 호출 순서대로 꺼낸다.
            시퀀스 소진 후에는 candidates_fixed(또는 [])를 반환한다.

        WHY: 통과/미달/다중후보/빈검출 4종 시나리오를 생성자에서 주입해
             테스트 코드에서 직접 Grounding DINO 동작을 모사한다.
        """
        self._candidates_fixed: list = candidates_fixed if candidates_fixed is not None else []
        # 시퀀스 복사 — 소비 중 원본 변경 방지
        self._candidates_sequence: list[list] = (
            list(candidates_sequence) if candidates_sequence is not None else []
        )
        # 스파이 카운터 — 초기값 0, detect 호출마다 +1
        self.detect_call_count: int = 0

    def detect(self, frame: np.ndarray, prompt: str = "person.") -> list:
        """프레임에서 후보 Detection 리스트를 반환한다(결정적). 호출마다 카운터 +1.

        Given: RGB HxWx3 프레임, 텍스트 프롬프트(무시됨 — 결정적 반환)
        When:  detect 호출
        Then:  주입된 시나리오에 따른 Detection 리스트 반환, detect_call_count +1

        WHY: 카운터를 먼저 증가시켜야 조기 반환에서도 카운트된다.
             frame·prompt는 무시 — 테스트에서 원하는 후보를 생성자에서 주입한다.
        """
        # WHY: 카운터를 먼저 증가
        self.detect_call_count += 1

        if self._candidates_sequence:
            # 시퀀스에서 순서대로 꺼낸다
            return self._candidates_sequence.pop(0)
        return list(self._candidates_fixed)
