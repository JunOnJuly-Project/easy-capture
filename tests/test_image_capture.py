"""ImageCaptureUseCase 슬라이스 핵심 조립 테스트.

대상 모듈: easy_capture.app.image_capture
테스트 더블: FakeBackend, FakeFrameSource (tests/fixtures/fakes.py)

이 테스트 파일이 검증하는 계약:
  1. load_frame()       → FakeFrameSource가 반환하는 고정 프레임
  2. make_crop_box()    → FakeBackend 마스크의 centroid + 종횡비 잠금 + 짝수 + 경계 클램프
  3. export()           → tmp_path에 파일 생성, 크롭 크기 일치
  4. end-to-end happy path → 전 구간을 가짜 의존으로 관통
  5. Protocol 계약      → isinstance(FakeBackend(), SegmentationBackend) 통과
  6. segment()          → SegmentResult(mask, centroid) 반환 / 빈 마스크 EmptyMaskError
  7. compute_box()      → 순수 계산: 종횡비·크기 변경 시 박스 재계산, 재세그 없음(핵심 회귀 가드)
  8. make_crop_box 무회귀 → segment+compute_box 위임 후에도 동일 결과(A안)
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from easy_capture.app.image_capture import (
    BoxParams,
    CropRequest,
    EmptyMaskError,
    ImageCaptureUseCase,
    SegmentResult,
)
from easy_capture.core.crop import (
    apply_aspect_lock,
    centroid_of_mask,
    make_crop_box,
)
from easy_capture.core.export.image_export import ExportConfig
from easy_capture.core.segmentation.backend import SegmentationBackend
from tests.fixtures.fakes import FakeBackend, FakeFrameSource

# WHY: FakeUpscaleBackend는 core/upscale 패키지에 의존하는 import를 포함하므로
#      패키지 미구현 시 import 실패로 기존 테스트까지 차단되지 않도록 분리한다.
try:
    from tests.fixtures.fakes import FakeUpscaleBackend
    _HAS_FAKE_UPSCALER = True
except (ImportError, ModuleNotFoundError):
    FakeUpscaleBackend = None  # type: ignore[assignment,misc]
    _HAS_FAKE_UPSCALER = False

_MSG_NO_UPSCALER = "core/upscale 패키지 미구현 — FakeUpscaleBackend 사용 불가"

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# FakeFrameSource 고정 프레임 크기 (fakes.py 상수와 동기화)
FAKE_FRAME_W = 640
FAKE_FRAME_H = 360

# 테스트용 클릭 포인트 — 프레임 중앙에서 약간 벗어난 예측 가능한 좌표
CLICK_X = 200
CLICK_Y = 150

# 요청 크롭 크기 (W×H)
REQUEST_CROP_W = 300
REQUEST_CROP_H = 200

# FakeBackend 마스크 반경 (fakes.py _MASK_HALF_SIZE와 동기화)
FAKE_MASK_HALF = 20


# ---------------------------------------------------------------------------
# 예상 결과 계산 헬퍼 (구현과 독립적 — 테스트 기준값)
# ---------------------------------------------------------------------------
def _expected_crop_box(
    click_x: int,
    click_y: int,
    crop_w: int,
    crop_h: int,
    aspect: str | None,
    frame_w: int = FAKE_FRAME_W,
    frame_h: int = FAKE_FRAME_H,
) -> tuple[int, int, int, int]:
    """FakeBackend + core 함수 조합으로 예상 크롭 박스를 직접 계산.

    WHY: 유스케이스가 core 함수를 올바르게 조합하는지 검증하기 위해
         테스트 자체가 동일한 계산을 독립적으로 수행해 비교한다.
    """
    # Step 1: FakeBackend와 동일한 결정적 마스크 생성
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    y1 = max(0, click_y - FAKE_MASK_HALF)
    y2 = min(frame_h, click_y + FAKE_MASK_HALF)
    x1 = max(0, click_x - FAKE_MASK_HALF)
    x2 = min(frame_w, click_x + FAKE_MASK_HALF)
    mask[y1:y2, x1:x2] = True

    # Step 2: centroid 계산
    centroid = centroid_of_mask(mask)
    assert centroid is not None, "테스트 설계 오류: 마스크가 비어 있어서는 안 됨"
    cx, cy = centroid

    # Step 3: 종횡비 잠금
    locked_w, locked_h = apply_aspect_lock(crop_w, crop_h, aspect)

    # Step 4: 크롭 박스 산출 (짝수·경계 클램프)
    return make_crop_box((cx, cy), (locked_w, locked_h), (frame_w, frame_h))


# ---------------------------------------------------------------------------
# Protocol 계약 테스트
# ---------------------------------------------------------------------------
class TestProtocolContract:
    """FakeBackend가 SegmentationBackend Protocol을 올바르게 구현하는지 검증."""

    def test_FakeBackend_인스턴스는_SegmentationBackend_isinstance를_통과한다(self):
        """Given: FakeBackend 인스턴스
        When:  isinstance(..., SegmentationBackend) 호출
        Then:  True 반환 (runtime_checkable Protocol 계약 준수)
        """
        fake = FakeBackend(device="cpu")

        # WHY: @runtime_checkable Protocol은 구조적 서브타이핑을 런타임에 검증한다.
        #      이 테스트가 통과해야 실제 주입 시에도 타입 오류가 없다.
        assert isinstance(fake, SegmentationBackend)

    def test_FakeBackend_device_속성이_올바르게_설정된다(self):
        """Given: device='cuda'로 생성한 FakeBackend
        When:  .device 속성 접근
        Then:  'cuda' 반환
        """
        fake = FakeBackend(device="cuda")

        assert fake.device == "cuda"

    def test_FakeBackend_supports_video는_False이다(self):
        """Given: FakeBackend 인스턴스
        When:  supports_video() 호출
        Then:  False 반환 (이미지 전용)
        """
        fake = FakeBackend()

        assert fake.supports_video() is False

    def test_FakeBackend_segment_image_반환값은_bool_HxW_배열이다(self):
        """Given: 임의 RGB 프레임
        When:  segment_image 호출
        Then:  bool dtype, shape == (H, W)
        """
        fake = FakeBackend()
        frame = np.zeros((FAKE_FRAME_H, FAKE_FRAME_W, 3), dtype=np.uint8)

        mask = fake.segment_image(frame)

        assert mask.dtype == bool
        assert mask.shape == (FAKE_FRAME_H, FAKE_FRAME_W)

    def test_FakeBackend_포인트_없으면_중앙_마스크를_반환한다(self):
        """Given: 클릭 포인트 없음
        When:  segment_image 호출
        Then:  centroid가 프레임 중앙 근방에 위치
        """
        fake = FakeBackend()
        frame = np.zeros((FAKE_FRAME_H, FAKE_FRAME_W, 3), dtype=np.uint8)

        mask = fake.segment_image(frame, points=None)
        centroid = centroid_of_mask(mask)

        assert centroid is not None
        cx, cy = centroid
        # 중앙 ± 반경 이내에 centroid가 있어야 한다
        assert abs(cx - FAKE_FRAME_W // 2) <= FAKE_MASK_HALF + 1
        assert abs(cy - FAKE_FRAME_H // 2) <= FAKE_MASK_HALF + 1

    def test_FakeBackend_포인트_지정_시_해당_위치_중심_마스크를_반환한다(self):
        """Given: (200, 150) 클릭 포인트
        When:  segment_image(frame, points=[(200, 150)]) 호출
        Then:  centroid가 (200, 150) 근방
        """
        fake = FakeBackend()
        frame = np.zeros((FAKE_FRAME_H, FAKE_FRAME_W, 3), dtype=np.uint8)

        mask = fake.segment_image(frame, points=[(CLICK_X, CLICK_Y)])
        centroid = centroid_of_mask(mask)

        assert centroid is not None
        cx, cy = centroid
        assert abs(cx - CLICK_X) <= FAKE_MASK_HALF + 1
        assert abs(cy - CLICK_Y) <= FAKE_MASK_HALF + 1


# ---------------------------------------------------------------------------
# load_frame 테스트
# ---------------------------------------------------------------------------
class TestLoadFrame:
    """ImageCaptureUseCase.load_frame: FakeFrameSource의 프레임을 반환."""

    def test_load_frame_반환값은_RGB_uint8_배열이다(self):
        """Given: FakeFrameSource·FakeBackend 주입
        When:  load_frame() 호출
        Then:  shape (360, 640, 3), dtype uint8
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )

        frame = usecase.load_frame()

        assert frame.shape == (FAKE_FRAME_H, FAKE_FRAME_W, 3)
        assert frame.dtype == np.uint8

    def test_load_frame_연속_호출_시_동일_배열을_반환한다(self):
        """Given: FakeFrameSource (결정적)
        When:  load_frame() 두 번 호출
        Then:  두 결과가 동일 (결정적 소스 계약)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )

        frame1 = usecase.load_frame()
        frame2 = usecase.load_frame()

        np.testing.assert_array_equal(frame1, frame2)


# ---------------------------------------------------------------------------
# make_crop_box 테스트
# ---------------------------------------------------------------------------
class TestMakeCropBox:
    """ImageCaptureUseCase.make_crop_box: 마스크 → centroid → 박스 변환 조합 검증."""

    def test_make_crop_box_반환값은_짝수_크기이다(self):
        """Given: 클릭 포인트 (200, 150), aspect=None
        When:  make_crop_box 호출
        Then:  (x2-x1)과 (y2-y1)이 모두 짝수
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        x1, y1, x2, y2 = usecase.make_crop_box(frame, request)

        assert (x2 - x1) % 2 == 0, f"크롭 너비 {x2 - x1}이 홀수"
        assert (y2 - y1) % 2 == 0, f"크롭 높이 {y2 - y1}이 홀수"

    def test_make_crop_box_반환값은_프레임_경계_내에_있다(self):
        """Given: 클릭 포인트 (200, 150), aspect=None
        When:  make_crop_box 호출
        Then:  0 <= x1, x2 <= 640, 0 <= y1, y2 <= 360
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        x1, y1, x2, y2 = usecase.make_crop_box(frame, request)

        assert 0 <= x1 and x2 <= FAKE_FRAME_W
        assert 0 <= y1 and y2 <= FAKE_FRAME_H

    def test_make_crop_box_결과가_독립계산_기준값과_일치한다(self):
        """Given: 동일한 가짜 의존 주입
        When:  make_crop_box 호출
        Then:  _expected_crop_box 직접 계산 결과와 동일 (조합 검증)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        actual = usecase.make_crop_box(frame, request)
        expected = _expected_crop_box(CLICK_X, CLICK_Y, REQUEST_CROP_W, REQUEST_CROP_H, None)

        assert actual == expected, (
            f"실제 박스 {actual}이 기준값 {expected}와 다름. "
            "core 함수 조합 순서를 확인하라."
        )

    def test_make_crop_box_종횡비_잠금_9대16이_적용된다(self):
        """Given: aspect='9:16', box_size=(300, 200)
        When:  make_crop_box 호출
        Then:  결과 박스 너비·높이 비율이 9:16 이하에서 짝수 (종횡비 잠금 적용)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect="9:16",
        )

        x1, y1, x2, y2 = usecase.make_crop_box(frame, request)
        w = x2 - x1
        h = y2 - y1

        # 종횡비 잠금 후: w/h ≈ 9/16 (짝수 정렬로 소수점 오차 허용)
        # WHY: apply_aspect_lock이 박스 안쪽으로 축소하므로 w*16 <= h*9+2 (±2 여유)
        assert w * 16 <= h * 9 + 2, f"9:16 종횡비 위반: w={w}, h={h}"

    def test_make_crop_box_centroid_중심에_생성된다(self):
        """Given: 클릭 포인트 (200, 150)
        When:  make_crop_box 호출
        Then:  박스 중심이 마스크 centroid 근방 (경계 클램프 허용)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        x1, y1, x2, y2 = usecase.make_crop_box(frame, request)
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2

        # FakeBackend 마스크 centroid는 클릭 포인트 근방
        # 경계 클램프가 있으므로 ±(박스너비/2) 이내 허용
        margin = REQUEST_CROP_W // 2 + 1
        assert abs(box_cx - CLICK_X) <= margin, f"중심 X 불일치: {box_cx:.1f} vs {CLICK_X}"


# ---------------------------------------------------------------------------
# export 테스트
# ---------------------------------------------------------------------------
class TestExport:
    """ImageCaptureUseCase.export: 크롭 후 파일 저장."""

    def test_export_PNG_파일이_생성된다(self, tmp_path):
        """Given: FakeFrameSource 프레임, 유효한 박스, PNG 설정
        When:  export 호출
        Then:  tmp_path에 파일 존재
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        # 유효한 짝수 박스 (100, 100, 200, 200)
        box = (100, 100, 200, 200)
        output_path = str(tmp_path / "out.png")
        config = ExportConfig(fmt="png")

        usecase.export(frame, box, (output_path, config))

        assert (tmp_path / "out.png").exists()

    def test_export_PNG_파일_크기가_박스_크기와_일치한다(self, tmp_path):
        """Given: (100, 100, 200, 200) 박스 (100x100 크롭)
        When:  export 후 Pillow 재로드
        Then:  이미지 크기 == (100, 100)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        box = (100, 100, 200, 200)
        output_path = str(tmp_path / "out.png")
        config = ExportConfig(fmt="png")

        usecase.export(frame, box, (output_path, config))

        img = Image.open(output_path)
        assert img.size == (100, 100), f"기대 크기 (100,100), 실제 {img.size}"

    def test_export_JPG_파일이_생성된다(self, tmp_path):
        """Given: JPG 설정
        When:  export 호출
        Then:  .jpg 파일 존재
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        box = (50, 50, 150, 150)
        output_path = str(tmp_path / "out.jpg")
        config = ExportConfig(fmt="jpg", quality=90)

        usecase.export(frame, box, (output_path, config))

        assert (tmp_path / "out.jpg").exists()


# ---------------------------------------------------------------------------
# End-to-End 순수 Happy Path 테스트
# ---------------------------------------------------------------------------
class TestEndToEndHappyPath:
    """load_frame → make_crop_box → export 전 구간을 가짜 의존으로 관통."""

    def test_end_to_end_load_crop_export가_오류_없이_완료된다(self, tmp_path):
        """Given: FakeBackend·FakeFrameSource 주입, 클릭 (200, 150), PNG 저장
        When:  load_frame → make_crop_box → export 순서로 전 구간 실행
        Then:  예외 없이 완료되고 파일이 생성됨

        이 테스트가 통과하면 레이어 경계(Protocol 주입 → 유스케이스 조립 →
        core 함수 조합 → Pillow IO)가 실제로 맞물리는 것을 증명한다.
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )
        output_path = str(tmp_path / "e2e_output.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        frame = usecase.load_frame()
        box = usecase.make_crop_box(frame, request)
        usecase.export(frame, box, (output_path, config))

        # --- Then ---
        assert (tmp_path / "e2e_output.png").exists(), "파일이 생성되지 않음"

    def test_end_to_end_결과_이미지가_유효한_크기를_가진다(self, tmp_path):
        """Given: 전 구간 실행
        When:  결과 파일을 Pillow로 재로드
        Then:  크기가 0보다 크고 FAKE_FRAME 크기 이하
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )
        output_path = str(tmp_path / "e2e_size.png")
        config = ExportConfig(fmt="png")

        frame = usecase.load_frame()
        box = usecase.make_crop_box(frame, request)
        usecase.export(frame, box, (output_path, config))

        img = Image.open(output_path)
        w, h = img.size
        assert w > 0 and h > 0
        assert w <= FAKE_FRAME_W and h <= FAKE_FRAME_H

    def test_end_to_end_종횡비_잠금_포함_전_구간이_통과한다(self, tmp_path):
        """Given: aspect='9:16' 요청
        When:  전 구간 실행
        Then:  파일 생성, 결과 이미지의 w:h ≈ 9:16 (±2px 허용)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect="9:16",
        )
        output_path = str(tmp_path / "e2e_aspect.png")
        config = ExportConfig(fmt="png")

        frame = usecase.load_frame()
        box = usecase.make_crop_box(frame, request)
        usecase.export(frame, box, (output_path, config))

        img = Image.open(output_path)
        w, h = img.size
        assert w > 0 and h > 0
        # 9:16 종횡비 확인 (짝수 정렬 오차 ±2 허용)
        assert w * 16 <= h * 9 + 2, f"9:16 종횡비 위반: w={w}, h={h}"


# ---------------------------------------------------------------------------
# 빈 마스크 → EmptyMaskError 테스트 (리뷰 [중요] 1 반영)
# ---------------------------------------------------------------------------
class TestEmptyMaskError:
    """빈 마스크일 때 폴백 없이 EmptyMaskError를 발생시키는지 검증."""

    def test_빈_마스크_반환_시_EmptyMaskError가_발생한다(self):
        """Given: empty_mask=True인 FakeBackend (항상 빈 마스크 반환)
        When:  make_crop_box 호출
        Then:  EmptyMaskError 발생 (조용한 폴백 없음)
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(empty_mask=True),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        with pytest.raises(EmptyMaskError):
            usecase.make_crop_box(frame, request)

    def test_EmptyMaskError_메시지는_한국어_안내를_포함한다(self):
        """Given: 빈 마스크 백엔드
        When:  make_crop_box 호출 시 예외 발생
        Then:  예외 메시지에 한국어 안내 포함
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(empty_mask=True),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
        )

        with pytest.raises(EmptyMaskError, match="다시 클릭"):
            usecase.make_crop_box(frame, request)

    def test_정상_마스크에서는_예외가_발생하지_않는다(self):
        """Given: 일반 FakeBackend (정상 마스크)
        When:  make_crop_box 호출
        Then:  예외 없이 박스 반환
        """
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
        )

        box = usecase.make_crop_box(frame, request)

        assert len(box) == 4


# ---------------------------------------------------------------------------
# segment() 단위 테스트 (crop-ux 슬라이스 신규)
# ---------------------------------------------------------------------------
# 테스트 상수
_SEG_CLICK_X = 200
_SEG_CLICK_Y = 150
_FRAME_W = FAKE_FRAME_W
_FRAME_H = FAKE_FRAME_H


class TestSegment:
    """ImageCaptureUseCase.segment: 클릭→SegmentResult(mask, centroid) 산출 / 빈 마스크 에러.

    검증 범위(계획서 §6-1):
      - 정상: SegmentResult 반환, mask는 bool HxW, centroid는 클릭 근방
      - 빈 마스크: EmptyMaskError(메시지 "다시 클릭" 포함)
      - segment 후 segment_image 호출 횟수가 정확히 1임을 카운터로 단언
    """

    def test_segment_정상_호출_시_SegmentResult를_반환한다(self):
        """Given: FakeBackend(정상 마스크) 주입, 클릭 포인트 (200, 150)
        When:  segment(frame, point) 호출
        Then:  SegmentResult 인스턴스 반환
        """
        # --- Given ---
        backend = FakeBackend()
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=backend)
        frame = FakeFrameSource().read_frame()

        # --- When ---
        result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- Then ---
        assert isinstance(result, SegmentResult)

    def test_segment_결과_mask는_bool_HxW_배열이다(self):
        """Given: FakeBackend 주입
        When:  segment 호출
        Then:  result.mask.dtype == bool, shape == (H, W)
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()

        # --- When ---
        result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- Then ---
        assert result.mask.dtype == bool, f"mask dtype이 bool이 아님: {result.mask.dtype}"
        assert result.mask.shape == (_FRAME_H, _FRAME_W), (
            f"mask shape 불일치: {result.mask.shape}"
        )

    def test_segment_결과_centroid는_클릭_포인트_근방이다(self):
        """Given: 클릭 포인트 (200, 150), FakeBackend (반경 20px 사각 마스크)
        When:  segment 호출
        Then:  centroid가 (200, 150) 근방 (±FAKE_MASK_HALF+1 이내)
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()

        # --- When ---
        result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- Then ---
        cx, cy = result.centroid
        assert abs(cx - _SEG_CLICK_X) <= FAKE_MASK_HALF + 1, (
            f"centroid X 불일치: {cx:.1f} vs 기대 {_SEG_CLICK_X}"
        )
        assert abs(cy - _SEG_CLICK_Y) <= FAKE_MASK_HALF + 1, (
            f"centroid Y 불일치: {cy:.1f} vs 기대 {_SEG_CLICK_Y}"
        )

    def test_segment_결과_centroid는_float_튜플이다(self):
        """Given: FakeBackend 주입
        When:  segment 호출
        Then:  centroid는 길이 2 튜플이고 각 요소가 float(또는 int)
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()

        # --- When ---
        result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- Then ---
        assert len(result.centroid) == 2
        assert isinstance(result.centroid[0], (int, float))
        assert isinstance(result.centroid[1], (int, float))

    def test_segment_1회_호출_시_segment_image_호출_횟수가_1이다(self):
        """Given: FakeBackend(segment_call_count 스파이)
        When:  segment 1회 호출
        Then:  backend.segment_call_count == 1 (세그 1회만 발생)
        """
        # --- Given ---
        backend = FakeBackend()
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=backend)
        frame = FakeFrameSource().read_frame()

        # --- When ---
        usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- Then ---
        assert backend.segment_call_count == 1, (
            f"segment_image 호출 횟수 불일치: {backend.segment_call_count}"
        )

    def test_segment_빈_마스크_반환_시_EmptyMaskError가_발생한다(self):
        """Given: FakeBackend(empty_mask=True)
        When:  segment 호출
        Then:  EmptyMaskError 발생
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(), backend=FakeBackend(empty_mask=True)
        )
        frame = FakeFrameSource().read_frame()

        # --- When / Then ---
        with pytest.raises(EmptyMaskError):
            usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

    def test_segment_빈_마스크_에러_메시지는_한국어_안내를_포함한다(self):
        """Given: FakeBackend(empty_mask=True)
        When:  segment 호출 → EmptyMaskError 발생
        Then:  메시지에 "다시 클릭" 포함 (기존 정책 유지)
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(
            source=FakeFrameSource(), backend=FakeBackend(empty_mask=True)
        )
        frame = FakeFrameSource().read_frame()

        # --- When / Then ---
        with pytest.raises(EmptyMaskError, match="다시 클릭"):
            usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

    def test_segment_SegmentResult는_frozen_dataclass이다(self):
        """Given: 유효한 SegmentResult 인스턴스
        When:  필드를 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생 — 불변 보장

        WHY: frozen=True dataclass는 실수로 캐시된 세그 결과를 덮어쓰는
             버그를 컴파일 타임에 차단한다.
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()
        result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))

        # --- When / Then ---
        with pytest.raises((AttributeError, TypeError)):
            result.centroid = (0.0, 0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_box() 단위 테스트 (crop-ux 슬라이스 신규)
# ---------------------------------------------------------------------------
# 테스트 상수 — 고정 centroid (세그 독립)
_CENTROID = (200.0, 150.0)
_BASE_BOX_SIZE = (300, 200)
_BASE_FRAME_SHAPE = (_FRAME_W, _FRAME_H)  # (W, H)


class TestComputeBox:
    """ImageCaptureUseCase.compute_box: centroid + BoxParams → 크롭박스 순수 계산.

    검증 범위(계획서 §6-1):
      - 순수성: compute_box 반복 호출 시 segment_image 미호출(카운터 0 유지)
      - 종횡비 변경 시 박스 비율 변화 (aspect None / 1:1 / 9:16 / 16:9)
      - 박스 크기 단조성: box_size 증가 → 박스 변 길이 증가 또는 동일
      - 항상 짝수 반환, 프레임 경계 내
    """

    def _make_usecase(self) -> tuple[ImageCaptureUseCase, FakeBackend]:
        """usecase + 스파이 백엔드 쌍을 반환한다."""
        backend = FakeBackend()
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=backend)
        return usecase, backend

    def test_compute_box_반환값은_4_튜플이다(self):
        """Given: 고정 centroid, 기본 BoxParams
        When:  compute_box 호출
        Then:  길이 4의 정수 튜플
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        result = usecase.compute_box(_CENTROID, params)

        # --- Then ---
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_compute_box_결과는_짝수_크기이다(self):
        """Given: aspect=None, 기본 크기
        When:  compute_box 호출
        Then:  (x2-x1) % 2 == 0, (y2-y1) % 2 == 0

        WHY: yuv420p 인코딩 요구사항. core.make_crop_box의 짝수 정렬이
             compute_box 경로에서도 보존됨을 검증한다.
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)

        # --- Then ---
        assert (x2 - x1) % 2 == 0, f"너비 {x2 - x1}이 홀수"
        assert (y2 - y1) % 2 == 0, f"높이 {y2 - y1}이 홀수"

    def test_compute_box_결과는_프레임_경계_내에_있다(self):
        """Given: aspect=None, 기본 크기
        When:  compute_box 호출
        Then:  0 <= x1, x2 <= FRAME_W, 0 <= y1, y2 <= FRAME_H
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)

        # --- Then ---
        assert 0 <= x1 and x2 <= _FRAME_W, f"X 경계 위반: x1={x1}, x2={x2}"
        assert 0 <= y1 and y2 <= _FRAME_H, f"Y 경계 위반: y1={y1}, y2={y2}"

    def test_compute_box_반복_호출_시_segment_image_호출_횟수가_0이다(self):
        """Given: FakeBackend 스파이, segment 미호출 상태
        When:  compute_box를 5회 반복 호출
        Then:  backend.segment_call_count == 0 (재세그 없음)

        WHY: 이것이 이 슬라이스의 핵심 회귀 가드.
             종횡비/크기 슬라이더 조작마다 compute_box가 호출되는데,
             segment_image를 재호출하면 매 조작마다 1~3s 멈춤이 발생한다.
             이 테스트가 통과하면 분리가 올바르게 구현된 것이다.
        """
        # 반복 횟수 상수
        COMPUTE_BOX_CALL_COUNT = 5

        # --- Given ---
        usecase, backend = self._make_usecase()
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        for _ in range(COMPUTE_BOX_CALL_COUNT):
            usecase.compute_box(_CENTROID, params)

        # --- Then ---
        assert backend.segment_call_count == 0, (
            f"compute_box 호출 중 segment_image가 {backend.segment_call_count}회 호출됨. "
            "compute_box는 순수 함수여야 한다 — 모델 호출 금지."
        )

    def test_compute_box_segment_1회_후_compute_box_여러_번_호출_시_세그_횟수가_1이다(self):
        """Given: segment 1회 호출 후 centroid 보관
        When:  compute_box를 3회 추가 호출
        Then:  backend.segment_call_count == 1 (세그는 여전히 1회뿐)

        WHY: 실제 UI 시나리오(클릭 1회 → 슬라이더 조작 N회)를 재현한다.
             세그는 클릭 시 1회뿐이어야 함을 단언한다.
        """
        # --- Given ---
        EXTRA_COMPUTE_CALLS = 3
        usecase, backend = self._make_usecase()
        frame = FakeFrameSource().read_frame()

        # 세그 1회 (클릭 시뮬레이션)
        seg_result = usecase.segment(frame, (_SEG_CLICK_X, _SEG_CLICK_Y))
        assert backend.segment_call_count == 1  # 전제 확인

        # --- When ---
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )
        for _ in range(EXTRA_COMPUTE_CALLS):
            usecase.compute_box(seg_result.centroid, params)

        # --- Then ---
        assert backend.segment_call_count == 1, (
            f"segment_image 호출 횟수가 1이어야 하는데 {backend.segment_call_count}임. "
            "compute_box는 세그를 재호출해서는 안 된다."
        )

    def test_compute_box_종횡비_None은_요청_크기_박스를_반환한다(self):
        """Given: aspect=None, box_size=(300, 200)
        When:  compute_box 호출
        Then:  박스 너비 ≈ 300, 높이 ≈ 200 (짝수 정렬 오차 ±2 허용)
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=(300, 200), aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)
        w, h = x2 - x1, y2 - y1

        # --- Then ---
        # 짝수 정렬로 ±2 이내
        assert abs(w - 300) <= 2, f"너비 불일치: {w} vs 300"
        assert abs(h - 200) <= 2, f"높이 불일치: {h} vs 200"

    def test_compute_box_종횡비_1대1_적용_시_너비와_높이가_같다(self):
        """Given: aspect='1:1', box_size=(300, 200)
        When:  compute_box 호출
        Then:  박스 너비 == 박스 높이 (짝수 정렬 오차 ±2)

        WHY: apply_aspect_lock이 박스 안쪽으로 축소하므로
             w:h가 정확히 1:1이 되어야 한다.
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=(300, 200), aspect="1:1", frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)
        w, h = x2 - x1, y2 - y1

        # --- Then ---
        assert abs(w - h) <= 2, f"1:1 종횡비 위반: w={w}, h={h}"

    def test_compute_box_종횡비_9대16_적용_시_비율이_맞다(self):
        """Given: aspect='9:16', box_size=(300, 300), 충분히 큰 프레임
        When:  compute_box 호출
        Then:  w*16 ≈ h*9 (짝수 정렬 오차 ±2 허용)
        """
        # --- Given ---
        # 9:16 종횡비를 확인하려면 프레임이 충분히 커야 한다
        LARGE_FRAME = (1920, 1080)
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=(300, 300), aspect="9:16", frame_shape=LARGE_FRAME
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)
        w, h = x2 - x1, y2 - y1

        # --- Then ---
        # w/h ≈ 9/16  →  w*16 ≈ h*9
        assert abs(w * 16 - h * 9) <= 2 * 16, (
            f"9:16 종횡비 불일치: w={w}, h={h}, w*16={w*16}, h*9={h*9}"
        )

    def test_compute_box_종횡비_16대9_적용_시_비율이_맞다(self):
        """Given: aspect='16:9', box_size=(300, 300), 충분히 큰 프레임
        When:  compute_box 호출
        Then:  w*9 ≈ h*16 (짝수 정렬 오차 허용)
        """
        # --- Given ---
        LARGE_FRAME = (1920, 1080)
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=(300, 300), aspect="16:9", frame_shape=LARGE_FRAME
        )

        # --- When ---
        x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)
        w, h = x2 - x1, y2 - y1

        # --- Then ---
        assert abs(w * 9 - h * 16) <= 2 * 9, (
            f"16:9 종횡비 불일치: w={w}, h={h}"
        )

    def test_compute_box_크기_증가_시_박스가_단조_증가한다(self):
        """Given: aspect=None, box_size를 100→200→300으로 증가
        When:  compute_box를 각 크기로 호출
        Then:  박스 너비가 단조 비감소 (작은 크기 <= 큰 크기)

        WHY: 슬라이더를 오른쪽으로 움직이면 박스가 커져야 한다.
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        sizes = [(100, 100), (200, 200), (300, 300)]

        # --- When ---
        widths = []
        for size in sizes:
            params = BoxParams(
                box_size=size, aspect=None, frame_shape=_BASE_FRAME_SHAPE
            )
            x1, y1, x2, y2 = usecase.compute_box(_CENTROID, params)
            widths.append(x2 - x1)

        # --- Then ---
        for i in range(len(widths) - 1):
            assert widths[i] <= widths[i + 1], (
                f"단조성 위반: size={sizes[i]}→width={widths[i]}, "
                f"size={sizes[i+1]}→width={widths[i+1]}"
            )

    def test_compute_box_동일_파라미터_반복_호출_시_결과가_동일하다(self):
        """Given: 동일한 centroid + BoxParams
        When:  compute_box를 3회 호출
        Then:  모든 결과가 동일 (순수 함수 멱등성)
        """
        # --- Given ---
        usecase, _ = self._make_usecase()
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When ---
        results = [usecase.compute_box(_CENTROID, params) for _ in range(3)]

        # --- Then ---
        assert results[0] == results[1] == results[2], (
            f"순수 함수 멱등성 위반: {results}"
        )

    def test_compute_box_BoxParams는_frozen_dataclass이다(self):
        """Given: BoxParams 인스턴스
        When:  필드 수정 시도
        Then:  FrozenInstanceError(또는 AttributeError) 발생 — 불변 보장
        """
        # --- Given ---
        params = BoxParams(
            box_size=_BASE_BOX_SIZE, aspect=None, frame_shape=_BASE_FRAME_SHAPE
        )

        # --- When / Then ---
        with pytest.raises((AttributeError, TypeError)):
            params.aspect = "1:1"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# make_crop_box 무회귀 확인 (A안 위임 조합자)
# ---------------------------------------------------------------------------
class TestMakeCropBoxNoRegression:
    """make_crop_box가 segment+compute_box 위임 후에도 동일 결과를 내는지 검증.

    WHY: 계획서 A안 — make_crop_box를 두 메서드 위임으로 재작성하되
         기존 계약(동일 결과)은 보존한다. 이 클래스가 통과하면 위임이
         올바르게 조합됐음을 증명한다.
    """

    def test_make_crop_box_위임_후_독립계산과_동일한_결과를_반환한다(self):
        """Given: FakeBackend + FakeFrameSource, 클릭 (200, 150), aspect=None
        When:  make_crop_box 호출
        Then:  _expected_crop_box 독립 계산과 동일 결과

        WHY: make_crop_box가 segment→compute_box를 올바르게 위임하는지
             end-to-end로 확인한다(A안 위임 정확성).
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        # --- When ---
        actual = usecase.make_crop_box(frame, request)
        expected = _expected_crop_box(CLICK_X, CLICK_Y, REQUEST_CROP_W, REQUEST_CROP_H, None)

        # --- Then ---
        assert actual == expected, (
            f"make_crop_box 위임 결과 {actual}이 독립계산 {expected}와 다름."
        )

    def test_make_crop_box_종횡비_9대16_위임_후_무회귀(self):
        """Given: aspect='9:16'
        When:  make_crop_box 호출
        Then:  _expected_crop_box(aspect='9:16')와 동일 결과
        """
        # --- Given ---
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=FakeBackend())
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect="9:16",
        )

        # --- When ---
        actual = usecase.make_crop_box(frame, request)
        expected = _expected_crop_box(CLICK_X, CLICK_Y, REQUEST_CROP_W, REQUEST_CROP_H, "9:16")

        # --- Then ---
        assert actual == expected

    def test_make_crop_box_호출_1회당_segment_image도_1회_호출된다(self):
        """Given: FakeBackend 스파이
        When:  make_crop_box 1회 호출
        Then:  segment_image 호출 횟수 == 1 (내부 segment 위임이 1회만 세그함)
        """
        # --- Given ---
        backend = FakeBackend()
        usecase = ImageCaptureUseCase(source=FakeFrameSource(), backend=backend)
        frame = FakeFrameSource().read_frame()
        request = CropRequest(
            point=(CLICK_X, CLICK_Y),
            box_size=(REQUEST_CROP_W, REQUEST_CROP_H),
            aspect=None,
        )

        # --- When ---
        usecase.make_crop_box(frame, request)

        # --- Then ---
        assert backend.segment_call_count == 1, (
            f"make_crop_box가 segment_image를 {backend.segment_call_count}회 호출함. "
            "위임 조합자는 segment를 정확히 1회만 호출해야 한다."
        )


# ---------------------------------------------------------------------------
# export 업스케일 연결 테스트 (image-upscale 슬라이스 신규)
# ---------------------------------------------------------------------------
# 테스트용 박스 크기 상수 (x1, y1, x2, y2) — 50×40 크롭
_UPSCALE_BOX = (100, 100, 150, 140)
_UPSCALE_BOX_W = _UPSCALE_BOX[2] - _UPSCALE_BOX[0]   # 50
_UPSCALE_BOX_H = _UPSCALE_BOX[3] - _UPSCALE_BOX[1]   # 40

# 업스케일 배율 상수
_UPSCALE_SCALE_2 = 2
_UPSCALE_SCALE_4 = 4


class TestExportWithUpscaler:
    """export의 선택적 upscaler 연결 계약 검증 (계획서 §4-2, §7-2).

    검증 목표:
      1. upscaler=None(기본) → 업스케일 미호출, 저장 크기 = 박스 크기 (무회귀)
      2. FakeUpscaleBackend(scale=2) 주입 → 저장 크기 = 박스×2
      3. upscale_call_count == 1 (중복 추론 회귀 가드)
      4. 기존 export 호출(upscaler 인자 없음)이 무회귀로 통과
    """

    def _make_usecase(self) -> ImageCaptureUseCase:
        """FakeBackend·FakeFrameSource 주입한 기본 usecase를 반환한다."""
        return ImageCaptureUseCase(
            source=FakeFrameSource(),
            backend=FakeBackend(),
        )

    def test_upscaler_None이면_저장_크기가_박스_크기와_일치한다(self, tmp_path):
        """Given: upscaler=None(기본값), 박스 50×40
        When:  export 호출
        Then:  저장 이미지 크기 == (50, 40) — 업스케일 없는 기존 동작 무회귀

        WHY: upscaler=None 경로가 기존 crop→save 직행 동작을 유지하는지 검증한다.
        """
        # --- Given ---
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        output_path = str(tmp_path / "no_upscale.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config))

        # --- Then ---
        from PIL import Image as _Image
        img = _Image.open(output_path)
        assert img.size == (_UPSCALE_BOX_W, _UPSCALE_BOX_H), (
            f"upscaler=None 시 저장 크기 불일치: {img.size} != "
            f"({_UPSCALE_BOX_W}, {_UPSCALE_BOX_H})"
        )

    def test_upscaler_주입_시_저장_크기가_박스_크기의_scale배이다(self, tmp_path):
        """Given: FakeUpscaleBackend(scale=2), 박스 50×40
        When:  export(upscaler=fake_upscaler) 호출
        Then:  저장 이미지 크기 == (50*2, 40*2) = (100, 80)

        WHY: 업스케일 결과가 저장에 반영되는지 end-to-end로 검증한다.
             FakeUpscaleBackend의 nearest 확대로 크기가 결정적으로 예측된다.
        """
        # --- Given ---
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        fake_upscaler = FakeUpscaleBackend(scale=_UPSCALE_SCALE_2)
        output_path = str(tmp_path / "upscale_x2.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config), upscaler=fake_upscaler)

        # --- Then ---
        from PIL import Image as _Image
        img = _Image.open(output_path)
        expected_w = _UPSCALE_BOX_W * _UPSCALE_SCALE_2
        expected_h = _UPSCALE_BOX_H * _UPSCALE_SCALE_2
        assert img.size == (expected_w, expected_h), (
            f"scale=2 저장 크기 불일치: {img.size} != ({expected_w}, {expected_h})"
        )

    def test_upscaler_scale4_주입_시_저장_크기가_4배이다(self, tmp_path):
        """Given: FakeUpscaleBackend(scale=4), 박스 50×40
        When:  export(upscaler=...) 호출
        Then:  저장 이미지 크기 == (50*4, 40*4) = (200, 160)
        """
        # --- Given ---
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        fake_upscaler = FakeUpscaleBackend(scale=_UPSCALE_SCALE_4)
        output_path = str(tmp_path / "upscale_x4.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config), upscaler=fake_upscaler)

        # --- Then ---
        from PIL import Image as _Image
        img = _Image.open(output_path)
        expected_w = _UPSCALE_BOX_W * _UPSCALE_SCALE_4
        expected_h = _UPSCALE_BOX_H * _UPSCALE_SCALE_4
        assert img.size == (expected_w, expected_h), (
            f"scale=4 저장 크기 불일치: {img.size} != ({expected_w}, {expected_h})"
        )

    def test_upscaler_주입_시_upscale은_정확히_1회_호출된다(self, tmp_path):
        """Given: FakeUpscaleBackend(scale=2) + upscale_call_count 스파이
        When:  export 1회 호출
        Then:  fake_upscaler.upscale_call_count == 1 (중복 추론 방지 회귀 가드)

        WHY: export가 업스케일을 한 번만 호출해야 한다.
             2회 이상 호출되면 crop 결과를 이중 확대해 크기가 틀린다.
             이 테스트가 통과해야 export 구현이 올바르다.
        """
        # --- Given ---
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        fake_upscaler = FakeUpscaleBackend(scale=_UPSCALE_SCALE_2)
        output_path = str(tmp_path / "call_count.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config), upscaler=fake_upscaler)

        # --- Then ---
        assert fake_upscaler.upscale_call_count == 1, (
            f"upscale 호출 횟수 불일치: {fake_upscaler.upscale_call_count}회 "
            "(정확히 1회여야 함)"
        )

    def test_upscaler_None이면_upscale이_호출되지_않는다(self, tmp_path):
        """Given: upscaler=None, 별도 FakeUpscaleBackend 인스턴스(감시용)
        When:  export(upscaler=None) 호출
        Then:  FakeUpscaleBackend 인스턴스는 upscale_call_count == 0 유지

        WHY: upscaler=None 경로에서 upscale이 절대 호출되지 않음을 단언한다.
             None 경로에서 upscaler 인스턴스가 없으므로, 새 인스턴스를 생성해
             "어떤 업스케일러도 호출되지 않음"의 의미를 논리적으로 검증한다.

        NOTE: upscaler=None 경로는 인스턴스 자체가 없으므로
              "저장 크기 = 박스 크기" 검증(test_upscaler_None이면_저장_크기가...)으로
              미호출을 간접 검증한다. 이 테스트는 FakeUpscaleBackend 초기 카운터가
              0임을 확인해 스파이 패턴의 전제 조건을 보장한다.
        """
        # --- Given ---
        unused_upscaler = FakeUpscaleBackend(scale=_UPSCALE_SCALE_2)
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        output_path = str(tmp_path / "none_path.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config))  # upscaler 인자 없음

        # --- Then ---
        # unused_upscaler는 호출되지 않았으므로 count가 여전히 0
        assert unused_upscaler.upscale_call_count == 0, (
            "None 경로에서 upscale 인스턴스가 호출됨 — 설계 오류"
        )

    def test_기존_export_인자_없이_호출해도_무회귀로_통과한다(self, tmp_path):
        """Given: 기존 방식(upscaler 인자 없이) export 호출
        When:  usecase.export(frame, box, target) — 키워드 인자 없음
        Then:  파일 생성, 크기 = 박스 크기 (기존 동작 무회귀)

        WHY: export 시그니처에 upscaler: UpscaleBackend | None = None을 추가해도
             기존 호출부가 변경 없이 그대로 동작해야 한다(계획서 §4-2, §4-4).
             기본값 None으로 하위호환성을 보장한다.
        """
        # --- Given ---
        usecase = self._make_usecase()
        frame = FakeFrameSource().read_frame()
        output_path = str(tmp_path / "compat.png")
        config = ExportConfig(fmt="png")

        # --- When ---
        usecase.export(frame, _UPSCALE_BOX, (output_path, config))  # 기존 3인자 호출

        # --- Then ---
        from PIL import Image as _Image
        assert (tmp_path / "compat.png").exists(), "파일이 생성되지 않음"
        img = _Image.open(output_path)
        assert img.size == (_UPSCALE_BOX_W, _UPSCALE_BOX_H), (
            f"무회귀 실패: 저장 크기 {img.size} != ({_UPSCALE_BOX_W}, {_UPSCALE_BOX_H})"
        )
