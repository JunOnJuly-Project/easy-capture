"""ImageCaptureUseCase 슬라이스 핵심 조립 테스트.

대상 모듈: easy_capture.app.image_capture
테스트 더블: FakeBackend, FakeFrameSource (tests/fixtures/fakes.py)

이 테스트 파일이 검증하는 계약:
  1. load_frame()       → FakeFrameSource가 반환하는 고정 프레임
  2. make_crop_box()    → FakeBackend 마스크의 centroid + 종횡비 잠금 + 짝수 + 경계 클램프
  3. export()           → tmp_path에 파일 생성, 크롭 크기 일치
  4. end-to-end happy path → 전 구간을 가짜 의존으로 관통
  5. Protocol 계약      → isinstance(FakeBackend(), SegmentationBackend) 통과

NOTE: 구현 없으므로 import 실패로 Red 상태. TDD 정상.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from easy_capture.app.image_capture import CropRequest, EmptyMaskError, ImageCaptureUseCase
from easy_capture.core.crop import (
    apply_aspect_lock,
    centroid_of_mask,
    make_crop_box,
)
from easy_capture.core.export.image_export import ExportConfig
from easy_capture.core.segmentation.backend import SegmentationBackend
from tests.fixtures.fakes import FakeBackend, FakeFrameSource

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
