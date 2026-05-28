"""업스케일 슬라이스 단위 테스트.

검증 범위:
  1. UpscaleBackend Protocol 계약 — FakeUpscaleBackend 구조적 준수
  2. FakeUpscaleBackend 동작 — scale 속성·shape·dtype·upscale_call_count 스파이
  3. reconstruction_to_rgb_uint8 순수 함수 — shape·dtype·clamp·채널 순서

torch import 없음(경계 불변식 검증):
  core/upscale은 numpy만 의존. 이 테스트 파일은 torch를 단 한 줄도 import하지 않는다.
  FakeUpscaleBackend 모의 입력도 numpy ndarray로만 구성한다.

구현 전이므로 core/upscale 패키지 import 실패로 RED 상태가 정상.
"""
from __future__ import annotations

import numpy as np
import pytest

# --- 구현 전 RED 예상 import ---
# WHY: try/except로 감싸는 이유 — core/upscale 패키지가 아직 없을 때
#      파일 import 자체가 실패해 pytest가 수집조차 못하는 대신,
#      각 테스트가 "구현 없음" 으로 개별 FAIL을 내도록 한다.
#      이로써 어떤 테스트가 RED인지 명확히 확인할 수 있다.
try:
    from easy_capture.core.upscale.backend import UpscaleBackend
    _HAS_UPSCALE_BACKEND = True
except ModuleNotFoundError:
    UpscaleBackend = None  # type: ignore[assignment,misc]
    _HAS_UPSCALE_BACKEND = False

try:
    from easy_capture.core.upscale.normalize import reconstruction_to_rgb_uint8
    _HAS_NORMALIZE = True
except ModuleNotFoundError:
    reconstruction_to_rgb_uint8 = None  # type: ignore[assignment]
    _HAS_NORMALIZE = False

from tests.fixtures.fakes import FakeUpscaleBackend

# 구현 없음 메시지 상수
_MSG_NO_BACKEND = "core/upscale/backend.py 미구현 — UpscaleBackend Protocol 없음"
_MSG_NO_NORMALIZE = "core/upscale/normalize.py 미구현 — reconstruction_to_rgb_uint8 없음"

# ---------------------------------------------------------------------------
# 테스트 상수
# ---------------------------------------------------------------------------
# 테스트용 더미 이미지 크기 (작은 값으로 빠른 테스트)
_FAKE_H = 8
_FAKE_W = 12
_FAKE_C = 3

# 정규화 함수 검증용 CHW 입력 크기
_NORM_C = 3
_NORM_H = 6
_NORM_W = 10

# 클램프 경계 상수
_CLAMP_MIN = 0.0
_CLAMP_MAX = 1.0
_UINT8_MAX = 255

# 0.5 → 128 반올림 기준값 (np.rint(0.5 * 255) = 128.0 → 128)
_HALF_FLOAT = 0.5
_HALF_UINT8 = 128


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture()
def rgb_image_hwc() -> np.ndarray:
    """테스트용 결정적 RGB HxWx3 uint8 이미지.

    R채널=x좌표, G채널=y좌표 기반 → nearest 확대 후 픽셀값 예측 가능.
    """
    img = np.zeros((_FAKE_H, _FAKE_W, _FAKE_C), dtype=np.uint8)
    img[:, :, 0] = (np.arange(_FAKE_W) * _UINT8_MAX // (_FAKE_W - 1)).astype(np.uint8)
    img[:, :, 1] = (
        (np.arange(_FAKE_H) * _UINT8_MAX // (_FAKE_H - 1)).astype(np.uint8)[:, np.newaxis]
    )
    img[:, :, 2] = 64
    return img


@pytest.fixture()
def chw_float_normal() -> np.ndarray:
    """(3, H, W) float 배열. 값이 [0, 1] 내에 있는 정상 입력."""
    arr = np.zeros((_NORM_C, _NORM_H, _NORM_W), dtype=np.float32)
    arr[0] = 0.0    # R채널 전부 0.0
    arr[1] = 1.0    # G채널 전부 1.0
    arr[2] = _HALF_FLOAT  # B채널 전부 0.5
    return arr


@pytest.fixture()
def chw_float_out_of_range() -> np.ndarray:
    """(3, H, W) float 배열. [0,1] 범위 밖 값 포함 (클램프 검증용)."""
    arr = np.zeros((_NORM_C, _NORM_H, _NORM_W), dtype=np.float32)
    arr[0] = -0.5   # 음수 → 클램프 후 0
    arr[1] = 1.5    # 1 초과 → 클램프 후 255
    arr[2] = _HALF_FLOAT
    return arr


# ---------------------------------------------------------------------------
# 1. UpscaleBackend Protocol 계약 + FakeUpscaleBackend
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_UPSCALE_BACKEND, reason=_MSG_NO_BACKEND)
class TestUpscaleBackendProtocol:
    """FakeUpscaleBackend가 UpscaleBackend Protocol을 올바르게 구현하는지 검증.

    UpscaleBackend는 @runtime_checkable Protocol이므로
    isinstance 검사가 구조적 서브타이핑을 런타임에 확인한다.
    """

    def test_FakeUpscaleBackend_인스턴스는_UpscaleBackend_isinstance를_통과한다(self):
        """Given: FakeUpscaleBackend 인스턴스
        When:  isinstance(..., UpscaleBackend) 호출
        Then:  True 반환 (runtime_checkable Protocol 계약 준수)

        WHY: @runtime_checkable Protocol이므로 구조적 서브타이핑이 런타임에
             검증된다. 이 테스트가 통과해야 실제 주입 시 타입 오류가 없다.
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu", scale=2)

        # --- When / Then ---
        assert isinstance(fake, UpscaleBackend)

    def test_FakeUpscaleBackend_device_속성이_올바르게_설정된다(self):
        """Given: device='cuda'로 생성한 FakeUpscaleBackend
        When:  .device 속성 접근
        Then:  'cuda' 반환 (Protocol 필수 속성)
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cuda", scale=2)

        # --- When / Then ---
        assert fake.device == "cuda"

    def test_FakeUpscaleBackend_scale_속성이_올바르게_설정된다(self):
        """Given: scale=4로 생성한 FakeUpscaleBackend
        When:  .scale 속성 접근
        Then:  4 반환 (Protocol 필수 속성)
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu", scale=4)

        # --- When / Then ---
        assert fake.scale == 4

    def test_FakeUpscaleBackend_scale_기본값은_2이다(self):
        """Given: scale 인자 없이 생성한 FakeUpscaleBackend
        When:  .scale 속성 접근
        Then:  2 반환
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu")

        # --- When / Then ---
        assert fake.scale == 2


# ---------------------------------------------------------------------------
# 2. FakeUpscaleBackend 동작 검증
# ---------------------------------------------------------------------------
class TestFakeUpscaleBackend:
    """FakeUpscaleBackend의 upscale 출력 shape·dtype·스파이 카운터 검증."""

    def test_upscale_출력_shape는_scale배_HWC이다(self, rgb_image_hwc):
        """Given: (H, W, 3) uint8 이미지, scale=2
        When:  upscale 호출
        Then:  shape == (H*2, W*2, 3)

        WHY: scale배 nearest 확대를 검증한다.
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu", scale=2)

        # --- When ---
        result = fake.upscale(rgb_image_hwc)

        # --- Then ---
        expected_h = _FAKE_H * 2
        expected_w = _FAKE_W * 2
        assert result.shape == (expected_h, expected_w, _FAKE_C), (
            f"기대 shape ({expected_h}, {expected_w}, {_FAKE_C}), 실제 {result.shape}"
        )

    def test_upscale_출력_shape_scale4(self, rgb_image_hwc):
        """Given: (H, W, 3) uint8 이미지, scale=4
        When:  upscale 호출
        Then:  shape == (H*4, W*4, 3)
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu", scale=4)

        # --- When ---
        result = fake.upscale(rgb_image_hwc)

        # --- Then ---
        assert result.shape == (_FAKE_H * 4, _FAKE_W * 4, _FAKE_C)

    def test_upscale_출력_dtype은_uint8이다(self, rgb_image_hwc):
        """Given: uint8 입력 이미지, scale=2
        When:  upscale 호출
        Then:  출력 dtype == uint8

        WHY: UpscaleBackend Protocol 반환 계약(HWC uint8)을 검증한다.
        """
        # --- Given ---
        fake = FakeUpscaleBackend(device="cpu", scale=2)

        # --- When ---
        result = fake.upscale(rgb_image_hwc)

        # --- Then ---
        assert result.dtype == np.uint8, f"기대 dtype uint8, 실제 {result.dtype}"

    def test_upscale_호출_전_call_count는_0이다(self):
        """Given: 새로 생성한 FakeUpscaleBackend
        When:  upscale 미호출
        Then:  upscale_call_count == 0 (초기값)
        """
        # --- Given / When ---
        fake = FakeUpscaleBackend()

        # --- Then ---
        assert fake.upscale_call_count == 0

    def test_upscale_1회_호출_후_call_count는_1이다(self, rgb_image_hwc):
        """Given: FakeUpscaleBackend 스파이
        When:  upscale 1회 호출
        Then:  upscale_call_count == 1

        WHY: export가 업스케일을 정확히 1회만 호출하는지 단언하기 위한
             핵심 스파이 카운터가 올바르게 동작하는지 검증한다.
        """
        # --- Given ---
        fake = FakeUpscaleBackend()

        # --- When ---
        fake.upscale(rgb_image_hwc)

        # --- Then ---
        assert fake.upscale_call_count == 1

    def test_upscale_3회_호출_후_call_count는_3이다(self, rgb_image_hwc):
        """Given: FakeUpscaleBackend 스파이
        When:  upscale 3회 호출
        Then:  upscale_call_count == 3 (누적 카운트)
        """
        # --- Given ---
        fake = FakeUpscaleBackend()
        CALL_COUNT = 3

        # --- When ---
        for _ in range(CALL_COUNT):
            fake.upscale(rgb_image_hwc)

        # --- Then ---
        assert fake.upscale_call_count == CALL_COUNT

    def test_upscale_결과가_결정적이다(self, rgb_image_hwc):
        """Given: 동일 입력 이미지
        When:  upscale 두 번 호출
        Then:  두 결과가 동일 (nearest 확대는 결정적)

        WHY: 테스트에서 저장 이미지 크기를 정확히 예측할 수 있어야 한다.
        """
        # --- Given ---
        fake = FakeUpscaleBackend()

        # --- When ---
        result1 = fake.upscale(rgb_image_hwc)
        result2 = fake.upscale(rgb_image_hwc)

        # --- Then ---
        np.testing.assert_array_equal(result1, result2)

    def test_upscale_nearest_확대_픽셀값이_원본과_일치한다(self, rgb_image_hwc):
        """Given: (H, W, 3) 이미지, scale=2
        When:  upscale 호출 후 (0,0) 픽셀 확인
        Then:  확대 이미지의 (0,0), (0,1), (1,0), (1,1)이 원본 (0,0)과 동일

        WHY: nearest 확대의 올바른 구현을 픽셀 레벨에서 검증한다.
        """
        # --- Given ---
        scale = 2
        fake = FakeUpscaleBackend(scale=scale)
        original_pixel = rgb_image_hwc[0, 0]  # shape (3,)

        # --- When ---
        result = fake.upscale(rgb_image_hwc)

        # --- Then ---
        for row in range(scale):
            for col in range(scale):
                np.testing.assert_array_equal(
                    result[row, col],
                    original_pixel,
                    err_msg=f"확대 픽셀 ({row},{col})이 원본 (0,0)과 다름",
                )


# ---------------------------------------------------------------------------
# 3. reconstruction_to_rgb_uint8 순수 함수 단위 테스트
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_NORMALIZE, reason=_MSG_NO_NORMALIZE)
class TestReconstructionToRgbUint8:
    """reconstruction_to_rgb_uint8: (3,H,W) float → (H,W,3) uint8 정규화 검증.

    torch import 없이 numpy ndarray만 사용한다(core 경계 불변식 검증).
    계획서 §3-2 / §7-1 기준.
    """

    def test_출력_shape는_HWC이다(self, chw_float_normal):
        """Given: (3, H, W) float 입력
        When:  reconstruction_to_rgb_uint8 호출
        Then:  shape == (H, W, 3) — CHW → HWC 전치 검증

        WHY: Swin2SR의 CHW float reconstruction을 HWC uint8로 변환하는
             핵심 축 전치가 올바른지 검증한다.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_normal)

        # --- Then ---
        assert result.shape == (_NORM_H, _NORM_W, _NORM_C), (
            f"기대 shape ({_NORM_H}, {_NORM_W}, {_NORM_C}), 실제 {result.shape}"
        )

    def test_출력_dtype은_uint8이다(self, chw_float_normal):
        """Given: float32 CHW 입력
        When:  reconstruction_to_rgb_uint8 호출
        Then:  출력 dtype == uint8

        WHY: save_image / crop_array가 uint8 배열을 기대한다.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_normal)

        # --- Then ---
        assert result.dtype == np.uint8, f"기대 dtype uint8, 실제 {result.dtype}"

    def test_0점0_입력은_0으로_변환된다(self, chw_float_normal):
        """Given: R채널이 전부 0.0인 입력 (chw_float_normal[0] = 0.0)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  결과 R채널(HWC에서 axis=2, index=0)이 전부 0

        WHY: 하한 매핑 0.0→0 정확성 검증.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_normal)

        # --- Then ---
        assert np.all(result[:, :, 0] == 0), (
            f"R채널 0.0→0 변환 실패: {result[:, :, 0]}"
        )

    def test_1점0_입력은_255로_변환된다(self, chw_float_normal):
        """Given: G채널이 전부 1.0인 입력 (chw_float_normal[1] = 1.0)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  결과 G채널(axis=2, index=1)이 전부 255

        WHY: 상한 매핑 1.0→255 정확성 검증.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_normal)

        # --- Then ---
        assert np.all(result[:, :, 1] == _UINT8_MAX), (
            f"G채널 1.0→255 변환 실패: {result[:, :, 1]}"
        )

    def test_0점5_입력은_128로_변환된다(self, chw_float_normal):
        """Given: B채널이 전부 0.5인 입력 (chw_float_normal[2] = 0.5)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  결과 B채널(axis=2, index=2)이 전부 128 (반올림 np.rint)

        WHY: 중간값 매핑 0.5→128 반올림 정확성 검증.
             계획서 §7-1 "0.5→128(반올림)" 명세.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_normal)

        # --- Then ---
        assert np.all(result[:, :, 2] == _HALF_UINT8), (
            f"B채널 0.5→128 변환 실패: {result[:, :, 2]}"
        )

    def test_음수_입력은_클램프되어_0이_된다(self, chw_float_out_of_range):
        """Given: R채널이 전부 -0.5인 입력 (범위 밖)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  결과 R채널이 전부 0 ([0,1] 하한 클램프)

        WHY: Swin2SR reconstruction이 [0,1] 밖으로 약간 벗어날 수 있다(계획서 §3-2).
             클램프가 없으면 uint8 언더플로우로 잘못된 색이 나온다.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_out_of_range)

        # --- Then ---
        assert np.all(result[:, :, 0] == 0), (
            f"음수 클램프 실패: R채널 최솟값 {result[:, :, 0].min()}"
        )

    def test_1초과_입력은_클램프되어_255가_된다(self, chw_float_out_of_range):
        """Given: G채널이 전부 1.5인 입력 (범위 밖)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  결과 G채널이 전부 255 ([0,1] 상한 클램프)

        WHY: 클램프가 없으면 uint8 오버플로우로 잘못된 색이 나온다.
        """
        # --- Given / When ---
        result = reconstruction_to_rgb_uint8(chw_float_out_of_range)

        # --- Then ---
        assert np.all(result[:, :, 1] == _UINT8_MAX), (
            f"1초과 클램프 실패: G채널 최댓값 {result[:, :, 1].max()}"
        )

    def test_채널_순서가_CHW에서_HWC로_올바르게_전치된다(self):
        """Given: C=0채널=1.0, C=1채널=0.0, C=2채널=0.5인 CHW 입력
        When:  reconstruction_to_rgb_uint8 호출
        Then:  HWC 결과의 axis=2 index=0이 255, index=1이 0, index=2가 128

        WHY: 채널 전치(CHW→HWC)가 올바른지 채널별 독립 검증.
             축 0↔2 전치 오류 시 R/G/B가 뒤바뀌어 색이 반전된다.
        """
        # --- Given ---
        # 채널 0: 1.0 전부 → uint8 255
        # 채널 1: 0.0 전부 → uint8 0
        # 채널 2: 0.5 전부 → uint8 128
        arr = np.zeros((_NORM_C, _NORM_H, _NORM_W), dtype=np.float32)
        arr[0] = 1.0
        arr[1] = 0.0
        arr[2] = _HALF_FLOAT

        # --- When ---
        result = reconstruction_to_rgb_uint8(arr)

        # --- Then ---
        assert np.all(result[:, :, 0] == _UINT8_MAX), "CHW[0]=1.0 → HWC[:,0] 255 전치 실패"
        assert np.all(result[:, :, 1] == 0), "CHW[1]=0.0 → HWC[:,1] 0 전치 실패"
        assert np.all(result[:, :, 2] == _HALF_UINT8), "CHW[2]=0.5 → HWC[:,2] 128 전치 실패"

    def test_출력_값_범위는_0에서_255이다(self):
        """Given: 임의의 float 값 포함 CHW 배열 (클램프 필요)
        When:  reconstruction_to_rgb_uint8 호출
        Then:  모든 픽셀값이 0 이상 255 이하

        WHY: 어떤 입력에서도 uint8 유효 범위를 보장하는 방어 테스트.
        """
        # --- Given ---
        rng = np.random.default_rng(seed=42)
        arr = rng.uniform(-1.0, 2.0, (_NORM_C, _NORM_H, _NORM_W)).astype(np.float32)

        # --- When ---
        result = reconstruction_to_rgb_uint8(arr)

        # --- Then ---
        assert result.min() >= 0, f"0 미만 값 존재: min={result.min()}"
        assert result.max() <= _UINT8_MAX, f"255 초과 값 존재: max={result.max()}"

    def test_float64_입력도_uint8을_반환한다(self):
        """Given: float64 dtype CHW 배열
        When:  reconstruction_to_rgb_uint8 호출
        Then:  출력 dtype == uint8 (입력 dtype에 무관)

        WHY: numpy 연산에서 float64 입력이 들어와도 uint8 계약을 유지한다.
        """
        # --- Given ---
        arr = np.full((_NORM_C, _NORM_H, _NORM_W), 0.5, dtype=np.float64)

        # --- When ---
        result = reconstruction_to_rgb_uint8(arr)

        # --- Then ---
        assert result.dtype == np.uint8

    def test_단일_픽셀_입력_shape_3x1x1도_처리된다(self):
        """Given: (3, 1, 1) 최소 크기 CHW 입력
        When:  reconstruction_to_rgb_uint8 호출
        Then:  shape == (1, 1, 3), dtype uint8

        WHY: 경계값(최소 입력 크기) 검증.
        """
        # --- Given ---
        arr = np.array([[[0.0]], [[1.0]], [[0.5]]], dtype=np.float32)

        # --- When ---
        result = reconstruction_to_rgb_uint8(arr)

        # --- Then ---
        assert result.shape == (1, 1, _NORM_C)
        assert result.dtype == np.uint8
