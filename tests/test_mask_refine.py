"""마스크 후처리 — largest_component 순수 로직 테스트 (마스크 정제 슬라이스 — RED).

대상 모듈: easy_capture.core.crop.mask_refine (신규 — 구현 전 RED)

배경:
  멀티샷 군무에서 SAM2 마스크가 부정확·과대(대상 배 + 옆사람 팔)했다. box
  프롬프트(detect bbox→SAM2)로 1차 개선하되, 그래도 인접한 다른 사람의 작은
  파편이 마스크에 섞일 수 있다. 해결: 마스크에서 가장 큰 연결성분(largest
  connected component)만 남기고 작은 파편을 제거한다.

검증 대상:
  largest_component(mask: np.ndarray) -> np.ndarray
    - 두 덩어리(큰 주체 + 작은 인접 팔) → 큰 덩어리만 남는다.
    - 단일 덩어리 → 불변.
    - 빈 마스크(all False) → 빈 그대로(예외 없음).
    - 4-연결 기준(대각선만 닿은 두 픽셀은 분리)을 따른다.
    - 반환 dtype/shape는 입력과 동일.

설계 경계 불변식:
  순수 core — numpy만. torch·transformers·PySide6·PyAV·scipy 비의존.
  WHY scipy 금지: scipy.ndimage.label을 쓰면 무거운 의존이 core에 들어온다.
       largest_component는 numpy만으로(BFS/flood fill) 구현 가능하다(KISS).

구현 전 RED 상태가 정상:
  core/crop/mask_refine.py 미존재 → ImportError로 skip(개별 집계).
"""
from __future__ import annotations

import numpy as np
import pytest

# --- mask_refine 미구현 → try/except 격리 ---
# WHY: 구현 전이므로 import 자체가 실패한다. 이 격리로 기존 테스트를
#      차단하지 않고 신규 테스트만 skip/fail로 개별 집계되게 한다.
try:
    from easy_capture.core.crop.mask_refine import largest_component
    _HAS_MASK_REFINE = True
except ImportError:
    largest_component = None  # type: ignore[assignment]
    _HAS_MASK_REFINE = False

_MSG_NO_MASK_REFINE = (
    "core/crop/mask_refine.py에 largest_component 미구현 — RED 예상"
)

# ---------------------------------------------------------------------------
# 테스트 상수 (매직넘버 금지) — 작은 합성 마스크
# ---------------------------------------------------------------------------
# 합성 마스크 크기 (작게 — 10x10)
MASK_H = 10
MASK_W = 10

# 큰 주체 덩어리 영역 (행 슬라이스, 열 슬라이스) — 6x5 = 30픽셀
BIG_ROWS = slice(1, 7)
BIG_COLS = slice(1, 6)
BIG_AREA = 6 * 5  # 30

# 작은 인접 팔 파편 영역 — 2x1 = 2픽셀 (큰 덩어리와 떨어진 우하단)
SMALL_ROWS = slice(8, 10)
SMALL_COLS = slice(8, 9)
SMALL_AREA = 2 * 1  # 2


def _empty_mask() -> np.ndarray:
    """전부 False인 빈 bool 마스크를 반환한다."""
    return np.zeros((MASK_H, MASK_W), dtype=bool)


def _mask_with_two_blobs() -> np.ndarray:
    """큰 주체 덩어리 + 작은 인접 팔 파편(서로 떨어짐) 마스크를 반환한다."""
    mask = _empty_mask()
    mask[BIG_ROWS, BIG_COLS] = True
    mask[SMALL_ROWS, SMALL_COLS] = True
    return mask


# ---------------------------------------------------------------------------
# 순수성(core 경계) 가드 — subprocess 격리 (test_cut_selection 패턴 차용)
# ---------------------------------------------------------------------------
# 순수성 가드 — import 금지 모듈 목록(core 경계 불변식). scipy 포함(numpy만 허용).
_FORBIDDEN_MODULES = ("torch", "transformers", "PySide6", "av", "scipy")


def _mask_refine_keeps_pure(forbidden_module: str) -> bool:
    """격리 서브프로세스에서 mask_refine만 import 후 forbidden 미로드 검증.

    WHY subprocess: 같은 pytest 세션의 UI 테스트(test_video_window_* 등)가 먼저
    PySide6 등을 로드하면 sys.modules에 잔류해 같은 프로세스 검사는 위양성이 난다.
    새 인터프리터로 격리해 mask_refine import만의 부수효과를 검사한다.
    (test_cut_selection.py의 _cut_selection_keeps_pure 패턴 그대로 차용)
    """
    import subprocess
    import sys as _sys

    check_code = (
        "import sys; "
        "from easy_capture.core.crop.mask_refine import largest_component; "
        f"assert not any(k == '{forbidden_module}' or "
        f"k.startswith('{forbidden_module}.') for k in sys.modules), "
        f"'{forbidden_module} 로드됨'"
    )
    result = subprocess.run(
        [_sys.executable, "-c", check_code], capture_output=True, text=True
    )
    return result.returncode == 0


class TestMaskRefinePurity:
    """mask_refine 모듈이 무거운 의존을 끌어오지 않는지 검증(core 경계 불변식)."""

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    @pytest.mark.parametrize("forbidden", _FORBIDDEN_MODULES)
    def test_mask_refine_import_시_무거운_의존을_로드하지_않는다(self, forbidden):
        """Given: 격리 서브프로세스에서 mask_refine만 import
        When:  sys.modules에서 금지 모듈 확인
        Then:  torch·transformers·PySide6·av·scipy 미로드(순수 numpy)

        WHY: core는 순수 도메인 — GPU/UI/IO/무거운 과학연산 라이브러리에
             의존하면 안 된다. largest_component는 numpy만으로 구현 가능하다.
             subprocess 격리로 같은 세션 타 테스트의 잔류 모듈 위양성을 회피한다.
        """
        assert _mask_refine_keeps_pure(forbidden), (
            f"mask_refine이 격리 import에서 금지 모듈 '{forbidden}'을 로드함 — "
            "core 순수성 위반"
        )


# ---------------------------------------------------------------------------
# largest_component — 최대 연결성분만 남긴다
# ---------------------------------------------------------------------------
class TestLargestComponent:
    """largest_component: 마스크에서 가장 큰 연결성분만 남기고 파편을 제거한다."""

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_두_덩어리면_큰_덩어리만_남는다(self):
        """Given: 큰 주체(30px) + 작은 인접 팔 파편(2px) 마스크
        When:  largest_component 호출
        Then:  큰 덩어리 영역만 True, 작은 파편 영역은 False

        WHY: 옆사람 팔 같은 인접 파편이 마스크에 섞이면 크롭이 과대해진다.
             가장 큰 연결성분(=대상 본체)만 남겨 마스크 과대를 해결한다.
        """
        mask = _mask_with_two_blobs()

        result = largest_component(mask)

        # 큰 덩어리는 전부 살아남는다
        assert result[BIG_ROWS, BIG_COLS].all(), "큰 주체 덩어리가 보존되지 않았다"
        # 작은 파편은 전부 제거된다
        assert not result[SMALL_ROWS, SMALL_COLS].any(), (
            "작은 인접 파편이 제거되지 않았다"
        )

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_큰_덩어리만_정확히_남고_나머지는_전부_False이다(self):
        """Given: 큰 덩어리 + 작은 파편 마스크
        When:  largest_component 호출
        Then:  남은 True 픽셀 수 == 큰 덩어리 면적(30)

        WHY: 큰 덩어리만 정확히 남기고 그 외 영역(파편 포함)은 전부 제거해야
             한다 — 면적 합으로 누락·잔존을 동시에 검증한다.
        """
        mask = _mask_with_two_blobs()

        result = largest_component(mask)

        assert int(result.sum()) == BIG_AREA, (
            f"남은 픽셀 수 불일치: {int(result.sum())} vs {BIG_AREA}(큰 덩어리)"
        )

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_단일_덩어리면_불변이다(self):
        """Given: 연결된 덩어리 하나뿐인 마스크
        When:  largest_component 호출
        Then:  입력과 동일한 마스크(파편이 없으니 변화 없음)

        WHY: 제거할 파편이 없으면 마스크를 그대로 두어야 한다 — 정상 마스크를
             손상시키면 안 된다(무회귀 가드).
        """
        mask = _empty_mask()
        mask[BIG_ROWS, BIG_COLS] = True

        result = largest_component(mask)

        assert np.array_equal(result, mask), "단일 덩어리 마스크가 변경되었다"

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_빈_마스크면_예외없이_빈_마스크를_반환한다(self):
        """Given: 전부 False인 빈 마스크
        When:  largest_component 호출
        Then:  예외 없이 전부 False인 마스크 반환

        WHY: occlusion 등으로 마스크가 비어 있을 수 있다. 연결성분이 하나도
             없을 때 IndexError·ValueError 없이 빈 마스크를 그대로 돌려줘야 한다.
        """
        mask = _empty_mask()

        result = largest_component(mask)

        assert not result.any(), "빈 마스크 입력에 True 픽셀이 생겼다"

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_대각선으로만_닿은_두_픽셀은_4연결_기준으로_분리된다(self):
        """Given: (2,2)에 큰 십자 덩어리, 대각선 위치 (0,0)에 단일 픽셀
        When:  largest_component 호출
        Then:  대각선 단일 픽셀은 제거된다(4-연결: 상하좌우만 연결)

        WHY: 4-연결 기준이면 대각선으로만 닿은 픽셀은 별개 성분이다.
             8-연결(대각 포함)로 잘못 구현하면 파편이 본체에 붙어버린다 —
             연결성 기준을 명시적으로 가드한다.
        """
        mask = _empty_mask()
        # 큰 덩어리: (2,2) 중심 3x3 십자/블록 (면적 9)
        mask[2:5, 2:5] = True
        # 대각선으로만 닿은 단일 픽셀 (1,1)은 (2,2)와 4-연결로는 분리.
        # WHY: (1,1)과 (2,2)는 대각 이웃 — 4-연결이면 끊긴다.
        mask[1, 1] = True

        result = largest_component(mask)

        # 큰 블록은 보존
        assert result[2:5, 2:5].all(), "큰 블록이 보존되지 않았다"
        # 대각선 단일 픽셀은 4-연결 분리 → 제거
        assert not bool(result[1, 1]), (
            "대각선으로만 닿은 픽셀이 4-연결인데 제거되지 않았다(8-연결 의심)"
        )

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_반환_shape가_입력과_동일하다(self):
        """Given: (10, 10) 마스크
        When:  largest_component 호출
        Then:  반환 shape == (10, 10)

        WHY: 후처리는 마스크 형태를 보존해야 centroid·bbox 좌표계가 어긋나지
             않는다(크롭 파이프라인 불변식).
        """
        mask = _mask_with_two_blobs()

        result = largest_component(mask)

        assert result.shape == (MASK_H, MASK_W), (
            f"shape 불일치: {result.shape} vs {(MASK_H, MASK_W)}"
        )

    @pytest.mark.skipif(not _HAS_MASK_REFINE, reason=_MSG_NO_MASK_REFINE)
    def test_반환_dtype가_bool이다(self):
        """Given: bool 마스크
        When:  largest_component 호출
        Then:  반환 dtype == bool

        WHY: 후속 centroid_of_mask·bbox_of_mask가 bool 마스크를 기대한다.
             dtype이 int로 바뀌면 하위 계약과 어긋난다.
        """
        mask = _mask_with_two_blobs()

        result = largest_component(mask)

        assert result.dtype == np.bool_, (
            f"dtype 불일치: {result.dtype} vs bool"
        )
