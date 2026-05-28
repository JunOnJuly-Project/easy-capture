"""infra/shot_detect 통합 테스트 (PySceneDetect, 합성 클립).

scenedetect/imageio(ffmpeg) 미설치 시 skip. 실제 장면 전환이 있는 합성 mp4를
생성해 detect_cut_frames가 컷을 감지하는지 검증한다(CPU, 리뷰 [치명적] 회귀 가드).

WHY: scenedetect 0.7은 numpy 프레임 직접 푸시 API가 없어 경로 기반으로 동작한다.
     이 테스트가 없어 깨진 호출(detect_scenes(frame=))이 미검출됐다(리뷰 [치명적]).
"""
from __future__ import annotations

import numpy as np
import pytest

imageio = pytest.importorskip("imageio", reason="imageio 미설치 — 합성 클립 생성 불가")
pytest.importorskip("scenedetect", reason="scenedetect 미설치 — 컷 감지 건너뜀")

from easy_capture.infra.shot_detect import detect_cut_frames

# 합성 클립 상수
_CLIP_H, _CLIP_W = 64, 64
_SHOT_LEN = 15  # 각 장면 프레임 수
_FPS = 10
# 컷은 첫 장면 끝(_SHOT_LEN) 근처에 잡힌다 — 인코딩 오차 허용 범위
_CUT_LOW, _CUT_HIGH = _SHOT_LEN - 5, _SHOT_LEN + 5


def _write_clip(path, shades: list[int]) -> None:
    """각 shade(0~255)로 _SHOT_LEN 프레임씩 이어붙인 합성 mp4를 쓴다.

    shade가 급변하는 지점이 ContentDetector에 컷으로 감지된다.
    """
    frames: list[np.ndarray] = []
    for shade in shades:
        frame = np.full((_CLIP_H, _CLIP_W, 3), shade, dtype=np.uint8)
        frames += [frame] * _SHOT_LEN
    imageio.mimwrite(str(path), frames, fps=_FPS, macro_block_size=1)


class TestDetectCutFrames:
    """detect_cut_frames: 합성 클립의 장면 전환을 컷 인덱스로 반환."""

    def test_단일_컷_클립에서_컷을_감지한다(self, tmp_path):
        """Given: 검정→흰색 2장면 클립
        When:  detect_cut_frames 호출
        Then:  최소 1개 컷, 첫 장면 끝(_SHOT_LEN) 근처
        """
        clip = tmp_path / "two_shot.mp4"
        _write_clip(clip, shades=[0, 255])

        cuts = detect_cut_frames(str(clip))

        assert len(cuts) >= 1, "장면 전환을 컷으로 감지하지 못함"
        assert any(_CUT_LOW <= c <= _CUT_HIGH for c in cuts), (
            f"컷 인덱스 {cuts}가 예상 위치({_CUT_LOW}~{_CUT_HIGH})를 벗어남"
        )

    def test_컷_없는_클립은_빈_리스트를_반환한다(self, tmp_path):
        """Given: 전 구간 동일(컷 없음) 클립
        When:  detect_cut_frames 호출
        Then:  빈 리스트
        """
        clip = tmp_path / "single_shot.mp4"
        _write_clip(clip, shades=[128])

        cuts = detect_cut_frames(str(clip))

        assert cuts == [], f"컷이 없는데 {cuts} 반환"

    def test_컷_인덱스는_오름차순이다(self, tmp_path):
        """Given: 검정→흰색→검정 3장면 클립
        When:  detect_cut_frames 호출
        Then:  컷 인덱스가 오름차순 정렬
        """
        clip = tmp_path / "three_shot.mp4"
        _write_clip(clip, shades=[0, 255, 0])

        cuts = detect_cut_frames(str(clip))

        assert cuts == sorted(cuts), f"컷 인덱스가 정렬되지 않음: {cuts}"
