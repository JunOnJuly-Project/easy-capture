"""PySceneDetect 기반 컷 감지 (infra, CPU 검증 가능).

계획서 §2-5 · PoC h2_cut_retrack.py 패턴 계승.

설계 원칙:
  - 지연 import: scenedetect를 함수 내부에서 import(video_io의 av 패턴 계승).
  - 출력 = 순수 데이터: scenedetect 타입을 list[int]로 정규화해 누출 방지.
  - 입력 = 이미 추출된 프레임 리스트: 이중 디코드 방지(DRY).
  - CPU에서 동작 → 합성 클립으로 통합 테스트 가능.

WHY: PySceneDetect ContentDetector는 CPU에서 동작하므로 GPU 블로커 없이
     컷 감지 정확도를 합성 클립으로 검증할 수 있다(ADR 0007 이중 경로 계승).
"""
from __future__ import annotations

import numpy as np


def detect_cut_frames(
    frames: list[np.ndarray],
    threshold: float = 27.0,
) -> list[int]:
    """프레임 시퀀스에서 컷 경계 프레임 인덱스 리스트를 반환한다(순수 데이터 출력).

    PySceneDetect ContentDetector로 장면을 감지한 뒤, 각 장면의 시작 프레임
    인덱스(첫 장면 0 제외)를 "컷 인덱스"로 반환한다.
    예: [0,80) [80,150) → 컷=[80].

    Args:
        frames:    구간 RGB HxWx3 uint8 프레임 리스트(이미 추출된 시퀀스).
        threshold: ContentDetector 민감도(기본 27.0, 클수록 둔감).

    Returns:
        컷이 시작되는 프레임 인덱스 오름차순 리스트. 컷 없으면 [].

    WHY: 지연 import로 scenedetect 미설치 환경에서 다른 기능이 차단되지 않는다.
         프레임 푸시 API(SceneManager + add_frame_to_scene_manager)를 써서
         메모리 프레임 입력을 지원하고 이중 디코드를 방지한다.
    """
    # 지연 import — scenedetect 미설치 시 이 함수 호출 전까지 오류 없음
    from scenedetect import SceneManager
    from scenedetect.detectors import ContentDetector

    if not frames:
        return []

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    h, w = frames[0].shape[:2]
    # PySceneDetect 프레임 푸시 API
    for frame in frames:
        scene_manager.detect_scenes(frame=frame)

    scenes = scene_manager.get_scene_list()
    # 첫 장면 시작(인덱스 0) 제외, 각 장면의 시작 프레임 인덱스 추출
    cut_indices = [scene[0].get_frames() for scene in scenes[1:]]
    return sorted(cut_indices)
