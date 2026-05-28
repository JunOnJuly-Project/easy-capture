"""PySceneDetect 기반 컷 감지 (infra, CPU 검증 가능).

계획서 §2-5 · PoC h2_cut_retrack.py 패턴 계승.

설계 원칙:
  - 지연 import: scenedetect를 함수 내부에서 import(video_io의 av 패턴 계승).
  - 출력 = 순수 데이터: scenedetect 타입을 list[int]로 정규화해 누출 방지.
  - 입력 = video_path + 구간(start_frame, end_frame): PoC h2 detect() 패턴 계승.
  - CPU에서 동작 → 합성 클립으로 통합 테스트 가능.

WHY: scenedetect 0.7에서 numpy 프레임 직접 푸시 공개 API가 없다.
     detect_scenes(frame=...) 키워드는 존재하지 않고
     detect_scenes(video=VideoStream) 방식만 지원한다(리뷰 [치명적] 수정).
     PoC h2_cut_retrack.py의 scenedetect.detect() 경로 기반 호출 방식을 따른다.
"""
from __future__ import annotations

# 구간 상대 인덱스 계산에 사용 (절대 인덱스 → 구간 내 상대 인덱스)
_FIRST_SCENE_START = 0  # 첫 장면 시작 프레임 번호(항상 0)


def detect_cut_frames(
    video_path: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    threshold: float = 27.0,
) -> list[int]:
    """비디오 파일에서 컷 경계 프레임 인덱스 리스트를 반환한다(순수 데이터 출력).

    PySceneDetect ContentDetector로 장면을 감지한 뒤, 각 장면의 시작 프레임
    인덱스(첫 장면 0 제외)를 구간 내 상대 인덱스로 반환한다.
    예: start_frame=0, 80프레임에서 컷 → [80].
        start_frame=50, 130(절대) 프레임에서 컷 → [80] (130-50=80, 상대 인덱스).

    Args:
        video_path:  분석할 비디오 파일 경로.
        start_frame: 분석 시작 프레임(포함, 기본 0).
        end_frame:   분석 종료 프레임(포함, 기본 None=끝까지).
        threshold:   ContentDetector 민감도(기본 27.0, 클수록 둔감).

    Returns:
        컷이 시작되는 구간 내 상대 프레임 인덱스 오름차순 리스트. 컷 없으면 [].

    WHY: scenedetect 0.7은 open_video + SceneManager.detect_scenes(video=stream)
         경로 기반 API만 제공한다. PoC h2_cut_retrack.py의 detect() 패턴 계승.
         출력을 구간 내 상대 인덱스로 정규화해 app 레이어가 절대 프레임 번호를
         알 필요 없게 한다(캡슐화).
    """
    # 지연 import — scenedetect 미설치 시 이 함수 호출 전까지 오류 없음
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    # WHY: scenedetect 0.7 open_video()는 start_time 키워드를 받지 않는다.
    #      시작 지점은 VideoStream.seek(프레임 번호)로 이동한다(리뷰 [치명적]).
    video = open_video(video_path)
    if start_frame > 0:
        video.seek(start_frame)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video=video, end_time=end_frame, show_progress=False)

    scenes = scene_manager.get_scene_list()
    # 첫 장면 시작(start_frame) 제외, 각 장면의 시작 절대 프레임 → 구간 내 상대 인덱스
    cut_absolute = [scene[0].frame_num for scene in scenes[1:]]
    cut_relative = [abs_idx - start_frame for abs_idx in cut_absolute]
    return sorted(cut_relative)
