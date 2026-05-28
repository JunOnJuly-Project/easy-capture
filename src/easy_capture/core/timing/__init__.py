"""core.timing — 타임리맵 순수 로직 패키지.

구간별 가변 재생속도(슬로우모션·패스트포워드)를 순수 함수로 계산한다.
imageio·torch·PySide6·av에 비의존 — numpy·stdlib만 사용.
ADR 0013 참조.

공개 심볼:
  SpeedSegment           — 재생 속도 구간 값객체
  PlaybackSchedule       — 타임리맵 결과 중간 표현
  normalize_segments     — 구간 정렬·검증·겹침 금지
  build_playback_schedule — 핵심 스케줄 생성
  schedule_to_cfr_indices — MP4 CFR 변환
  clamp_durations_for_gif — GIF 10ms 하한 가드
"""
from easy_capture.core.timing.timeremap import (
    PlaybackSchedule,
    SpeedSegment,
    build_playback_schedule,
    clamp_durations_for_gif,
    normalize_segments,
    schedule_to_cfr_indices,
)

__all__ = [
    "SpeedSegment",
    "PlaybackSchedule",
    "normalize_segments",
    "build_playback_schedule",
    "schedule_to_cfr_indices",
    "clamp_durations_for_gif",
]
