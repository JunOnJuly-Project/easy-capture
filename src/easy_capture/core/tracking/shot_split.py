"""샷 분할 순수 함수 (계획서 §4-4).

컷 프레임 인덱스 리스트 + 전체 프레임 수 → (start, end) 샷 구간 리스트.

설계 원칙:
  - 순수 함수 — PySceneDetect·torch·transformers 비의존.
  - end는 exclusive (range 관례: frames[start:end]).
  - k컷 → k+1샷, 프레임 손실·중복 없음.

WHY: 샷 분할 로직이 오케스트레이션(app)과 감지(infra) 사이에서
     별도 순수 함수로 분리돼야 compute_boxes 순수성 불변식을 유지할 수 있다.
     gap_policy 인접 위치(core/tracking)에 두어 추적 도메인 응집도를 높인다.
"""
from __future__ import annotations


def split_into_shots(
    n_frames: int,
    cut_frames: list[int],
) -> list[tuple[int, int]]:
    """컷 인덱스 리스트로 프레임 구간을 샷 (start, end) 리스트로 분할한다(순수).

    end는 exclusive (range 관례: frames[start:end]).
    k컷 → k+1샷, 각 샷의 (end - start) 합계 == n_frames.

    Args:
        n_frames:   전체 프레임 수.
        cut_frames: 각 샷이 시작되는 프레임 인덱스 리스트(오름차순, 0 제외).
                    예: [80] → [0,80)·[80,n_frames).

    Returns:
        (start, end) 튜플 리스트. 컷 없으면 [(0, n_frames)].

    WHY: 컷이 k개이면 k+1샷이 생성된다(기본 산술).
         propagate_call_count == 샷 수 카운터 가드와 직접 연결된다.
    """
    boundaries = sorted(cut_frames)
    starts = [0] + boundaries
    ends = boundaries + [n_frames]
    return list(zip(starts, ends))
