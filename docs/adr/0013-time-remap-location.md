# ADR 0013 — 타임리맵 순수 로직 위치 및 PlaybackSchedule 이중 표현

- 상태: 채택
- 날짜: 2026-05-28

## 맥락

[비디오 구간별 가변 재생속도 계획서](../plans/video-speed-remap-plan.md)는 구간별 배속(슬로우모션·패스트포워드)을 "추적·크롭이 끝난 프레임 시퀀스에 대해 각 출력 프레임의 표시 시간 또는 프레임 복제/드롭을 재계산한다"는 책임으로 정의한다. 이 계획서는 영상전문가·PM·덕후 3인 페르소나 검토를 거쳐 1차 조건부 컨펌을 받았다.

구현에 앞서 두 가지 결정이 명문화되어야 한다.

1. **위치 결정**: 타임리맵 순수 로직을 어느 모듈에 둘 것인가. `core/export/video_export.py`([ADR 0011](0011-video-export-location.md))에 함께 두거나, `app/video_capture.py`([ADR 0008](0008-app-usecase-layer.md))에 넣거나, 독립 패키지로 분리할 수 있다.

2. **출력 표현 결정**: GIF는 본질적으로 VFR(프레임별 delay)이라 표시시간을 직접 표현하는 반면, MP4는 CFR(고정 fps)이라 표시시간 대신 프레임 복제/드롭으로 시간을 표현해야 한다. 두 백엔드에서 동시에 지원하기 위해 중간 표현을 어떻게 정의할 것인가.

또한 GIF 인코더(imageio 2.28+)는 per-frame duration이 10ms 미만이면 `delay=0`으로 저장하고, 대부분 뷰어는 `delay=0`을 "지정 없음"으로 해석해 기본 ~100ms로 강제 적용한다. base 30fps × 배속 4x = 8.3ms처럼 빠르게 만들려던 구간이 오히려 느려지는 역전이 발생할 수 있다. 이 가드의 위치와 책임 소재도 결정해야 한다.

## 결정

### 1. 타임리맵 순수 로직을 신규 `core/timing/timeremap.py`에 둔다

`core/timing/` 패키지를 신규 생성하고, 타임리맵 로직 전체를 여기에 위치시킨다.

- 이 로직은 인코딩 라이브러리(imageio·imageio-ffmpeg)에 비의존하며, 추적 갭 정책(`core/tracking/gap_policy.py`)과도 책임이 다르다. "어떤 프레임을 보일 것인가"(gap_policy)와 "선택된 프레임들을 시간축에서 어떻게 늘리고 줄일 것인가"(timeremap)는 직교하는 책임이다.
- numpy·stdlib만 의존하므로 core 비의존 불변식([ADR 0008](0008-app-usecase-layer.md))을 완전히 만족한다. `imageio`는 `encode_frames` 내부 지연 import에만 머문다는 [ADR 0011](0011-video-export-location.md) 결정과 충돌하지 않는다.
- 독립 패키지로 분리함으로써 90%+ 단위 테스트가 imageio·torch 없이 가능하다.

```
src/easy_capture/core/
└── timing/
    ├── __init__.py   # SpeedSegment, PlaybackSchedule, build_playback_schedule,
    │                 # clamp_durations_for_gif, schedule_to_cfr_indices export
    └── timeremap.py  # numpy/stdlib만 의존 — core 경계 완전 준수
```

### 2. `PlaybackSchedule(frame_indices, durations_ms)` 이중 표현으로 GIF·MP4 양쪽을 하나의 스케줄에서 지원한다

```python
@dataclass(frozen=True)
class PlaybackSchedule:
    """타임리맵 결과 중간 표현(불변). GIF·MP4 양쪽 산출 가능.

    frame_indices: 출력할 (타임리맵 입력 기준) 프레임 인덱스 시퀀스.
                   슬로우=복제로 같은 인덱스 반복, 패스트=일부 인덱스 생략.
    durations_ms:  frame_indices와 1:1 대응하는 프레임별 표시시간(ms).
    """
    frame_indices: list[int]
    durations_ms: list[float]
```

- **GIF 경로**: `durations_ms`를 per-frame duration 리스트로 직접 전달한다. 프레임 복제 없이 표시시간만 늘리면 되므로 GIF의 VFR 특성을 자연스럽게 활용한다.
- **MP4 경로**: 고정 fps라 표시시간을 직접 표현할 수 없으므로, `schedule_to_cfr_indices(schedule)` 헬퍼로 슬로우 구간은 같은 인덱스를 정수배 복제하고 패스트 구간은 등간격 드롭해 CFR 시간축 인덱스 시퀀스를 만든다.
- **항등 보장**: `segments == []`이면 `frame_indices = list(range(n_frames))`, `durations_ms = [1000/base_fps] * n_frames`로 기존 단일 fps 경로와 동일한 결과를 낸다.

핵심 함수 시그니처:

```python
def build_playback_schedule(
    n_frames: int,
    segments: list[SpeedSegment],
    base_fps: float,
) -> PlaybackSchedule:
    """프레임 수 + 구간 배속 + 기준 fps → 재생 스케줄(순수).

    WHY 순수 함수: 인코딩 라이브러리·UI 상태에 비의존하므로
    imageio/torch 없이 단위 테스트 가능. 90%+ 커버리지 가드.
    """

def schedule_to_cfr_indices(
    schedule: PlaybackSchedule,
) -> list[int]:
    """MP4 CFR 경로용: 슬로우=프레임 복제, 패스트=등간격 드롭 인덱스 시퀀스."""
```

### 3. 인코딩과 타임리맵의 책임을 분리한다 — `VideoCaptureUseCase.export`가 `build_output_indices` 다음, 인코딩 전에 삽입한다

`app/video_capture.py`의 export 흐름:

```
gap_policy(build_output_indices)          # 어떤 원본 프레임을 보일 것인가
        │
        ▼
[타임리맵: 출력 인덱스 → 재생 스케줄]     # 그 프레임들을 시간축에서 어떻게 배열할 것인가
        │
        ▼
crop_frames → encode_frames(스케줄 반영)  # 공간 크롭 → 인코딩
```

`schedule.frame_indices`는 0..N-1(타임리맵 입력 길이) 기준이므로, 실제 원본 프레임은 `output_indices[schedule.frame_indices[k]]`로 재색인한다(2단계 인덱싱). boxes도 동일하게 재색인한다.

**WHY 2단계 인덱싱**: gap_policy는 "어떤 원본 프레임을 보일 것인가"(occlusion·CUT 정책), 타임리맵은 "그 출력 프레임들을 시간축에서 어떻게 배열할 것인가"를 결정한다. 두 책임이 직교하므로 각자의 인덱스 공간을 유지하고 연결 지점만 명시한다. CUT 정책으로 원본 프레임이 빠지면 출력 인덱스가 달라지는데, 타임리맵이 원본 기준으로 구간을 해석했다면 인덱스가 어긋난다. 출력 프레임 기준으로 구간을 정의함으로써 이 혼동을 원천 차단한다.

`VideoExportConfig`에 `segments` 필드를 추가한다:

```python
@dataclass(frozen=True)
class VideoExportConfig:
    fmt: str = "gif"
    fps: float = 12.0
    gap_policy: GapPolicy = GapPolicy.BACKGROUND
    segments: tuple[SpeedSegment, ...] = ()   # 빈 튜플=등속(무회귀 보장)
```

**WHY 튜플**: frozen dataclass 필드 해시 가능성 보장. `VideoExportConfig` 불변성과 일관된다.

### 4. GIF 10ms 하한 가드를 `core/timing`의 순수 함수 `clamp_durations_for_gif`로 격리한다

```python
def clamp_durations_for_gif(
    schedule: PlaybackSchedule,
) -> tuple[PlaybackSchedule, list[int]]:
    """GIF per-frame duration < 10ms 를 20ms로 클램프(순수).

    WHY 20ms(=50fps 상당): GIF delay는 centisecond 단위로 양자화되며,
    10ms 미만이면 인코더가 delay=0으로 저장하고 대부분 뷰어는 delay=0을
    "지정 없음"으로 해석해 기본 ~100ms로 강제 적용한다.
    빠르게 만들려던 구간이 오히려 느려지는 역전을 방지하기 위해
    10ms 미만을 20ms로 클램프한다.

    Returns:
        (클램프된 스케줄, 클램프 적용된 프레임 인덱스 목록)
        인덱스 목록은 UI/노트북 경고 표시용.
    """
```

- GIF 경로에서만 `build_playback_schedule` 직후, `_encode_gif` 호출 전에 적용한다. 인코더 구현(`_encode_gif`)을 오염시키지 않는다.
- MP4는 프레임 복제/드롭으로 시간을 표현하므로 10ms 양자화 문제와 무관하다. GIF 전용 가드임을 명시한다.
- `배속 0.25~4.0은 절대 한계가 아니라 base_fps 의존 한계`다. GIF는 50fps(=20ms) 이상 빠르게 만들 수 없다. 이 제약을 ADR·계획서·UI 툴팁에 명시한다.

## 대안

**(a) `core/export/video_export.py`에 타임리맵 혼합**

인코딩과 시간 계산을 같은 모듈에 둔다.

- 거부 이유: SRP 위반. 인코딩(부수효과=파일 쓰기)과 시간 계산(순수 변환)의 책임이 한 파일에 혼재된다. `video_export.py`가 비대해지고, 타임리맵 단위 테스트가 imageio 지연 import 경로와 엮인다. [ADR 0011](0011-video-export-location.md)이 imageio를 지연 import로 격리한 취지와 역행한다.

**(b) `app/video_capture.py`에 타임리맵 인라인**

유스케이스 파일 안에서 직접 스케줄을 계산한다.

- 거부 이유: 타임리맵 순수 로직이 오케스트레이션 레이어에 묻혀 재사용이 불가해진다. Colab 노트북·테스트에서 `build_playback_schedule`을 `app` 의존 없이 직접 호출할 수 없어진다. app 레이어를 "흐름 조립" 단일 책임으로 유지한다는 [ADR 0008](0008-app-usecase-layer.md) 원칙과 어긋난다.

**(c) 단일 표현 — duration만 사용**

`PlaybackSchedule`을 `durations_ms`만 갖는 단일 표현으로 정의한다.

- 거부 이유: MP4는 고정 fps라 표시시간을 직접 표현할 수 없다. `schedule_to_cfr_indices`로 변환하는 단계가 어차피 필요하므로, 이중 표현에 추가 비용이 없다. 단일 표현으로는 GIF·MP4 양쪽을 대칭적으로 지원할 수 없다.

**(d) 단일 표현 — 인덱스(복제/드롭) 시퀀스만 사용**

`frame_indices`만 갖는 단일 표현으로 정의한다.

- 거부 이유: GIF는 프레임 복제 없이 표시시간 조정만으로 슬로우/패스트를 표현할 수 있다. 인덱스 복제 방식을 GIF에 강제하면 GIF 파일 크기가 불필요하게 증가하고, GIF의 VFR 특성을 활용하지 못한다. 단일 표현 강제는 GIF 백엔드를 비효율적으로 만든다.

## 결과

### 긍정적 영향

- **90%+ 단위 테스트 가능**: `core/timing/timeremap.py`는 numpy·stdlib만 의존하므로 imageio·torch 설치 없이 `pytest`로 완전 검증 가능하다. 순수 로직의 경계 불변식(항등·배속·클램프·방어)을 결정적으로 테스트할 수 있다.
- **GIF 10ms 역전 방지**: `clamp_durations_for_gif`가 인코딩 전에 하한을 보장한다. "빠르게 만들려다 오히려 느려지는" 현상을 타임리맵 단계에서 차단하고, 클램프 발생 인덱스 목록을 반환해 UI·노트북에서 사용자에게 경고할 수 있다.
- **무회귀 보장**: `VideoExportConfig.segments = ()`(기본 빈 튜플)이면 `build_playback_schedule`이 항등 스케줄을 반환하고 기존 export 경로를 그대로 통과한다. 기존 `test_video_export.py` 전 케이스는 수정 없이 통과해야 한다(Story 1·2·3·4 모든 슬라이스 DoD).
- **GIF·MP4 대칭 지원**: `PlaybackSchedule` 이중 표현 하나로 두 백엔드를 독립적으로 지원한다. 한 백엔드의 요구가 바뀌어도 다른 백엔드와 `build_playback_schedule` 자체는 영향받지 않는다.
- **SOLID 준수**: SRP(시간 계산·인코딩 분리), OCP(`VideoExportConfig.segments` 추가로 기존 경로 불변 확장), DIP(UI·노트북은 `SpeedSegment`/`VideoExportConfig` 추상 데이터만 알고 인코딩 구현 비의존), ISP(`PlaybackSchedule`의 두 필드를 GIF·MP4가 각자 필요한 쪽만 사용).

### 부정적 영향 / 트레이드오프

- **신규 패키지 추가**: `core/timing/` 패키지가 새로 생긴다. ADR 0012의 `detection_backend.py` 추가와 동일한 트레이드오프. 경계가 명확해지는 이점이 파일 수 증가 비용을 상회한다고 판단한다.
- **2단계 인덱싱 복잡도**: `output_indices[schedule.frame_indices[k]]` 재색인 패턴은 한 번만 이해하면 직관적이지만, 처음 마주치는 독자에게 WHY 설명이 없으면 혼동 여지가 있다. `app/video_capture.py`의 해당 코드에 한국어 WHY 주석으로 명시한다.
- **MVP 슬로우 품질 한계**: 프레임 복제 방식은 stutter(끊김)이 보인다. 부드러운 슬로우(RIFE 보간)는 v1.1 후속으로 명시한다. UI의 0.25x 항목 툴팁에 "프레임 복제 방식이라 끊겨 보일 수 있어요(부드러운 슬로우는 v1.1 보간 예정)"를 안내한다.
- **GIF 패스트 한계의 base_fps 의존성**: 배속 0.25~4.0은 절대 한계가 아니라 base_fps에 따라 GIF 50fps(20ms) 장벽에 먼저 닿을 수 있다. 이 제약을 UI 툴팁·노트북 경고 셀·이 ADR에 명시해 사용자 오해를 방지한다.

### Story 1 DoD (core 타임리맵 순수 로직)

본 ADR은 `feature/timing/timeremap-core` 브랜치 머지 전에 작성·채택되어야 한다. 해당 Story의 완료 정의:

- `build_playback_schedule`, `schedule_to_cfr_indices`, `clamp_durations_for_gif` 세 함수 구현 및 `core/timing/__init__.py` export 완료.
- `test_timeremap.py` 단위 테스트 90%+ 통과(항등·단일 구간·다중 구간·경계·배속 정량·GIF 10ms 클램프·방어적 ValueError·MP4 변환·import 경계 불변식).
- 기존 `test_video_export.py` 전 케이스 무회귀(segments=() 경로 불변).
- `core/timing/timeremap.py` import 시 imageio·torch·PySide6 미로드 확인.

### 후속 연계

- [ADR 0011](0011-video-export-location.md) — `core/export/video_export.py`에 `VideoExportConfig.segments` 필드 추가 및 `_encode_gif` per-frame duration 확장이 이 ADR의 결정 3을 구현한다. imageio 지연 import 원칙을 유지한다.
- [ADR 0009](0009-upscale-export-integration.md) — 옵션 백엔드 주입 패턴. `clamp_durations_for_gif` 적용 여부를 GIF 경로에서만 조건부로 삽입하는 방식이 `upscaler=None` 조건부 주입 패턴과 유사하다.
- [ADR 0008](0008-app-usecase-layer.md) — `VideoCaptureUseCase.export`의 2단계 인덱싱 삽입 지점과 `segments=()` 무회귀 요구가 이 ADR의 결정 3에 근거한다.
- [비디오 구간별 가변 재생속도 계획서](../plans/video-speed-remap-plan.md) §3 결정 3·결정 4·§4-2·§5 Story 1: 이 ADR이 계획서의 "신규 ADR 0013 제안" 트리거를 해소한다.
