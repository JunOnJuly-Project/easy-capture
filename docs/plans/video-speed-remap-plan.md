# 비디오 구간별 가변 재생속도(타임리맵) 개발 계획

- 상태: 검토 반영본 (Phase 0 브레인스토밍 + Phase 1 계획서 + 3인 페르소나 1차 조건부 컨펌 반영)
- 날짜: 2026-05-28
- 대상 도메인: `core/timing`(신규) · `core/export` · `app/video_capture` · `ui/video_window` · `poc/colab`
- 도메인 엄격도: L2 (DDD 애그리거트 + 80% 커버리지). 타임리맵 순수 로직은 L3 수준(90%+)으로 가드.
- 선행 ADR: 0008(app usecase layer), 0011(video export location). 본 계획에서 신규 ADR 0013(타임리맵 위치) 제안.

> **검토 반영 이력 (1차, 3인 전원 ⚠️조건부 컨펌·반대 없음):**
> ① [치명적] GIF 패스트 10ms 하한 붕괴 가드 — §3 결정1, §7 리스크 추가.
> ② [치명적·UX] 구간 지정 — 미리보기 프레임→구간 버튼/mm:ss 입력 흡수 — §3 결정5, §5 Story5, §9.
> ③ [중요] 후속 누락 — "하이라이트 트림+슬로우/루프" — §6/§8 로드맵, GIF 자동 루프 강점 명시.
> ④ [중요·PM] 출력 프레임 수 폭증 리스크 + MVP 마일스톤 그룹(알파/베타) + Story1 DoD 보강.
> ⑤ [제안] 배속 프리셋 0.33x 추가·0.25x 툴팁·4x 재평가 — §3 결정1/결정5.

---

## 1. 개요

### 1-1. 요구사항(사용자)

- 특정 프레임 **구간**을 느리게(슬로우모션) 또는 빠르게(패스트포워드) 재생한다.
- **여러 구간을 동시에** 지정할 수 있다.
- 출력은 **GIF·MP4 둘 다** 지원한다.
- **데스크톱(`ui/video_window`)·Colab 노트북** 양쪽에서 구간·배속을 지정한다.

### 1-2. 한 줄 정의

> "추적·크롭이 끝난 프레임 시퀀스에 대해, 사용자가 지정한 `[(시작, 끝, 배속)]` 구간들에 따라 각 출력 프레임의 **표시 시간(GIF) 또는 프레임 복제/드롭(MP4)**을 재계산해 가변 속도 움짤을 만든다."

### 1-3. 위치 — 기존 파이프라인에서의 삽입 지점

현재 `VideoCaptureUseCase.export()`(app/video_capture.py:188)의 흐름:

```
gap_policy(build_output_indices)  →  프레임/박스 선택  →  crop_frames  →  encode_frames
        (출력 인덱스 결정)              (선택)             (순수 크롭)        (GIF/MP4 쓰기)
```

타임리맵은 **`build_output_indices` 다음, `encode_frames` 의 시간 표현 단계**에 삽입한다.

```
gap_policy  →  프레임/박스 선택  →  [타임리맵: 출력 인덱스 → 재생 스케줄]  →  crop_frames  →  encode_frames(스케줄 반영)
```

**WHY (developer 관점, 결합 순서):** `gap_policy`는 "어떤 원본 프레임을 보일 것인가"(occlusion 갭 처리)를 결정하고, 타임리맵은 "그렇게 선택된 출력 프레임들을 시간축에서 어떻게 늘리고 줄일 것인가"를 결정한다. 두 책임은 직교한다. 타임리맵은 `build_output_indices`가 만든 출력 인덱스 시퀀스(길이 N)를 입력으로 받으므로, 구간 인덱스는 **출력 프레임 기준**으로 해석된다(원본 프레임 기준이 아님 — CUT 정책으로 프레임이 빠지면 인덱스가 어긋나기 때문).

---

## 2. 다관점 브레인스토밍 요약 (델파이 압축)

각 관점의 핵심 주장과 합의 결과를 계획에 반영한다.

### developer (실현 가능성·순수 로직)
- 타임리맵을 **순수 함수**로 분리: 입력 = `n_frames + segments + base_fps`, 출력 = **재생 스케줄**(중간 표현). 인코딩 라이브러리·UI에 비의존.
- 중간 표현은 GIF·MP4 양쪽을 모두 산출할 수 있어야 한다 → **출력 프레임 인덱스 시퀀스 + 프레임별 표시시간(ms)** 두 가지를 동시에 담는 dataclass `PlaybackSchedule`.
  - GIF: `imageio` per-frame `duration` 리스트로 직접 매핑(프레임 복제 불필요).
  - MP4: 고정 fps라 표시시간을 표현 못 함 → **프레임 인덱스 복제(슬로우)/드롭(패스트)** 으로 변환.
- 기존 `encode_frames`/`VideoExportConfig` 확장 방식 채택. 무회귀: segments가 비면 기존 단일 fps 경로 그대로.
- 결합 순서: `build_output_indices`(출력 인덱스) → 타임리맵(스케줄). 합의됨(§1-3).

### tester (검증 가능성)
- 순수 로직 가드: 구간 없음=항등, 단일/다중 구간, 겹침·경계, 배속 0.5x/2x, 프레임 복제 수 검증.
- Fake/합성 프레임으로 GIF/MP4 라운드트립(기존 `test_video_export.py` 패턴 계승).
- 배속 표현 단위 테스트(배속 ↔ 표시시간 변환 정확성).
- **타임리맵 순수 함수를 인코딩과 분리**해야 의존성 없이 단위 테스트 가능(developer 안과 합치).

### reviewer (클린코드·경계)
- 함수 20줄·매개변수 3개·`frozen dataclass` 준수. 구간 모델은 `SpeedSegment` dataclass.
- core 경계 불변식 유지: 타임리맵은 numpy/stdlib만. `imageio`는 `encode_frames` 내부 지연 import 유지(ADR 0011).
- **무회귀 가드 필수**: segments 빈 리스트 → 기존 단일 fps 경로와 바이트 동일하게.
- 방어: 배속 음수/0/과대(예 100x) → `ValueError`(한국어). 구간 범위 초과·역전(start≥end) 검증.

### 영상전문가 (품질·타이밍)
- 슬로우모션 품질: **단순 프레임 복제(stutter)** 는 MVP, **프레임 보간(optical flow/RIFE)** 은 v1.1.
- VFR vs CFR: GIF는 본질적으로 VFR(per-frame delay) → 표시시간 직접 표현이 자연스럽다. MP4는 CFR 권장(플레이어 호환) → 프레임 복제로 CFR 유지.
- 오디오 동기: 슬로우 시 영상 길이가 늘어나 오디오와 어긋난다. **현재 MVP는 무음 움짤**(ADR 0011: `VideoExportConfig`는 무음 기준)이므로 **이번 범위 밖**. 오디오 동기는 PoC H4 자산 위에 별도 후속.
- **GIF 패스트 10ms 하한(검토 [치명적]):** GIF delay는 centisecond(10ms) 단위로 양자화될 뿐 아니라, per-frame duration이 10ms 미만이면 PIL/GIF 인코더가 `delay=0`으로 저장한다. 대부분 뷰어는 `delay=0`을 "지정 없음"으로 보고 기본값(~100ms)으로 clamp → **빠르게 만들려던 구간이 오히려 느려지는 역전**이 발생한다. base 30fps×4x → 8.3ms → 0 으로 붕괴. 가드 필수(§3 결정1).

### PM (범위·scope creep 방지)
- MVP = **프레임 복제 기반** 정수/단순 배속(0.25x~4.0x). 보간·오디오·타임라인 GUI는 후속.
- 슬라이스 순서 고정: core 순수 로직 → GIF 통합 → MP4 통합 → UI → 노트북. 각 슬라이스 독립 머지 가능.
- 무회귀를 모든 슬라이스의 DoD(완료 정의)에 명시.
- **마일스톤 그룹(검토 [중요]):** 알파(노트북에서 데모 가능한 최소) = S1+S2+S4. 베타(데스크톱·MP4·재현) = S3+S5+S6. §5에 라벨.
- **출력 프레임 수 폭증(검토 [중요]):** 다중 슬로우 구간 → MP4 복제 프레임이 정수배로 증가 → 인코딩 시간·용량 폭증. 사전 계산·임계 경고 필요(§7).

### 덕후 (실사용 UX)
- 직캠 "최애 결정적 순간"(점프·턴·윙크)을 슬로우로. 다중 구간(예: 인트로 패스트 → 하이라이트 슬로우) 동시 지정이 핵심.
- **구간 지정 UX(검토 [치명적·UX]):** 프레임 번호 단독 SpinBox로는 "최애 파트가 몇 번 프레임인지"를 알 수 없어 결정적 순간 찾기에 사실상 부적합 → MVP 가치 훼손. 데스크톱에 이미 있는 **프레임 미리보기**를 활용해 "현재 미리보기 프레임 → 구간 시작/끝" 버튼 또는 **mm:ss 입력**을 최소 1개는 MVP에 흡수해야 한다.
- **트림+슬로우+루프(검토 [중요]):** 덕후 1순위 시나리오는 "하이라이트 구간만 잘라내(트림) 슬로우로 만들고 무한 루프"다. 현재 계획은 전체-영상 리맵만 다룸 → 트림은 후속 로드맵에 명시. GIF `loop=0`(기존 `_encode_gif`가 이미 적용 — video_export.py:126) 자동 무한루프는 기존 강점으로 부각.
- 노트북은 파이썬 리스트 변수로 충분(덕후 중 코딩 가능층 + 재현성).

---

## 3. 핵심 결정사항 (1~6)

### 결정 1 — 배속 표현: **배속(factor)** 채택 + **GIF 패스트 10ms 하한 가드**

| 안 | 장점 | 단점 |
|---|---|---|
| **배속 factor** (0.5=슬로우, 2.0=패스트) | 직관적("절반 속도"), 구간별 독립, base_fps와 분리 | 목표 fps를 별도 계산해야 |
| 목표 fps | 인코더 직접 매핑 | 구간마다 base 대비 의미 모호, UX 직관성 낮음 |

**결정:** 배속 `factor` 채택. `factor < 1.0` = 슬로우(표시시간 증가/프레임 복제), `factor > 1.0` = 패스트(표시시간 감소/프레임 드롭), `factor == 1.0` = 등속.
- 표시시간 변환: `frame_duration_ms = (1000 / base_fps) / factor`.
- MVP 배속 범위: `0.25 ≤ factor ≤ 4.0`. 범위 밖은 `ValueError`(한국어). (영상전문가: 과대 배속 시 품질 붕괴 방어, reviewer: 0/음수 방어.)

**★ GIF 패스트 10ms 하한 가드(검토 [치명적] 반영):**
배속 범위(0.25~4.0)는 **절대 한계가 아니라 base_fps에 의존**한다. GIF는 per-frame duration이 10ms 미만이면 인코더가 `delay=0`으로 저장 → 뷰어가 기본 ~100ms로 clamp → **빠르게 만들려다 오히려 느려지는 역전**이 발생한다 (예: base 30fps × 4x = 8.3ms → 0). 이를 막기 위해:

1. **(필수) duration 클램프 + 경고:** GIF 경로에서 `(1000 / base_fps) / factor`로 산출한 표시시간이 10ms 미만이면 **20ms(=50fps 상당)로 클램프**하고 한국어 경고를 표시한다("이 배속은 GIF 한계(50fps)를 넘어 효과가 제한됩니다"). 클램프는 `_encode_gif`가 아니라 **타임리맵/스케줄 단계에서 GIF 대상일 때** 적용해(durations_ms 산출 직후) 인코더 구현과 분리한다.
2. **(보강) 동적 패스트 상한:** UI/노트북에서 선택 가능한 최대 factor를 `min(4.0, base_fps × 0.02)` 식으로 동적 산출 가능(예 12fps면 패스트 상한이 낮아짐). MVP는 우선 (1) 클램프+경고로 안전을 보장하고, 동적 상한 노출은 UI 슬라이스(S5)에서 base_fps를 아는 시점에 적용한다.
3. **명시:** "배속 0.25~4.0은 절대값이 아니라 base_fps 의존 한계이며, GIF는 50fps(=20ms) 이상 빠르게 만들 수 없다"를 결정·문서·UI 툴팁에 기재.

> MP4는 고정 fps라 이 문제에서 자유롭다(패스트는 프레임 드롭으로 표현, 표시시간 양자화 없음). 따라서 10ms 하한은 **GIF 전용 가드**다.

### 결정 2 — 구간 모델: `SpeedSegment(start, end, factor)` frozen dataclass

```
@dataclass(frozen=True)
class SpeedSegment:
    start: int      # 출력 프레임 기준 시작 인덱스 (포함)
    end: int        # 출력 프레임 기준 끝 인덱스 (배타 — [start, end))
    factor: float   # 배속. 0.25~4.0
```

**검증·정규화 규칙(순수 함수 `normalize_segments`):**
- `start < end` (역전·빈 구간 금지) → 위반 시 `ValueError`.
- `0.25 ≤ factor ≤ 4.0` → 위반 시 `ValueError`.
- 인덱스 범위 `0 ≤ start`, `end ≤ n_frames` → 초과 시 클램프 또는 `ValueError`(결정: 클램프 + 경고).
- **겹침 정책(MVP): 겹침 금지.** start 기준 정렬 후 인접 구간이 겹치면 `ValueError`("구간이 겹칩니다"). (KISS — 겹침 우선순위 규칙은 복잡도만 늘림. 후속에서 "뒤 구간 우선" 등 도입.)
- 지정되지 않은 구간(gap)은 자동으로 `factor=1.0`(등속) 처리.

**WHY:** `frozen=True`로 불변(기존 `VideoCropParams`·`VideoExportConfig` 패턴 계승). 정렬·검증을 별도 순수 함수로 분리해 매개변수 3개·함수 20줄 규칙 충족.

### 결정 3 — 타임리맵 순수 함수 위치/시그니처: **신규 `core/timing/timeremap.py`**

**위치 결정(ADR 0013 제안):** `core/timing/timeremap.py` 신규 모듈.
- `core/export`(인코딩)도 `core/tracking`(추적 갭)도 아닌 **독립 시간 도메인**. 타임리맵은 인코딩 직전 단계지만 인코딩 라이브러리에 비의존이고, gap_policy(추적 유효성)와도 책임이 다르다.
- 대안 (a) `core/export/video_export.py`에 추가 → export 모듈 비대화·SRP 위반. (b) `core/tracking/gap_policy.py` 옆 → 추적 도메인과 시간 도메인 혼재. → **신규 패키지가 경계 명확.**

**핵심 시그니처(순수, imageio 비의존):**

```
@dataclass(frozen=True)
class PlaybackSchedule:
    """타임리맵 결과 중간 표현(불변). GIF·MP4 양쪽 산출 가능.

    frame_indices: 출력할 (타임리맵 입력 기준) 프레임 인덱스 시퀀스.
                   슬로우=복제로 같은 인덱스 반복, 패스트=일부 인덱스 생략.
    durations_ms:  frame_indices와 1:1 대응하는 프레임별 표시시간(ms).
    """
    frame_indices: list[int]
    durations_ms: list[float]

def build_playback_schedule(
    n_frames: int,
    segments: list[SpeedSegment],
    base_fps: float,
) -> PlaybackSchedule:
    """프레임 수 + 구간 배속 + 기준 fps → 재생 스케줄(순수)."""

def clamp_durations_for_gif(
    schedule: PlaybackSchedule,
) -> tuple[PlaybackSchedule, list[int]]:
    """GIF 10ms 하한 가드(결정1). durations_ms<10 을 20ms로 클램프하고,
    클램프된 프레임 인덱스 목록을 함께 반환(UI 경고용). 순수."""
```

- **이중 표현이 핵심:** `durations_ms`(GIF 직접 사용) + `frame_indices`(MP4 복제/드롭 변환용)를 동시 산출.
  - GIF 경로: `frame_indices`로 프레임을 선택(복제 없이 원본 길이 유지 가능) + `durations_ms`를 per-frame duration으로 전달. **GIF 대상일 때만 `clamp_durations_for_gif`를 적용**해 10ms 붕괴를 방어.
  - MP4 경로: 고정 fps라 duration 표현 불가 → `frame_indices`에서 슬로우 구간은 프레임을 정수배 복제, 패스트 구간은 등간격 드롭해 **CFR 시간축**으로 변환(`schedule_to_cfr_indices` 헬퍼).
- 항등 보장: `segments == []` → `frame_indices = range(n_frames)`, `durations_ms = [1000/base_fps]*n_frames` → 기존 단일 fps 경로와 동일.

### 결정 4 — GIF/MP4 구현: `encode_frames` 확장 (무회귀 우선)

`VideoExportConfig`에 `segments` 필드 추가(기본 빈 튜플 → 기존 경로 보존):

```
@dataclass(frozen=True)
class VideoExportConfig:
    fmt: str = "gif"
    fps: float = 12.0
    gap_policy: GapPolicy = GapPolicy.BACKGROUND
    segments: tuple[SpeedSegment, ...] = ()   # 신규 — 빈 튜플=등속(무회귀)
```

**WHY 튜플:** frozen dataclass 필드는 해시 가능해야 안전 → list 대신 tuple(`VideoExportConfig` 불변성 보장).

- `_encode_gif`: 기존 단일 `duration` → **per-frame `duration` 리스트** 지원으로 확장. `imageio.get_writer(duration=[...])` 또는 프레임별 `append_data` + meta. (기존 단일 fps 경로는 `segments=()`일 때 그대로 유지.) **`loop=0`(무한 루프) 유지** — 덕후 강점(움짤 자동 반복).
- `_encode_mp4`: `schedule_to_cfr_indices`로 복제/드롭된 인덱스 시퀀스를 받아 고정 fps로 인코딩. (macro_block_size=1 유지.)
- `crop_frames`는 **타임리맵으로 확정된 인덱스 시퀀스**로 크롭하므로, 슬로우 구간 복제 프레임은 동일 크롭이 반복된다(추가 크롭 비용 없음 — 인덱스 재사용).

**무회귀 가드(reviewer):** `segments=()`인 기존 `test_video_export.py` 전 케이스가 그대로 통과해야 한다(DoD).

### 결정 5 — UI: 데스크톱 다중 구간 테이블 + **미리보기 프레임 캡처 버튼** + 노트북 리스트 변수

**데스크톱(`ui/video_window.py`):**
- MVP: **구간 테이블 위젯**(`QTableWidget`) — 행마다 `[시작 프레임 SpinBox | 끝 프레임 SpinBox | 배속 ComboBox]`. "구간 추가"/"구간 삭제" 버튼.
- **★ 구간 지정 UX 보강(검토 [치명적·UX] 반영):** 프레임 번호 단독 입력은 "최애 파트가 몇 번 프레임인지" 알 수 없어 사실상 부적합 → 데스크톱에 이미 있는 **프레임 미리보기**(`FrameCanvas` + 구간 SpinBox로 첫 프레임 표시)를 활용해 **"현재 미리보기 → 구간 시작"·"현재 미리보기 → 구간 끝" 버튼**을 MVP에 흡수한다. 현재 보이는 프레임 인덱스를 선택 행의 start/end에 채운다. (대안/병행: SpinBox에 mm:ss 보조 표기 — base_fps로 환산. 최소 1개는 MVP 필수.)
- **배속 ComboBox 프리셋(검토 [제안] 반영):** `0.25x, 0.33x, 0.5x, 1x, 2x` 를 기본 노출.
  - `0.33x` 추가(0.5↔0.25 사이 단계 보강).
  - `0.25x` 항목에 툴팁: "프레임 복제 방식이라 끊겨 보일 수 있어요(부드러운 슬로우는 v1.1 보간 예정)".
  - `4x`는 직캠 실사용 빈도가 낮아(주류는 2x) **기본 프리셋에서 제외**, "직접 입력" 또는 고급 옵션으로만 노출. GIF에서는 결정1 10ms 하한에 걸리기 쉬움도 함께 안내.
- export 시점에 테이블 → `tuple[SpeedSegment, ...]` 변환 → `VideoExportConfig(segments=...)`.
- 검증 실패(겹침·범위·역전)는 한국어 `QMessageBox`로 안내(저장 차단). GIF 10ms 클램프 발생 시 상태바/메시지로 경고.
- 기존 `_on_export` 흐름에 segments 수집 1단계만 추가(최소 침습).

**노트북(`poc/colab/easy_capture_app_verify.ipynb`):**
- 파이썬 리스트 변수로 지정:
  ```python
  segments = [SpeedSegment(start=30, end=60, factor=0.5),   # 하이라이트 슬로우
              SpeedSegment(start=120, end=150, factor=2.0)]  # 지루한 구간 패스트
  config = VideoExportConfig(fmt="gif", fps=meta.fps, segments=tuple(segments))
  ```
- 미리보기 셀: 스케줄 길이·구간 요약 + **예상 출력 프레임 수·GIF 10ms 클램프 경고**를 print(재현성·폭증 사전 인지).

### 결정 6 — 범위: MVP vs 후속

| 항목 | MVP (이번) | 후속 |
|---|---|---|
| 슬로우 구현 | 프레임 복제(stutter) | 프레임 보간(RIFE/optical flow) — v1.1 |
| 배속 | 0.25~4.0 단순 배속(GIF는 base_fps 의존 패스트 한계) | 임의 연속 배속·ramp(가속/감속 곡선) |
| 오디오 | 무음(범위 밖) | 슬로우 시 오디오 타임스트레치(PoC H4 위에) |
| 구간 겹침 | 금지(ValueError) | 우선순위 규칙·중첩 허용 |
| 트림(구간만 추출) | 전체-영상 리맵만 | **하이라이트 구간만 잘라내기(트림) + 슬로우 + 루프** |
| UI | SpinBox 테이블 + 미리보기→구간 버튼 | 미리보기 타임라인 스크럽·드래그 구간 |
| 출력 | GIF(VFR, loop=0)·MP4(CFR 복제) | MP4 진짜 VFR·webm |

---

## 4. 아키텍처

### 4-1. 디렉토리 구조 (신규/변경)

```
src/easy_capture/core/
├── timing/                      # ★ 신규 패키지
│   ├── __init__.py              # SpeedSegment, PlaybackSchedule, build_playback_schedule, clamp_durations_for_gif, schedule_to_cfr_indices export
│   └── timeremap.py             # ★ 타임리맵 순수 로직 (numpy/stdlib only)
├── export/
│   └── video_export.py          # ◇ VideoExportConfig.segments 추가, _encode_gif(per-frame duration)/_encode_mp4(복제) 확장
src/easy_capture/app/
└── video_capture.py             # ◇ export(): 타임리맵 단계 삽입 + GIF 클램프
src/easy_capture/ui/
└── video_window.py              # ◇ 구간 테이블 + 미리보기→구간 버튼 + segments 수집
poc/colab/
└── easy_capture_app_verify.ipynb # ◇ segments 변수 + 미리보기/폭증·클램프 경고 셀
tests/
├── test_timeremap.py            # ★ 신규 — 순수 로직 가드(90%+) + GIF 클램프
├── test_video_export.py         # ◇ per-frame duration·복제 라운드트립 추가
└── test_video_capture.py        # ◇ export 타임리맵 결합 + 무회귀
docs/adr/
└── 0013-time-remap-location.md  # ★ 신규 ADR — 타임리맵 위치 결정 (Story1 머지 전 작성)
```

### 4-2. 데이터 흐름 (export 내부)

```
TrackResult.centroids ──► valid_flags
                              │
                              ▼
              build_output_indices(gap_policy)  ──► output_indices (길이 N)
                              │
   segments(출력 프레임 기준) ─┤
                              ▼
       build_playback_schedule(N, segments, fps) ──► PlaybackSchedule
                              │                         ├─ frame_indices (output_indices에 대한 재색인)
                              │                         └─ durations_ms
              ┌───────────────┴───────────────┐
         GIF 경로                          MP4 경로
   schedule = clamp_durations_for_gif(s)    cfr_idx = schedule_to_cfr_indices(schedule)
   selected = [output[i] for i in           selected = [output[i] for i in cfr_idx]
              schedule.frame_indices]        crops = crop_frames(selected, boxes')
   crops = crop_frames(selected, boxes')     _encode_mp4(crops, fps)
   _encode_gif(crops, durations_ms, loop=0)
```

**WHY 재색인:** `schedule.frame_indices`는 0..N-1(타임리맵 입력 길이) 기준이고, 실제 원본 프레임은 `output_indices[그 값]`이다. 즉 `원본 = output_indices[schedule.frame_indices[k]]`. boxes도 동일 재색인. 이 2단계 인덱싱으로 gap_policy와 타임리맵의 책임이 깔끔히 분리된다.

**WHY GIF만 클램프:** `clamp_durations_for_gif`는 GIF 경로에만 삽입한다. MP4는 프레임 복제/드롭으로 시간을 표현하므로 10ms 양자화 문제가 없다(결정1).

### 4-3. SOLID 적용

- **SRP:** 타임리맵(시간) ↔ gap_policy(추적 갭) ↔ encode(인코딩) ↔ crop(공간) 각자 단일 책임. GIF 클램프도 타임리맵 도메인의 순수 함수로 분리(인코더 비오염).
- **OCP:** `VideoExportConfig.segments` 추가로 export 분기 확장(기존 시그니처·동작 불변).
- **DIP:** UI·노트북은 `SpeedSegment`/`VideoExportConfig`(추상 데이터)만 알고 인코딩 구현 비의존.
- **ISP:** `PlaybackSchedule`은 GIF·MP4가 각자 필요한 필드만 사용(durations_ms / frame_indices).

---

## 5. 작업 분해 (Epic → Story → Task)

슬라이스 순서: **core 로직 → GIF → MP4 → UI → 노트북**. 각 슬라이스는 독립 머지 가능하고, 앞 슬라이스 없이 뒤 슬라이스 불가(의존 단방향).

**마일스톤 그룹(검토 [중요·PM] 반영):**
- **알파(노트북 최소 데모):** S1 + S2 + S4 — core 로직 + GIF 가변속도 + export 결합. 노트북에서 다중 구간 GIF 생성·검증 가능한 최소 출시 단위.
- **베타(데스크톱·MP4·재현):** S3 + S5 + S6 — MP4 복제 + 데스크톱 UI(미리보기→구간) + 노트북 통합. 일반 사용자 대상 완성.

### Epic: 비디오 구간별 가변 재생속도(타임리맵)

#### Story 1 — core 타임리맵 순수 로직 [브랜치 `feature/timing/timeremap-core`] · 우선순위: 높음 · **[알파]**
- [ ] Task 1-1: `SpeedSegment` frozen dataclass + `normalize_segments`(정렬·검증·겹침 금지·범위 클램프) — 순수
- [ ] Task 1-2: `PlaybackSchedule` frozen dataclass 정의
- [ ] Task 1-3: `build_playback_schedule(n_frames, segments, base_fps)` 구현 — 항등·배속 표시시간 산출
- [ ] Task 1-4: `schedule_to_cfr_indices(schedule)` — MP4용 복제/드롭 인덱스 변환
- [ ] Task 1-5: `clamp_durations_for_gif(schedule)` — GIF 10ms 하한 가드(결정1) — 순수
- [ ] Task 1-6: `core/timing/__init__.py` export + 한국어 docstring
- [ ] Task 1-7: 단위 테스트(`test_timeremap.py`) 90%+ — 수용 기준 §6-1
- 수용 기준: segments=[] → 항등(frame_indices=range, durations 균일). 0.5x → 표시시간 2배, 2.0x → 절반. 다중·경계·겹침 ValueError. GIF 10ms 클램프(base30×4x→8.3ms→20ms로 클램프+클램프 인덱스 보고). core 경계(imageio/torch 미import) 검증.
- **DoD: 신규 테스트 통과(90%+) + 기존 테스트 무회귀 + `schedule_to_cfr_indices`·`clamp_durations_for_gif` 포함 + ADR 0013 작성(머지 전 필수).**

#### Story 2 — GIF per-frame duration 통합 [브랜치 `feature/export/gif-variable-duration`] · 우선순위: 높음 · **[알파]**
- [ ] Task 2-1: `VideoExportConfig.segments: tuple[SpeedSegment, ...] = ()` 추가
- [ ] Task 2-2: `_encode_gif`를 per-frame duration 리스트 지원으로 확장(단일 fps 경로·`loop=0` 유지)
- [ ] Task 2-3: `encode_frames`에서 segments 유무 분기(빈 튜플=기존 경로)
- [ ] Task 2-4: GIF 라운드트립 테스트 — 슬로우 구간 프레임 duration 2배 + 10ms 클램프 회귀 가드
- 수용 기준: segments=() GIF는 기존과 바이트/duration 동일. 슬로우 구간 duration ≈ 기존×(1/factor) ±centisecond 오차. 패스트로 10ms 미만 산출 시 20ms 클램프 적용·역전(느려짐) 없음.
- DoD: 기존 `test_video_export.py` 전 케이스 무회귀.

#### Story 3 — MP4 프레임 복제/드롭 통합 [브랜치 `feature/export/mp4-frame-replication`] · 우선순위: 중간 · **[베타]**
- [ ] Task 3-1: `_encode_mp4`가 `schedule_to_cfr_indices` 결과 인덱스 시퀀스를 받도록 확장
- [ ] Task 3-2: 슬로우=프레임 복제, 패스트=드롭으로 CFR 시간축 구성
- [ ] Task 3-3: MP4 라운드트립 테스트(ffmpeg 조건부) — 슬로우 구간 프레임 수 증가 검증
- 수용 기준: 0.5x 구간 프레임 수 ≈ 원래×2(복제), 2.0x ≈ ×0.5(드롭). segments=() 무회귀.
- DoD: imageio-ffmpeg 미설치 시 importorskip, 설치 시 라운드트립 통과.

#### Story 4 — export 결합 (app) [브랜치 `feature/app/export-timeremap`] · 우선순위: 높음 · **[알파]**
- [ ] Task 4-1: `VideoCaptureUseCase.export()`에 타임리맵 단계 삽입(output_indices → schedule → 재색인)
- [ ] Task 4-2: GIF/MP4 경로 분기 + boxes 재색인 + GIF일 때 `clamp_durations_for_gif`
- [ ] Task 4-3: 예상 출력 프레임 수 사전 계산 헬퍼(폭증 경고용, §7) — 순수
- [ ] Task 4-4: 결합 테스트(`test_video_capture.py`) — Fake로 segments 적용 end-to-end + 무회귀
- 수용 기준: segments=() export 결과 기존과 동일. segments 지정 시 GIF duration·MP4 프레임 수 변화 확인. CUT 정책+segments 동시 적용 시 2단계 인덱싱 정확.
- DoD: `export()` 함수 20줄 규칙 유지(헬퍼 분리), 기존 export 테스트 무회귀.

#### Story 5 — 데스크톱 UI 구간 테이블 + 미리보기→구간 [브랜치 `feature/ui/speed-segment-table`] · 우선순위: 중간 · **[베타]**
- [ ] Task 5-1: `QTableWidget` 구간 테이블(시작/끝 SpinBox + 배속 ComboBox 프리셋 0.25/0.33/0.5/1/2x) + 추가/삭제 버튼
- [ ] Task 5-2: **미리보기→구간 버튼**("현재 프레임 → 시작/끝") — 현재 미리보기 프레임 인덱스를 선택 행에 주입 (검토 [치명적·UX])
- [ ] Task 5-3: 0.25x 툴팁·4x 고급 옵션화·GIF 패스트 한계 안내 + 동적 패스트 상한(base_fps 의존)
- [ ] Task 5-4: export 시 테이블 → `tuple[SpeedSegment, ...]` 변환 + 검증 + GIF 클램프 경고 표시
- [ ] Task 5-5: 검증 실패 한국어 QMessageBox(겹침·범위·역전)
- [ ] Task 5-6: 테이블→segments 변환 순수 헬퍼 단위 테스트(QWidget 비의존 로직 분리)
- 수용 기준: 다중 구간 입력 → 저장 시 반영. **미리보기 프레임을 보고 버튼으로 구간 지정 가능(프레임 번호 암기 불필요).** 겹침 입력 시 한국어 안내·저장 차단. GIF 패스트 한계 시 경고.
- DoD: 기존 video_window 테스트 무회귀, 변환 로직은 위젯과 분리해 테스트 가능.

#### Story 6 — Colab 노트북 통합 [브랜치 `feature/poc/notebook-timeremap`] · 우선순위: 낮음 · **[베타]**
- [ ] Task 6-1: `segments` 리스트 변수 셀 + `VideoExportConfig(segments=...)` 적용
- [ ] Task 6-2: 스케줄 요약 미리보기 셀(구간·총 길이 + 예상 출력 프레임 수 + GIF 10ms 클램프 경고 print)
- [ ] Task 6-3: GIF·MP4 양쪽 출력 셀 + 검증 안내(한국어)
- 수용 기준: 노트북에서 다중 구간 GIF·MP4 생성 성공. 데스크톱과 동일 core 함수 사용(재현성).

---

## 6. 테스트 전략

### 6-1. 타임리맵 순수 로직 (`test_timeremap.py`, 90%+)
- **항등:** `segments=[]` → `frame_indices == list(range(n))`, `durations_ms` 균일(`1000/fps`).
- **단일 구간:** `[(2,4,0.5)]` → 구간 표시시간 2배, 외부 등속.
- **다중 구간:** `[(0,2,2.0),(4,6,0.5)]` → 각 구간 독립 적용, gap은 등속.
- **경계:** `start=0`, `end=n_frames`, 인접 구간 경계 정확성.
- **배속 0.5/2.0:** 표시시간·복제수 정량 검증.
- **GIF 10ms 클램프:** base 30fps × 4x = 8.3ms → 20ms 클램프 + 클램프 인덱스 목록 정확. 클램프 후 역전(느려짐) 없음(모든 duration ≥ 20ms).
- **방어:** factor 0/음수/과대(>4)·범위 초과·역전(start≥end)·겹침 → `ValueError`(한국어 메시지 토큰 검증).
- **MP4 변환:** `schedule_to_cfr_indices` — 슬로우 복제수·패스트 드롭수 정량.
- **출력 프레임 수 예측:** 예상 출력 프레임 수 계산 헬퍼가 실제 `schedule_to_cfr_indices` 길이와 일치.
- **경계 불변식:** 모듈 import 시 imageio/torch/PySide6 미로드(import 가드 테스트).

### 6-2. 인코딩 라운드트립 (`test_video_export.py` 확장)
- 합성 프레임만(SAM2·PyAV·PySide6 비의존, 기존 픽스처 재사용).
- GIF: per-frame duration 메타 검증(슬로우 구간 duration ≈ 2배, centisecond 오차 허용). **패스트 10ms 클램프 회귀 가드**(8.3ms 의도 → 저장 후 ≥10ms, 기본 100ms 역전 없음).
- GIF `loop=0` 무한 루프 메타 회귀 가드(덕후 강점 보존).
- MP4: ffmpeg 조건부, 슬로우 구간 프레임 수 증가 검증.
- **무회귀:** `segments=()` 케이스 전부 기존과 동일.

### 6-3. 결합 (`test_video_capture.py` 확장)
- Fake 백엔드로 track→compute_boxes→export(segments) end-to-end.
- gap_policy(CUT) + segments 동시 적용 시 2단계 인덱싱 정확성.
- 예상 출력 프레임 수 헬퍼 ↔ 실제 export 프레임 수 일치(폭증 경고 신뢰성).
- 무회귀: `segments=()` export 결과 동일.

### 6-4. UI (`test_video_window_*.py` 패턴)
- 테이블→segments 변환 순수 헬퍼 단위 테스트(QWidget 인스턴스화 없이).
- 미리보기 프레임 인덱스 → 구간 행 주입 로직(위젯 분리 가능한 부분) 단위 테스트.

### 6-5. 후속 로드맵(테스트 관점 메모)
- **트림+슬로우+루프**(§8): 트림은 출력 인덱스 부분집합 선택 → 타임리맵 입력 길이 축소로 자연 확장. 별도 슬라이스에서 트림 구간 선택 + 기존 타임리맵 재사용 테스트.

---

## 7. 리스크 및 대응

| 리스크 | 심각도 | 대응 |
|---|---|---|
| **GIF 패스트 10ms 하한 붕괴**(delay=0 → 뷰어 ~100ms clamp → 의도 반대로 느려짐) | 중 | 결정1: GIF 경로 `clamp_durations_for_gif`로 10ms 미만을 20ms로 클램프 + 한국어 경고. 배속 한계가 base_fps 의존임을 UI 툴팁·문서 명시. 회귀 가드 테스트(6-1, 6-2). |
| **슬로우 품질**(프레임 복제 stutter, 끊김) | 중 | MVP는 복제로 한정·문서 명시. v1.1 보간(RIFE) ADR 별도. UI 0.25x 툴팁("끊겨 보일 수 있음")·"프레임 복제 방식" 안내. |
| **출력 프레임 수 폭증**(다중 슬로우 → MP4 복제 정수배 → 인코딩 시간·용량 급증) | 중 | export 전 `schedule_to_cfr_indices` 길이로 **예상 출력 프레임 수 사전 계산**(Task 4-3). 임계(예 기존 ×3 또는 N프레임 초과) 시 한국어 경고(데스크톱 QMessageBox·노트북 print). |
| **오디오 동기 깨짐**(슬로우 시) | 중 | 이번 범위는 무음(ADR 0011 무음 기준). 오디오 동기는 PoC H4 위 별도 후속 — 계획에 명시해 scope creep 차단. |
| **MP4 복제 한계**(정수배 아닌 배속 시 근사 오차·CFR 보간 불가) | 중 | base_fps 기준 등간격 복제/드롭으로 근사. 0.5x/2.0x 등 단순 배속 우선. 비정수 배속은 가장 가까운 프레임 매핑 + 문서 경고. |
| **구간 지정 UX 부적합**(프레임 번호 단독은 최애 파트 못 찾음 → MVP 가치 훼손) | 중 | 결정5: 미리보기 프레임→구간 버튼/mm:ss를 MVP에 흡수(Task 5-2). |
| **GIF centisecond 양자화**(10ms 배수 반올림) | 낮 | durations_ms 산출 후 GIF 인코딩 시 불가피한 양자화 — 테스트 허용오차(±10ms), 사용자 안내. |
| **2단계 인덱싱 버그**(gap_policy 출력 인덱스 ↔ 타임리맵 인덱스 혼동) | 중 | §4-2 재색인 규칙 명문화 + CUT 정책+segments 동시 결합 테스트(6-3) 필수. |
| **무회귀**(segments=() 경로 변경) | 높 | 모든 슬라이스 DoD에 기존 테스트 통과 포함. `VideoExportConfig` 기본값 빈 튜플. |
| **frozen dataclass 해시**(list 필드 비해시) | 낮 | `segments`는 list 아닌 tuple. |

---

## 8. MVP vs 후속 경계 (요약)

- **MVP(이번 Epic):** 프레임 복제 기반 단순 배속(0.25~4.0, GIF는 base_fps 의존 패스트 한계+클램프 가드), 다중 구간(겹침 금지), GIF(VFR, loop=0 무한루프)·MP4(CFR 복제), 데스크톱 SpinBox 테이블 + 미리보기→구간 버튼, 노트북 리스트 변수, 무음.
- **후속:**
  - **하이라이트 트림 + 슬로우 + 루프**(덕후 1순위): 구간만 잘라내 슬로우+무한루프 움짤 — 타임리맵 입력 길이 축소로 자연 확장.
  - 프레임 보간 슬로우(RIFE), 오디오 타임스트레치 동기, 임의/ramp 배속, 겹침 우선순위, 미리보기 타임라인 GUI(드래그 구간), MP4 진짜 VFR/webm.

---

## 9. 페르소나 검토 포인트 (3인 컨펌 루프 대상)

본 계획서는 메모리 규칙(persona-review-loop)에 따라 3인 페르소나 검토→수정→컨펌 루프를 거친다. **1차 검토 결과: 3인 전원 ⚠️조건부 컨펌(반대 없음). 위 ①~⑤ 반영 완료.** 2차(재검토) 확인 포인트:

- **베테랑 영상 전문가:** ① GIF 10ms 클램프(20ms) + 동적 패스트 상한이 역전 붕괴를 충분히 막는가. ② 프레임 복제 슬로우 stutter를 MVP로 한정한 판단. ③ GIF VFR / MP4 CFR 분리의 코덱 호환성. ④ 오디오 동기 범위 밖 결정의 후속 연결성. (1차 칭찬: VFR/CFR 분리·무음 슬로우 타당.)
- **베테랑 프로젝트 PM:** ① 마일스톤 그룹(알파=S1+S2+S4 / 베타=S3+S5+S6) 합리성. ② 출력 프레임 수 폭증 사전 계산·경고가 충분한가. ③ Story1 DoD에 `schedule_to_cfr_indices`·`clamp_durations_for_gif` 포함 + ADR 0013 머지 전 작성 명시. ④ 무회귀 DoD 전 슬라이스 적용. (1차 칭찬: 무회귀 DoD 모범적.)
- **베테랑 아이돌 덕후:** ① 미리보기→구간 버튼이 프레임 번호 암기 없이 최애 파트를 잡게 하는가(SpinBox 단독 부적합 해소). ② 트림+슬로우+루프가 후속에 명확히 잡혔는가. ③ 배속 프리셋(0.25/0.33/0.5/1/2x + 4x 고급)이 직캠 움짤에 충분한가. ④ GIF loop=0 자동 무한루프 강점 보존. (1차 칭찬: 다중 구간 테이블·GIF 자동 루프.)

수렴 규칙: 컨펌 패스에서는 "이미 반영된 항목 재트집 금지". 3인 전원 컨펌 시 ADR 0013 채택 + Story 1 착수(`/develop --tdd`).
