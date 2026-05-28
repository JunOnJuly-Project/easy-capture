# ADR 0011 — GIF/MP4 인코딩 위치

- 상태: 채택
- 날짜: 2026-05-28

## 맥락

비디오 모드([비디오 슬라이스 계획서](../plans/video-tracking-slice.md) §3-3)는 프레임 시퀀스를 GIF 또는 MP4로 내보낸다. 인코딩에는 `imageio`·`imageio-ffmpeg`(MP4 경로) 또는 `PyAV`가 필요하다. 이 의존성을 **어느 레이어에 두어야 하는가**를 결정해야 한다.

두 가지 제약이 충돌한다.

1. **core 비의존 불변식**([ADR 0008](0008-app-usecase-layer.md)): core는 torch·transformers·PySide6·`av`(PyAV)를 import하지 않는다.

2. **이미지 export 선례**: `core/export/image_export.py`가 이미 Pillow(외부 라이브러리)를 core에서 사용한다. Pillow는 순수 인코딩 라이브러리로서 무거운 ML 의존성과 다르다는 판단 하에 허용되었다.

`imageio`/`imageio-ffmpeg`는 ML 의존성이 아닌 인코딩 라이브러리다. 하지만 일반 import로 두면 core 테스트 환경에서 설치 여부에 따라 import 오류가 발생할 수 있다.

## 결정

`core/export/video_export.py`에 두되, **`imageio`·`imageio-ffmpeg` import를 `encode_frames` 함수 내부 지연 import로 격리**한다.

### 모듈 구조

```python
# core/export/video_export.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.tracking.gap_policy import GapPolicy, build_output_indices
from core.export.image_export import crop_array   # DRY — crop 재사용

@dataclass(frozen=True)
class VideoExportConfig:
    """움짤 내보내기 설정(불변). 이미지 ExportConfig 계승."""
    fmt: str = "gif"                             # "gif" | "mp4"
    fps: float = 12.0
    gap_policy: GapPolicy = GapPolicy.BACKGROUND  # 이번 슬라이스: 1종

def crop_frames(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """프레임별 박스로 크롭(순수). image_export.crop_array를 루프로 재사용(DRY)."""
    return [crop_array(f, b) for f, b in zip(frames, boxes)]

def encode_frames(
    crops: list[np.ndarray],
    path: str,
    config: VideoExportConfig,
) -> None:
    """크롭 프레임 시퀀스를 GIF/MP4로 인코딩(부수효과 = 파일 쓰기).

    WHY — imageio를 함수 내부 지연 import로 격리:
    core 모듈이 로드될 때마다 imageio가 import되면 imageio-ffmpeg 미설치
    환경(이미지 전용 사용, 단위 테스트)에서 ImportError가 발생한다.
    지연 import로 실제 인코딩 경로에서만 의존성을 활성화한다.
    Pillow 기반 image_export 선례와 동일한 기준이나, MP4는 ffmpeg 바인딩
    필요성 때문에 top-level import 대신 지연 import로 한 단계 더 격리한다.
    """
    import imageio  # noqa: PLC0415 — 의도적 지연 import
    imageio.mimwrite(path, crops, fps=config.fps)
```

### 핵심 설계 결정

- **core 위치 + 지연 import 조합**: core에 위치해 이미지 `image_export`(Pillow)와 대칭을 맞추면서, `imageio`는 함수 내부 지연 import로 격리한다. core 비의존 불변식의 금지 목록(torch·transformers·PySide6·`av`)에 `imageio`는 포함되지 않으므로 허용되며, 지연 import로 미설치 환경의 import 오류를 방어한다.

- **`crop_frames`는 `image_export.crop_array` 재사용**: 프레임별 크롭은 `crop_array`를 루프로 감싸는 순수 함수다. 슬라이스 로직을 중복하지 않는다(DRY).

- **고정 box size 불변식**: GIF/MP4는 전 프레임이 동일 W×H여야 인코딩된다. `VideoCaptureUseCase.compute_boxes`가 구간 내 고정 크기로 박스를 통일한다([계획서](../plans/video-tracking-slice.md) §3-3). `encode_frames`는 이 불변식을 전제로 단순화된다.

- **PyAV는 인코딩에 미사용**: [ADR 0005](0005-video-io-pyav-vfr.md)의 PyAV는 디코드(읽기) 전용이다. 인코딩에 PyAV를 쓰면 `av`가 core에 진입해 비의존 불변식을 깨뜨린다. `imageio`+`imageio-ffmpeg`가 MP4 인코딩에 더 간결하다.

## 대안

**(a) infra에 배치 — `infra/video_encode.py`**

인코딩 의존성을 infra에 둔다. core 비의존 원칙을 가장 엄격하게 지키는 안이다.

- 거부 이유: `core/export/image_export.py`(Pillow)와 비대칭이 생긴다. 이미지 export는 core에, 비디오 export는 infra에 있으면 레이어 경계가 일관되지 않아 유지보수 혼란을 야기한다. 또한 인코딩은 순수 변환(부수효과 = 파일 쓰기뿐)으로, 도메인 데이터 변환 성격을 가져 core가 소유하기에 적합하다. 리뷰에서 이 결정이 뒤집힐 경우 `video_export.py` 1개 파일만 infra로 이동하면 되며, `VideoCaptureUseCase`는 Protocol 미사용(직접 호출)이라 영향 범위가 최소다.

**(b) PyAV로 인코딩 — infra/video_encode.py**

디코드에 이미 쓰는 PyAV로 MP4 인코딩도 처리한다.

- 거부 이유: `av`가 core로 진입하면 core 비의존 불변식([ADR 0008](0008-app-usecase-layer.md))이 깨진다. infra에 두더라도 [ADR 0005](0005-video-io-pyav-vfr.md) PyAV는 디코드 전용으로 결정되었다. PyAV 인코딩 API는 `imageio`보다 장황하며, 이번 슬라이스(무음 단순 프레임 시퀀스)에 필요 이상의 복잡도를 도입한다.

## 결과

### 긍정적 영향

- **레이어 대칭**: 이미지 `core/export/image_export.py`(Pillow)와 비디오 `core/export/video_export.py`(imageio)가 같은 레이어에 위치한다. export 로직을 찾는 단일 진입점이 생긴다.
- **순수 인터페이스**: `crop_frames`는 순수 함수다. 인코딩 의존성과 분리되어 있어 `imageio` 없이도 크롭 결과를 단위 테스트할 수 있다.
- **GIF 라운드트립 테스트 가능**: `imageio.mimread`로 인코딩 결과를 디코딩해 프레임 수·크기 일치를 검증하는 라운드트립 테스트가 단일 의존성 내에서 완결된다. MP4는 `importorskip("imageio_ffmpeg")`으로 조건부 테스트.
- **미설치 환경 방어**: 지연 import로 `imageio-ffmpeg` 미설치 환경에서도 `video_export` 모듈 로드 자체는 성공한다.

### 부정적 영향 / 트레이드오프

- **core 순수성 타협**: 엄격한 관점에서 core는 numpy·stdlib만 사용해야 한다. `imageio` 허용은 Pillow 선례와 동일 기준이나, 기준 자체가 외부 라이브러리를 일부 허용하는 실용적 타협이다. 이 기준이 향후 더 무거운 의존성 허용의 전례가 되지 않도록 금지 목록(torch·transformers·PySide6·av)은 명문화된 상태로 유지한다.
- **업스케일·오디오 결합 미포함**: 프레임별 업스케일([ADR 0009](0009-upscale-export-integration.md))과 오디오 동기(PoC H4)는 후속 슬라이스에서 `encode_frames` 시그니처 확장 또는 래퍼로 통합한다. 이번 슬라이스의 `VideoExportConfig`는 무음·단일 fps 기준으로 최소화되었다.

### 후속 연계

- [ADR 0010](0010-video-segmentation-backend.md) — 같은 비디오 슬라이스의 Protocol 분리 결정.
- [ADR 0009](0009-upscale-export-integration.md) — 업스케일 export 통합 패턴. 비디오 export에 업스케일을 결합할 때 `crop_frames` 이후 단계에 옵션 주입으로 확장한다.
- [ADR 0005](0005-video-io-pyav-vfr.md) — PyAV 디코드 전용 결정. 인코딩과 디코딩의 레이어 분리를 유지한다.
- [비디오 슬라이스 계획서](../plans/video-tracking-slice.md) §3-3·§9: 이 ADR이 "GIF/MP4 인코딩 위치" 트리거를 해소한다.
