"""core/upscale 패키지 — 업스케일 Protocol + 순수 정규화 함수.

경계 불변식: torch·transformers·PySide6·av import 금지.
numpy만 의존한다.

포함:
  - backend.UpscaleBackend: Protocol (ADR 0004 / ADR 0007)
  - normalize.reconstruction_to_rgb_uint8: 순수 정규화 함수
"""
