"""테스트용 가짜 의존성(테스트 더블) 패키지.

실제 PyAV·SAM2·transformers·torch를 전혀 로드하지 않고
Protocol 계약만 준수하는 결정적(deterministic) 구현을 제공한다.
"""
from tests.fixtures.fakes import FakeBackend, FakeFrameSource

__all__ = ["FakeBackend", "FakeFrameSource"]
