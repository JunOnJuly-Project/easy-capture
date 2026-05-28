"""ui/video_window의 occlusion 갭 정책 콤보 매핑 테스트.

_GAP_POLICY_ITEMS(라벨↔GapPolicy)가 모든 정책을 빠짐없이 커버하고
콤보 기본 선택이 BACKGROUND인지 검증한다.

PySide6 모듈 상수 접근만 하므로 QApplication 인스턴스화는 불필요하다(offscreen 설정).
"""
from __future__ import annotations

import os

# WHY: video_window import가 PySide6 위젯 클래스를 정의하므로 헤드리스 플랫폼 지정
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 — UI 모듈 import 불가")

from easy_capture.core.tracking.gap_policy import GapPolicy
from easy_capture.ui.video_window import _GAP_POLICY_ITEMS


class TestGapPolicyItems:
    """_GAP_POLICY_ITEMS: 콤보 라벨 ↔ GapPolicy 매핑 계약."""

    def test_모든_GapPolicy를_빠짐없이_포함한다(self):
        """Given: _GAP_POLICY_ITEMS
        When:  정책 집합 추출
        Then:  GapPolicy 전체(BACKGROUND·CUT·FREEZE)와 일치
        """
        policies = {policy for _, policy in _GAP_POLICY_ITEMS}

        assert policies == set(GapPolicy), "콤보가 일부 정책을 누락/추가함"

    def test_콤보_기본_선택은_BACKGROUND이다(self):
        """Given: 콤보 첫 항목(index 0 = 기본 선택)
        When:  정책 확인
        Then:  BACKGROUND (export 기본값과 일치 — 무회귀)
        """
        assert _GAP_POLICY_ITEMS[0][1] == GapPolicy.BACKGROUND

    def test_라벨은_중복되지_않는다(self):
        """Given: 콤보 라벨들
        When:  중복 검사
        Then:  모두 유일
        """
        labels = [label for label, _ in _GAP_POLICY_ITEMS]

        assert len(labels) == len(set(labels))

    def test_각_항목은_비어있지_않은_라벨과_GapPolicy_쌍이다(self):
        """Given: 각 항목
        When:  타입 검사
        Then:  (비어있지 않은 str, GapPolicy)
        """
        for label, policy in _GAP_POLICY_ITEMS:
            assert isinstance(label, str) and label, f"라벨이 비었음: {label!r}"
            assert isinstance(policy, GapPolicy)
