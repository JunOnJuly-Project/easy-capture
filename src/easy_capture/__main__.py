"""앱 진입점: `python -m easy_capture`.

AppRouter를 통해 모드 선택 → 메인 윈도우 라우팅을 수행한다.
무거운 모델 로드는 UI 표시 이후(첫 클릭 시)로 미룬다(ADR 0007).
"""
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from easy_capture.app.router import AppRouter

    app = QApplication(sys.argv)
    router = AppRouter(app)
    router.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
