"""앱 진입점: `python -m easy_capture`.

시작 화면(이미지/GIF 모드 선택)을 띄운다. 무거운 모델 로드는 UI 표시 이후로 미룬다.
"""
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from easy_capture.ui.mode_select import ModeSelectWindow

    app = QApplication(sys.argv)
    window = ModeSelectWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
