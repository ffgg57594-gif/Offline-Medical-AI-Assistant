"""
gui_app.py  —  Modern desktop GUI (PySide6) with chat + live voice orb.

Features:
    * Arabic-first chat UI with citation badges (clickable → verbatim paragraph)
    * Live voice orb that pulsates while listening/speaking
    * Instant barge-in button
    * All ML inference on background threads — UI never freezes
    * Sub-second first-token latency thanks to clause-streaming TTS
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .config import (
    REFUSAL_MESSAGE,
    STRICT_SIMILARITY_THRESHOLD,
    VOICE_MODELS_DIR,
)
from .llm_engine import LLMEngine, Citation
from .retriever import RetrievalResult, Retriever, create_retriever
from .live_voice import LiveVoiceSession


class CitationBadge(QWidget):
    """Clickable citation badge: shows verbatim paragraph in a tooltip-like popup."""

    clicked = Signal(str)

    def __init__(self, book_name: str, page_number: int, on_show: Callable[[str], None]) -> None:
        super().__init__()
        self.book_name = book_name
        self.page_number = page_number
        self._on_show = on_show
        self.setStyleSheet(
            "QLabel { color: #4a90d9; font-size: 9pt; cursor: pointer; }"
            "QLabel:hover { color: #1a4a7a; text-decoration: underline; }"
        )
        self.label = QLabel(f"[المصدر: {book_name} - صفحة {page_number}]")
        self.label.mousePressEvent = self._mouse_press

    def _mouse_press(self, event) -> None:
        self._on_show(f"{self.book_name} - صفحة {self.page_number}")


class VoiceOrb(QWidget):
    """Pulsating orb visualizer for voice state (idle / listening / speaking)."""

    STATE_IDLE = "idle"
    STATE_LISTENING = "listening"
    STATE_SPEAKING = "speaking"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = self.STATE_IDLE
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self.setFixedSize(80, 80)
        self.setStyleSheet("background-color: transparent;")

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if self._state != state:
            self._state = state
            if state == self.STATE_IDLE:
                self._timer.stop()
            else:
                self._timer.start(40)
            self.update()

    def _pulse(self) -> None:
        self._phase += 0.12
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2

        if self._state == self.STATE_IDLE:
            color = QColor(74, 144, 226)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(cx - 32), int(cy - 32), 64, 64)
        elif self._state == self.STATE_LISTENING:
            base = 40 + int(40 * (1 + self._phase % 1) / 2)
            color = QColor(base, base, 255 - base)
            for i in range(3):
                r = 20 + i * 14 + int(6 * (self._phase + i) % 1)
                painter.setBrush(QColor(68, 170, 219, max(60, 220 - r * 4)))
                painter.setPen(QPen(QColor(68, 170, 219), 1))
                painter.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)
            painter.setBrush(QColor(68, 170, 219))
            painter.drawEllipse(int(cx - 16), int(cy - 16), 32, 32)
        elif self._state == self.STATE_SPEAKING:
            base = 40 + int(40 * (1 + self._phase % 1) / 2)
            color = QColor(68, 219, 142)
            for i in range(3):
                r = 20 + i * 14 + int(6 * (self._phase + i) % 1)
                painter.setBrush(QColor(60, 230, 160, max(60, 220 - r * 4)))
                painter.setPen(QPen(QColor(60, 230, 160), 1))
                painter.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)
            painter.setBrush(QColor(60, 230, 160))
            painter.drawEllipse(int(cx - 16), int(cy - 16), 32, 32)


class ChatMessage(QWidget):
    """A single chat bubble (user or assistant) with optional citation badge."""

    def __init__(
        self,
        text: str,
        is_user: bool,
        on_citation: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._on_citation = on_citation
        self.setContentsMargins(8, 4, 8, 4)

        self.label = QTextEdit()
        self.label.setReadOnly(True)
        self.label.setFrameStyle(QTextEdit.FrameShape.NoFrame)
        self.label.setStyleSheet(
            "QTextEdit { color: #eef3f8; font-size: 10pt; line-height: 1.4; padding: 6px; border-radius: 8px; }"
        )
        if is_user:
            self.label.setStyleSheet(self.label.styleSheet() + " background-color: #1a3a5c;")
        else:
            self.label.setStyleSheet(self.label.styleSheet() + " background-color: #254166;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.label)

        badge_container = QWidget()
        badge_container.setContentsMargins(0, 2, 0, 0)
        badge_layout = QHBoxLayout(badge_container)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.addStretch()
        self.badge_container = badge_layout
        main_layout.addWidget(badge_container)

        self.set_text(text)

    def set_text(self, text: str) -> None:
        self.label.setHtml(self._format_text(text))
        while self.badge_container.count():
            child = self.badge_container.itemAt(0).widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()

    def _format_text(self, text: str) -> str:
        html = text.replace("\n", "<br>").replace(" ", "&nbsp;")
        badge = self._parse_badge(text)
        if badge:
            book, page = badge
            badge_widget = CitationBadge(book, page, lambda p: self._on_citation(book, page))
            badge_widget.label.mousePressEvent = badge_widget._mouse_press
            self.badge_container.addWidget(badge_widget)
            html = html.replace(
                f"[المصدر: {book} - صفحة {page}]",
                "",
            )
        return html

    def _parse_badge(self, text: str) -> tuple[str, int] | None:
        import re
        m = re.search(r"\[المصدر:\s*([^\s-]+)\s*-\s*صفحة\s*(\d+)\]", text)
        if m:
            return m.group(1), int(m.group(2))
        return None


class MainWindow(QMainWindow):
    MAIN_WINDOW_STYLESHEET = (
        "QMainWindow { background-color: #0f172a; }"
        "QMenuBar { background-color: #0f172a; color: #e2e8f0; }"
        "QMenuBar::item { padding: 4px 12px; border-radius: 4px; }"
        "QMenuBar::item:selected { background-color: #1e293b; }"
        "QMenu { background-color: #1e293b; color: #e2e8f0; }"
        "QMenu::item:selected { background-color: #334155; }"
        "QToolBar { background-color: #0f172a; border-bottom: 1px solid #1e293b; spacing: 6px; }"
        "QToolButton { background-color: transparent; color: #e2e8f0; padding: 6px 10px; border: none; border-radius: 6px; font-size: 10pt; }"
        "QToolButton:hover { background-color: #1e293b; }"
        "QStatusBar { background-color: #0f172a; color: #94a3b8; }"
        "QLineEdit { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; font-size: 10pt; }"
        "QLineEdit:focus { border: 1px solid #4a90d9; }"
        "QPushButton { background-color: #2563eb; color: white; border: none; border-radius: 8px; padding: 8px 16px; font-size: 10pt; }"
        "QPushButton:hover { background-color: #3b82f6; }"
        "QPushButton:pressed { background-color: #1d4ed8; }"
        "QPushButton:disabled { background-color: #334155; color: #64748b; }"
        "QScrollBar:vertical { background-color: #0f172a; width: 8px; border-radius: 4px; }"
        "QScrollBar::handle:vertical { background-color: #334155; border-radius: 4px; min-height: 30px; }"
        "QScrollBar::handle:vertical:hover { background-color: #475569; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar:horizontal { background-color: #0f172a; height: 8px; border-radius: 4px; }"
        "QScrollBar::handle:horizontal { background-color: #334155; border-radius: 4px; min-width: 30px; }"
        "QScrollBar::handle:horizontal:hover { background-color: #475569; }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }"
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("المساعد الطبي الذكي - Offline Medical AI Assistant")
        self.setMinimumSize(1100, 700)
        self.resize(1200, 750)
        self.setStyleSheet(self.MAIN_WINDOW_STYLESHEET)

        self._retriever: Retriever | None = None
        self._llm: LLMEngine | None = None
        self._voice: LiveVoiceSession | None = None
        self._llm_stream_task: threading.Thread | None = None
        self._orb: VoiceOrb | None = None

        self._init_ui()
        self._init_menu()
        self._init_menubar()
        self._start_backend()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with orb + title
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 20, 20, 10)
        header_layout.addSpacing(10)

        self._orb = VoiceOrb()
        self._orb.set_state(VoiceOrb.STATE_IDLE)
        header_layout.addWidget(self._orb)

        title = QLabel("المساعد الطبي الذكي")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e2e8f0;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.voice_status_label = QLabel("جاهز — انتظر كلامك")
        self.voice_status_label.setStyleSheet("color: #94a3b8; font-size: 9pt;")
        header_layout.addWidget(self.voice_status_label)

        header_layout.addSpacing(20)
        layout.addWidget(header)

        # Chat area
        chat_area = QScrollArea()
        chat_area.setWidgetResizable(True)
        chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chat_area.setStyleSheet("background-color: transparent; border: none;")
        self._chat_container = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(20, 10, 20, 10)
        self._chat_layout.setSpacing(8)
        self._chat_layout.addStretch()
        chat_area.setWidget(self._chat_container)
        layout.addWidget(chat_area, 1)

        # Input area
        input_bar = QWidget()
        input_bar.setStyleSheet("QWidget { background-color: #0f172a; border-top: 1px solid #1e293b; padding: 16px; }")
        input_layout = QHBoxLayout(input_bar)
        input_layout.setContentsMargins(20, 0, 20, 0)

        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText("اسأل سؤالاً طبيًا (يُجَاب من الكتب المدمجة فقط)...")
        self.message_edit.returnPressed.connect(self._on_send_message)
        input_layout.addWidget(self.message_edit, 1)

        send_btn = QPushButton("إرسال")
        send_btn.clicked.connect(self._on_send_message)
        input_layout.addWidget(send_btn)

        self.voice_btn = QPushButton("🎤 صوت مباشر")
        self.voice_btn.clicked.connect(self._toggle_voice)
        input_layout.addWidget(self.voice_btn)

        self.stop_voice_btn = QPushButton("⏹ إيقاف")
        self.stop_voice_btn.clicked.connect(self._stop_voice)
        self.stop_voice_btn.setDisabled(True)
        self.stop_voice_btn.setStyleSheet("background-color: #dc2626;")
        input_layout.addWidget(self.stop_voice_btn)

        layout.addWidget(input_bar)

        # Welcome message
        self._add_message("مرحبًا! أنا مساعد طبي يعتمد على الكتب الطبية المدمجة فقط. "
                          "كل معلومة تُجاب عنها تُستشهد بمصدرها بدقة.  "
                          "اضغط 🎤 لتتحدث، أو اكتب سؤالك هنا.", is_user=False)

    def _init_menu(self) -> None:
        menubar = self.menuBar()
        menubar.setStyleSheet("background-color: #0f172a;")

        file_menu = menubar.addMenu("الملف")
        close_action = QAction("إعادة تحميل الكتب", self)
        close_action.setShortcut("Ctrl+R")
        close_action.triggered.connect(self._reload_all)
        file_menu.addAction(close_action)
        exit_action = QAction("خروج", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("مساعدة")
        about_action = QAction("حول", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        toolbar = QToolBar("main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("background-color: #0f172a;")
        refresh_action = QAction("🔄 إعادة تحميل", self)
        refresh_action.triggered.connect(self._reload_all)
        toolbar.addAction(refresh_action)
        self.addAction(toolbar)

    # -----------------------------------------------------------------------
    # Backend startup
    # -----------------------------------------------------------------------
    def _start_backend(self) -> None:
        self.statusBar().showMessage("جارٍ تهيئة المحرك...")
        QTimer.singleShot(100, self._init_backend)

    def _init_backend(self) -> None:
        try:
            self._retriever = create_retriever()
            print(f"[gui] retriever ready — {self._retriever.passage_count} passages loaded")
            self.statusBar().showMessage(f"الكتب جاهزة — {self._retriever.passage_count} فقرة")

            self._llm = LLMEngine()
            print("[gui] LLM engine loaded")

            self._voice = LiveVoiceSession(
                on_transcript=self._on_voice_transcript,
                on_speaking_change=self._on_speaking_change,
                on_refusal=self._on_voice_refusal,
            )
            self.voice_btn.setDisabled(False)
            self.statusBar().showMessage("جاهز")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "خطأ", f"فشل تمهيد النظام:\n{exc}")
            self.statusBar().showMessage("خطأ في التمهيد")

    # -----------------------------------------------------------------------
    # Chat handling
    # -----------------------------------------------------------------------
    def _add_message(self, text: str, is_user: bool) -> None:
        msg = ChatMessage(text, is_user, self._on_citation_click)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, msg)
        self._chat_container.adjustSize()

    def _on_send_message(self) -> None:
        text = self.message_edit.text().strip()
        if not text:
            return
        self.message_edit.clear()
        self._add_message(text, is_user=True)
        self._process_query(text)

    def _process_query(self, query: str) -> None:
        if self._retriever is None or self._llm is None:
            return
        self.statusBar().showMessage("جارٍ البحث...")

        blocked, results = self._retriever.is_guardrail_blocked(query)
        if blocked:
            QTimer.singleShot(0, lambda: self._show_refusal(query))
            return

        cite: Citation | None = None
        if results and results[0].score > STRICT_SIMILARITY_THRESHOLD:
            cite = Citation(book_name=results[0].passage.book_name, page_number=results[0].passage.page_number)

        stream_text = ""
        start_time = time.perf_counter_ns()
        first_token_done = [False]

        def on_token(token: str) -> None:
            nonlocal stream_text
            stream_text += token
            if not first_token_done[0]:
                first_token_done[0] = True
                self.statusBar().showMessage(f"استجابة في {(time.perf_counter_ns() - start_time) / 1e6:.0f} ms")
            self._append_to_last_message(stream_text)

        thread = threading.Thread(
            target=self._llm.generate_stream,
            args=(query,),
            kwargs={"on_token": on_token, "cite_result": cite},
            daemon=True,
        )
        thread.start()
        self._llm_stream_task = thread
        self._add_message("جارِ التفكير...", is_user=False)

    def _show_refusal(self, query: str) -> None:
        self._add_message(REFUSAL_MESSAGE, is_user=False)
        if self._voice is not None:
            self._voice.play_refusal()

    def _append_to_last_message(self, text: str) -> None:
        for w in reversed(self._chat_layout.children()):
            if isinstance(w, ChatMessage):
                w.set_text(text)
                w.update()
                return
        self._add_message(text, is_user=False)

    def _on_citation_click(self, book_name: str, page_number: int) -> None:
        if self._retriever is None:
            return
        paragraph = self._retriever.get_verbatim_paragraph(book_name, page_number)
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"{book_name} - صفحة {page_number}")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setInformativeText(paragraph[:1200])
        dialog.exec()

    # -----------------------------------------------------------------------
    # Voice handling
    # -----------------------------------------------------------------------
    def _toggle_voice(self) -> None:
        if self._voice is None:
            return
        if self._voice.running:
            self._stop_voice()
        else:
            self._voice.start_listening()
            self._orb.set_state(VoiceOrb.STATE_LISTENING)
            self.voice_btn.setDisabled(True)
            self.stop_voice_btn.setDisabled(False)
            self.voice_status_label.setText("استمع...")

    def _stop_voice(self) -> None:
        if self._voice is None:
            return
        self._voice.stop_listening()
        self._orb.set_state(VoiceOrb.STATE_IDLE)
        self.voice_btn.setDisabled(False)
        self.stop_voice_btn.setDisabled(True)
        self.voice_status_label.setText("جاهز — انتظر كلامك")

    def _on_voice_transcript(self, transcript: str) -> None:
        if not transcript.strip():
            return
        self._add_message(transcript, is_user=True)
        self._process_voice_query(transcript)

    def _process_voice_query(self, query: str) -> None:
        if self._retriever is None or self._llm is None:
            return
        blocked, results = self._retriever.is_guardrail_blocked(query)
        if blocked:
            self._orb.set_state(VoiceOrb.STATE_SPEAKING)
            self._voice.play_refusal()
            QTimer.singleShot(600, self._voice.stop_speech)
            return

        cite: Citation | None = None
        if results and results[0].score > STRICT_SIMILARITY_THRESHOLD:
            cite = Citation(book_name=results[0].passage.book_name, page_number=results[0].passage.page_number)

        def on_token(token: str) -> None:
            if self._voice is not None and self._voice.tts.speaking:
                self._voice.tts.play_audio(np.fromiter(
                    iter(()), dtype=np.int16, count=0), blocking=False)

        thread = threading.Thread(
            target=self._llm.generate_stream,
            args=(query,),
            kwargs={"on_token": on_token, "cite_result": cite},
            daemon=True,
        )
        thread.start()
        self._llm_stream_task = thread
        self._orb.set_state(VoiceOrb.STATE_SPEAKING)
        self.voice_status_label.setText("يتكلم...")

    def _on_speaking_change(self, speaking: bool) -> None:
        if not speaking:
            self._orb.set_state(VoiceOrb.STATE_IDLE)
            self.voice_status_label.setText("جاهز — انتظر كلامك")
            self.voice_btn.setDisabled(False)

    def _on_voice_refusal(self, message: str) -> None:
        self._add_message(message, is_user=False)
        self._orb.set_state(VoiceOrb.STATE_IDLE)

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------
    def _reload_all(self) -> None:
        if self._llm_stream_task is not None:
            self._llm_stream_task.join(timeout=1.0)
        self._chat_container.setParent(None)
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(20, 10, 20, 10)
        self._chat_layout.setSpacing(8)
        self._chat_layout.addStretch()
        chat_area = self.findChild(QScrollArea)
        if chat_area is not None:
            chat_area.setWidget(self._chat_container)
        self._init_backend()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "حول",
            "المساعد الطبي الذكي v1.0\n\n"
            "تطبيق طبي غير متصل بالإنترنت يعتمد على الكتب الطبية المدمجة فقط.\n"
            "جميع الإجابات تُستشهد بمصدرها من الكتاب الأصلي.\n\n"
            "التقنيات: llama-cpp-python, faster-whisper, Piper TTS, Silero VAD"
        )

    def closeEvent(self, event) -> None:
        if self._voice is not None:
            self._voice.shutdown()
        if self._llm is not None:
            self._llm.shutdown()
        if self._retriever is not None:
            self._retriever.shutdown()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Offline Medical AI Assistant")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
