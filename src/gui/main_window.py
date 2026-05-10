"""BrainBoard GUI — PyQt6 application with Hex/Ring/Tree keyboard layouts.

Run with:
    PYTHONPATH=. python -m src.gui.main_window

This is a standalone GUI for testing the speller layouts with mouse + keyboard.
For BCI integration, see src/inference/pipeline.py and connect on_action to
the GUI's `dispatch_action` slot via Qt signals.

Architecture
------------
- MainWindow: top-level, hosts the keyboard widget, the text area, and the
  word suggestion bar.
- BaseKeyboardWidget: shared base for all 3 keyboard variants. Owns a
  reference to a `Controller` that can be either a Speller (Hex/Ring) or a
  TreeSpeller. Listens to keyboard events (L/R/B/S keys) and translates to
  speller actions.
- HexKeyboardWidget: 6 hexagons around a central point, expandable on select.
- RingKeyboardWidget: 36 letters on a circle, with the active sector
  rendered as a fan above.
- TreeKeyboardWidget: shows current node + immediate children + the path
  to current cursor.

Keyboard shortcuts (work in any layout):
    a / ←     →  move left
    d / →     →  move right
    Space     →  select (blink)
    Backspace →  back / undo
    1, 2, 3   →  insert suggested word
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Optional

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.speller import (
    BigramAdaptiveLayout,
    Direction,
    Speller,
    StaticGridLayout,
    UnsupportedTransitionError,
)
from src.speller.bigram_layout import BigramAdaptiveLayout as _Bigram
from src.speller.hex_layout import HexLayout
from src.speller.state import (
    SpellerStateIdle,
    SpellerStateLetterNavigation,
    SpellerStateSectorNavigation,
    SpellerStateWriting,
)
from src.speller.tree_speller import TreeSpeller, collect_leaves
from src.speller.word_suggester import WordSuggester


# Color palette (warm earth-tones to match the Ring-O-Spell screenshots)
COLOR_BG = QColor("#F5F2EA")
COLOR_KEY = QColor("#FFFFFF")
COLOR_KEY_BORDER = QColor("#D8D2C4")
COLOR_KEY_TEXT = QColor("#3A3530")
COLOR_KEY_HIGHLIGHT = QColor("#C4521A")
COLOR_KEY_HIGHLIGHT_TEXT = QColor("#FFFFFF")
COLOR_KEY_SECTOR_TINT = QColor("#F4E5D6")
COLOR_DASH = QColor("#D8D2C4")


# ════════════════════════════════════════════════════════════════════════════
# Base keyboard widget
# ════════════════════════════════════════════════════════════════════════════


class BaseKeyboardWidget(QWidget):
    """Shared base for keyboard widgets. Subclasses override paintEvent."""

    letter_committed = pyqtSignal(str)
    state_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def handle_action(self, action: str) -> None:
        """Subclasses translate 'left'/'right'/'select'/'back' into their state model."""
        raise NotImplementedError

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.handle_action("left")
        elif e.key() in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.handle_action("right")
        elif e.key() in (Qt.Key.Key_Space, Qt.Key.Key_W, Qt.Key.Key_Return):
            self.handle_action("select")
        elif e.key() in (Qt.Key.Key_S, Qt.Key.Key_Backspace):
            self.handle_action("back")
        else:
            super().keyPressEvent(e)


# ════════════════════════════════════════════════════════════════════════════
# Hex-O-Spell keyboard
# ════════════════════════════════════════════════════════════════════════════


class HexKeyboardWidget(BaseKeyboardWidget):
    """6 hexagons around a center. Standard Speller state machine drives navigation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout_ = HexLayout()
        self.speller = Speller(
            layout=self.layout_,
            on_letter_select=lambda c: self.letter_committed.emit(c),
        )
        self.speller.state = SpellerStateWriting()  # skip the Idle gate for GUI

    def handle_action(self, action: str) -> None:
        try:
            match action:
                case "left":
                    self.speller.move(Direction.LEFT)
                case "right":
                    self.speller.move(Direction.RIGHT)
                case "select":
                    self.speller.select()
                case "back":
                    self.speller.back()
        except UnsupportedTransitionError:
            pass
        self.update()
        self.state_changed.emit()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), COLOR_BG)

        cx, cy = self.width() / 2, self.height() / 2
        outer_r = min(self.width(), self.height()) * 0.32
        hex_r = outer_r * 0.32

        state = self.speller.state

        # Determine cursor position (which hexagon is highlighted) and whether
        # we're inside a sector.
        if isinstance(state, SpellerStateSectorNavigation):
            sector_cursor = state.cursor
            inside_sector = False
        elif isinstance(state, SpellerStateLetterNavigation):
            sector_cursor = state.selected_sector
            inside_sector = True
        else:
            sector_cursor = -1
            inside_sector = False

        # Draw the 6 hexagons
        for i in range(6):
            angle = -math.pi / 2 + i * (2 * math.pi / 6)  # start at top
            hx = cx + outer_r * math.cos(angle)
            hy = cy + outer_r * math.sin(angle)

            letters = self.layout_.letters_in_sector(i)
            highlighted = (i == sector_cursor)
            self._draw_hexagon(p, hx, hy, hex_r, letters, highlighted, inside_sector and highlighted, state)

        # Draw status hint at top
        p.setPen(QPen(COLOR_KEY_TEXT))
        p.setFont(QFont("Helvetica", 10))
        hint = self._status_hint(state)
        p.drawText(QRectF(0, 8, self.width(), 24), Qt.AlignmentFlag.AlignCenter, hint)

        p.end()

    def _draw_hexagon(self, p: QPainter, cx: float, cy: float, r: float,
                      letters: list[str], highlighted: bool, expanded: bool,
                      state) -> None:
        """Draw one hexagon. If expanded, show 6 letters; if collapsed, show all 6 small."""
        # Hexagon outline
        poly = QPolygonF()
        for k in range(6):
            a = -math.pi / 2 + k * (2 * math.pi / 6)
            poly.append(QPointF(cx + r * math.cos(a), cy + r * math.sin(a)))
        if highlighted and not expanded:
            p.setBrush(COLOR_KEY_HIGHLIGHT)
            text_color = COLOR_KEY_HIGHLIGHT_TEXT
        elif expanded:
            p.setBrush(COLOR_KEY_SECTOR_TINT)
            text_color = COLOR_KEY_TEXT
        else:
            p.setBrush(COLOR_KEY)
            text_color = COLOR_KEY_TEXT
        p.setPen(QPen(COLOR_KEY_BORDER, 1.5))
        p.drawPolygon(poly)

        if not expanded:
            # Show all letters in the hexagon as a small grid (2 rows × 3 cols).
            p.setPen(QPen(text_color))
            p.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            rows, cols = 2, 3
            for i, letter in enumerate(letters):
                row, col = i // cols, i % cols
                tx = cx + (col - (cols - 1) / 2) * (r * 0.5)
                ty = cy + (row - (rows - 1) / 2) * (r * 0.5)
                txt = " " if letter == "·" else letter
                p.drawText(QRectF(tx - 12, ty - 10, 24, 20),
                           Qt.AlignmentFlag.AlignCenter, txt)
        else:
            # Expanded: show one letter per slot, with cursor highlighted.
            cursor = state.cursor if isinstance(state, SpellerStateLetterNavigation) else -1
            p.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
            for i, letter in enumerate(letters):
                a = -math.pi / 2 + i * (2 * math.pi / 6)
                lx = cx + r * 0.6 * math.cos(a)
                ly = cy + r * 0.6 * math.sin(a)
                if i == cursor:
                    p.setBrush(COLOR_KEY_HIGHLIGHT)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(QPointF(lx, ly), 16, 16)
                    p.setPen(QPen(COLOR_KEY_HIGHLIGHT_TEXT))
                else:
                    p.setPen(QPen(COLOR_KEY_TEXT))
                txt = " " if letter == "·" else letter
                p.drawText(QRectF(lx - 14, ly - 12, 28, 24),
                           Qt.AlignmentFlag.AlignCenter, txt)

    def _status_hint(self, state) -> str:
        if isinstance(state, SpellerStateWriting):
            return "Press Space to enter sector navigation"
        if isinstance(state, SpellerStateSectorNavigation):
            return "←/→ navigates sectors • Space enters • Backspace cancels"
        if isinstance(state, SpellerStateLetterNavigation):
            return "←/→ navigates letters • Space picks • Backspace exits sector"
        return ""


# ════════════════════════════════════════════════════════════════════════════
# Ring-O-Spell keyboard
# ════════════════════════════════════════════════════════════════════════════


class RingKeyboardWidget(BaseKeyboardWidget):
    """36 letters on a circle; current sector (top-6 bigrams) shown as a fan above."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        bigrams_path = (
            "data/language/english_bigrams.json"
        )
        self.layout_ = _Bigram(bigrams_path=bigrams_path)
        self.speller = Speller(
            layout=self.layout_,
            on_letter_select=lambda c: self.letter_committed.emit(c),
        )
        self.speller.state = SpellerStateWriting()

    def handle_action(self, action: str) -> None:
        try:
            match action:
                case "left":
                    self.speller.move(Direction.LEFT)
                case "right":
                    self.speller.move(Direction.RIGHT)
                case "select":
                    self.speller.select()
                case "back":
                    self.speller.back()
        except UnsupportedTransitionError:
            pass
        self.update()
        self.state_changed.emit()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), COLOR_BG)

        cx, cy = self.width() / 2, self.height() * 0.62
        ring_r = min(self.width(), self.height()) * 0.24

        state = self.speller.state

        # Render all 36 ring positions
        all_letters: list[str] = []
        for i in range(self.layout_.n_sectors):
            all_letters.extend(self.layout_.letters_in_sector(i))

        # Determine which sector / letter is currently highlighted.
        highlighted_sector = -1
        highlighted_letter_idx = -1
        if isinstance(state, SpellerStateSectorNavigation):
            highlighted_sector = state.cursor
        elif isinstance(state, SpellerStateLetterNavigation):
            highlighted_sector = state.selected_sector
            highlighted_letter_idx = state.cursor

        # Draw ring positions (small circles, dashed connections)
        ring_positions: list[QPointF] = []
        # Distribute 36 letters around the ring, starting at top.
        for i in range(len(all_letters)):
            a = -math.pi / 2 + (i / len(all_letters)) * 2 * math.pi
            x = cx + ring_r * math.cos(a)
            y = cy + ring_r * math.sin(a)
            ring_positions.append(QPointF(x, y))

        # Dashed connecting line between ring positions
        p.setPen(QPen(COLOR_DASH, 1, Qt.PenStyle.DashLine))
        for i, pos in enumerate(ring_positions):
            nxt = ring_positions[(i + 1) % len(ring_positions)]
            p.drawLine(pos, nxt)

        # Letter tiles on the ring
        p.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        for i, (letter, pos) in enumerate(zip(all_letters, ring_positions)):
            sector_idx = i // 6
            is_in_active_sector = sector_idx == highlighted_sector and isinstance(state, SpellerStateSectorNavigation)
            if is_in_active_sector:
                p.setBrush(COLOR_KEY_SECTOR_TINT)
            else:
                p.setBrush(COLOR_KEY)
            p.setPen(QPen(COLOR_KEY_BORDER, 1))
            tile = QRectF(pos.x() - 14, pos.y() - 14, 28, 28)
            p.drawRoundedRect(tile, 4, 4)
            p.setPen(QPen(COLOR_KEY_TEXT))
            txt = " " if letter == "·" else letter
            p.drawText(tile, Qt.AlignmentFlag.AlignCenter, txt)

        # Active sector "fan" above the ring (Ring-O-Spell signature look)
        if highlighted_sector >= 0:
            sector_letters = self.layout_.letters_in_sector(highlighted_sector)
            # The fan sits OUTSIDE the ring, on the side of the highlighted sector.
            sector_center_idx = highlighted_sector * 6 + 2.5
            sector_center_angle = -math.pi / 2 + (sector_center_idx / 36) * 2 * math.pi

            # Push the fan center well outside the ring.
            fan_distance = ring_r * 1.55
            fan_cx = cx + fan_distance * math.cos(sector_center_angle)
            fan_cy = cy + fan_distance * math.sin(sector_center_angle)

            tile_size = 42
            tile_spacing = tile_size + 6
            # Tiles arranged along an arc tangent to the ring (perpendicular to the radial).
            perp = sector_center_angle + math.pi / 2
            for j, letter in enumerate(sector_letters):
                offset = (j - 2.5) * tile_spacing
                tx = fan_cx + offset * math.cos(perp)
                ty = fan_cy + offset * math.sin(perp)

                if isinstance(state, SpellerStateLetterNavigation) and j == highlighted_letter_idx:
                    p.setBrush(COLOR_KEY_HIGHLIGHT)
                    text_color = COLOR_KEY_HIGHLIGHT_TEXT
                else:
                    p.setBrush(COLOR_KEY)
                    text_color = COLOR_KEY_TEXT
                p.setPen(QPen(COLOR_KEY_HIGHLIGHT, 2))
                tile = QRectF(tx - tile_size / 2, ty - tile_size / 2, tile_size, tile_size)
                p.drawRoundedRect(tile, 6, 6)
                p.setPen(QPen(text_color))
                p.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
                txt = " " if letter == "·" else letter
                p.drawText(tile, Qt.AlignmentFlag.AlignCenter, txt)

        # Status hint
        p.setPen(QPen(COLOR_KEY_TEXT))
        p.setFont(QFont("Helvetica", 10))
        hint = self._status_hint(state)
        p.drawText(QRectF(0, 8, self.width(), 24), Qt.AlignmentFlag.AlignCenter, hint)

        p.end()

    def _status_hint(self, state) -> str:
        if isinstance(state, SpellerStateWriting):
            return "Press Space to enter sector navigation"
        if isinstance(state, SpellerStateSectorNavigation):
            return "←/→ rotates sectors • Space enters"
        if isinstance(state, SpellerStateLetterNavigation):
            return "←/→ navigates letters • Space picks"
        return ""


# ════════════════════════════════════════════════════════════════════════════
# Tree keyboard
# ════════════════════════════════════════════════════════════════════════════


class TreeKeyboardWidget(BaseKeyboardWidget):
    """Huffman tree: shows current node + its left/right subtrees."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tree = TreeSpeller(on_letter_select=lambda c: self.letter_committed.emit(c))

    def handle_action(self, action: str) -> None:
        match action:
            case "left":
                self.tree.move(Direction.LEFT)
            case "right":
                self.tree.move(Direction.RIGHT)
            case "select":
                self.tree.select()
            case "back":
                self.tree.back()
        self.update()
        self.state_changed.emit()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), COLOR_BG)

        # Layout: current node big in center, two subtree summaries to L and R below.
        cx = self.width() / 2
        cy_top = self.height() * 0.30
        cy_bot = self.height() * 0.65
        node_size = 80

        # Current node
        p.setBrush(COLOR_KEY_HIGHLIGHT)
        p.setPen(QPen(COLOR_KEY_HIGHLIGHT, 2))
        p.drawRoundedRect(QRectF(cx - node_size / 2, cy_top - node_size / 2, node_size, node_size), 8, 8)
        p.setPen(QPen(COLOR_KEY_HIGHLIGHT_TEXT))
        p.setFont(QFont("Helvetica", 28, QFont.Weight.Bold))
        if self.tree.cursor.is_leaf:
            label = self.tree.cursor.letter or ""
        else:
            label = "•"  # internal node
        p.drawText(QRectF(cx - node_size / 2, cy_top - node_size / 2, node_size, node_size),
                   Qt.AlignmentFlag.AlignCenter, label)

        # Left/Right subtree summaries
        left_letters = self.tree.left_subtree_letters()
        right_letters = self.tree.right_subtree_letters()

        # Connecting lines from parent to children boxes
        p.setPen(QPen(COLOR_DASH, 2))
        p.drawLine(QPointF(cx, cy_top + node_size / 2), QPointF(cx - 180, cy_bot - 50))
        p.drawLine(QPointF(cx, cy_top + node_size / 2), QPointF(cx + 180, cy_bot - 50))

        # Boxes
        for label_text, letters, dx in [("LEFT (←)", left_letters, -180), ("RIGHT (→)", right_letters, +180)]:
            box_w, box_h = 220, 130
            box = QRectF(cx + dx - box_w / 2, cy_bot - 50, box_w, box_h)
            p.setBrush(COLOR_KEY)
            p.setPen(QPen(COLOR_KEY_BORDER, 1.5))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QPen(COLOR_KEY_TEXT))
            p.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
            p.drawText(QRectF(box.x(), box.y() + 6, box.width(), 18),
                       Qt.AlignmentFlag.AlignCenter, label_text)
            p.setFont(QFont("Helvetica", 12))
            content = ", ".join(c if c != " " else "␣" for c in letters[:18])
            if len(letters) > 18:
                content += " …"
            p.drawText(QRectF(box.x() + 8, box.y() + 28, box.width() - 16, box.height() - 36),
                       Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                       content if content else "(leaf)")

        # Status hint
        p.setPen(QPen(COLOR_KEY_TEXT))
        p.setFont(QFont("Helvetica", 10))
        hint = "← / → descends in tree • Space resets to root • Letter commits automatically at leaf"
        p.drawText(QRectF(0, 8, self.width(), 24), Qt.AlignmentFlag.AlignCenter, hint)

        p.end()


# ════════════════════════════════════════════════════════════════════════════
# Main window
# ════════════════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainBoard")
        self.resize(900, 740)

        # Word suggester
        try:
            self.suggester = WordSuggester()
        except FileNotFoundError:
            self.suggester = None

        self.typed_text = ""

        # Toolbar with layout selector
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("  Layout:  "))
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["Hex-O-Spell", "Ring-O-Spell", "Tree (Huffman)"])
        self.layout_selector.currentIndexChanged.connect(self._switch_layout)
        toolbar.addWidget(self.layout_selector)

        toolbar.addSeparator()
        clear_btn = QPushButton("Clear text")
        clear_btn.clicked.connect(self._clear_text)
        toolbar.addWidget(clear_btn)

        # Central widget
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Text area at top
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(80)
        self.text_display.setStyleSheet(
            "QTextEdit { font-size: 18pt; background-color: #FFFFFF; "
            "border: 1px solid #D8D2C4; border-radius: 4px; padding: 8px; }"
        )
        layout.addWidget(self.text_display)

        # Suggestion bar
        self.suggestion_bar = QWidget()
        sug_layout = QHBoxLayout(self.suggestion_bar)
        sug_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestion_buttons: list[QPushButton] = []
        for i in range(3):
            b = QPushButton("")
            b.setStyleSheet(
                "QPushButton { font-size: 14pt; padding: 8px; background-color: #FFFFFF; "
                "border: 1px solid #D8D2C4; border-radius: 4px; } "
                "QPushButton:hover { background-color: #F4E5D6; } "
                "QPushButton:disabled { color: #D8D2C4; }"
            )
            b.clicked.connect(lambda _, idx=i: self._accept_suggestion(idx))
            sug_layout.addWidget(b)
            self.suggestion_buttons.append(b)
        layout.addWidget(self.suggestion_bar)

        # Keyboard widget (replaceable by _switch_layout)
        self.keyboard_container = QWidget()
        self.keyboard_layout = QVBoxLayout(self.keyboard_container)
        self.keyboard_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.keyboard_container, stretch=1)

        self.keyboard_widget: BaseKeyboardWidget = None  # type: ignore
        self._switch_layout(0)  # start with Hex

        self.statusBar().showMessage(
            "Use ← / → to navigate, Space to select, Backspace to undo. Click 1/2/3 to accept a suggestion.",
            10000,
        )
        self._update_suggestions()

    def _switch_layout(self, idx: int) -> None:
        """Replace the keyboard widget."""
        if self.keyboard_widget is not None:
            self.keyboard_layout.removeWidget(self.keyboard_widget)
            self.keyboard_widget.deleteLater()

        if idx == 0:
            self.keyboard_widget = HexKeyboardWidget()
        elif idx == 1:
            self.keyboard_widget = RingKeyboardWidget()
        elif idx == 2:
            self.keyboard_widget = TreeKeyboardWidget()
        else:
            return

        self.keyboard_widget.letter_committed.connect(self._on_letter)
        self.keyboard_layout.addWidget(self.keyboard_widget)
        self.keyboard_widget.setFocus()

    def _on_letter(self, letter: str) -> None:
        if letter == "·":
            return
        self.typed_text += letter
        self._refresh_text()
        self._update_suggestions()

    def _accept_suggestion(self, idx: int) -> None:
        if idx >= len(self.suggestion_buttons):
            return
        word = self.suggestion_buttons[idx].text().strip()
        if not word:
            return

        # Replace the in-progress prefix with the full word + a space.
        # Find last space (or start) and chop everything from there.
        last_space = self.typed_text.rfind(" ")
        prefix_start = last_space + 1
        self.typed_text = self.typed_text[:prefix_start] + word + " "
        self._refresh_text()
        self._update_suggestions()
        self.keyboard_widget.setFocus()

    def _clear_text(self) -> None:
        self.typed_text = ""
        self._refresh_text()
        self._update_suggestions()
        self.keyboard_widget.setFocus()

    def _refresh_text(self) -> None:
        self.text_display.setPlainText(self.typed_text)

    def _update_suggestions(self) -> None:
        if self.suggester is None:
            for b in self.suggestion_buttons:
                b.setText("(no dict)")
                b.setEnabled(False)
            return

        # Current in-progress word = chars after last space.
        last_space = self.typed_text.rfind(" ")
        prefix = self.typed_text[last_space + 1:].lower()

        suggestions = self.suggester.suggest(prefix, n_results=3)
        for i, b in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                b.setText(f"{i + 1}. {suggestions[i]}")
                b.setEnabled(True)
            else:
                b.setText("")
                b.setEnabled(False)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        # Number keys 1/2/3 accept suggestions
        if e.key() in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
            idx = e.key() - Qt.Key.Key_1
            self._accept_suggestion(idx)
            return
        # Forward all other keys to the keyboard widget so it can handle L/R/B/S.
        if self.keyboard_widget is not None:
            self.keyboard_widget.keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(f"QMainWindow {{ background-color: {COLOR_BG.name()}; }}")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
