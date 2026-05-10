"""Interactive REPL for the speller — type L/R/B from the keyboard, watch state move.

This is the "I want to see what's going on in the project" mode: it bypasses
the EEG pipeline entirely and lets you drive the speller directly with
keyboard input. Useful for:

  - Onboarding: see the state machine and layout in action without wiring
    up hardware.
  - Sanity-checking a layout: pick BigramAdaptiveLayout and watch sectors
    re-shuffle after each typed letter.
  - Demoing the keyboard to people who'd otherwise need a 30-minute setup.

Controls
--------
    L / a / ←        →  speller.move(LEFT)
    R / d / →        →  speller.move(RIGHT)
    B / w / space    →  speller.select()    (the "blink")
    S / s            →  speller.back()
    Q                →  quit

Usage
-----
    PYTHONPATH=. python -m src.eeg_headset.cmd.scripted_repl --layout bigram
    PYTHONPATH=. python -m src.eeg_headset.cmd.scripted_repl --layout static
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.speller import (
    BigramAdaptiveLayout,
    Direction,
    Speller,
    StaticGridLayout,
    UnsupportedTransitionError,
)
from src.speller.state import (
    SpellerStateIdle,
    SpellerStateLetterNavigation,
    SpellerStateSectorNavigation,
    SpellerStateWriting,
)


def _render(speller: Speller, typed_so_far: str) -> str:
    """Render the current speller state as a friendly multiline string."""
    state = speller.state
    state_name = type(state).__name__.replace("SpellerState", "")
    lines = [
        "─" * 60,
        f"State: {state_name}    Typed so far: {typed_so_far!r}",
    ]

    if isinstance(state, SpellerStateIdle):
        lines.append("[Idle] Press B to start writing.")

    elif isinstance(state, SpellerStateWriting):
        lines.append("[Writing] Press B to enter sector navigation, S to go back to Idle.")

    elif isinstance(state, SpellerStateSectorNavigation):
        lines.append("[SectorNavigation] Cursor is on a sector. L/R rotates, B enters.")
        for i in range(speller.layout.n_sectors):
            letters = "".join(speller.layout.letters_in_sector(i))
            marker = "▶" if i == state.cursor else " "
            lines.append(f"  {marker} sector {i}: [{letters}]")

    elif isinstance(state, SpellerStateLetterNavigation):
        lines.append(f"[LetterNavigation] In sector {state.selected_sector}. L/R navigates, B picks.")
        letters = speller.layout.letters_in_sector(state.selected_sector)
        rendered = "  ".join(
            f"({c})" if i == state.cursor else f" {c} "
            for i, c in enumerate(letters)
        )
        lines.append(f"  {rendered}")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--layout",
        choices=["static", "bigram"],
        default="bigram",
        help="Speller layout to use (default: bigram)",
    )
    args = parser.parse_args(argv)

    if args.layout == "bigram":
        layout = BigramAdaptiveLayout()
    else:
        layout = StaticGridLayout()

    typed: list[str] = []
    speller = Speller(layout=layout, on_letter_select=lambda c: typed.append(c))

    print("Scripted speller REPL.")
    print("Commands: L/a = left, R/d = right, B/w/space = select (blink), S/s = back, Q = quit\n")
    print(_render(speller, ""))

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if cmd in ("q", "quit", "exit"):
            print(f"Final typed: {''.join(typed)!r}")
            return 0

        try:
            match cmd:
                case "l" | "a" | "left":
                    speller.move(Direction.LEFT)
                case "r" | "d" | "right":
                    speller.move(Direction.RIGHT)
                case "b" | "w" | "" | "blink" | "select":  # empty input = press Enter = blink
                    speller.select()
                case "s" | "back":
                    speller.back()
                case _:
                    print(f"  ? unknown command {cmd!r}. Try L, R, B, S, or Q.")
                    continue
        except UnsupportedTransitionError as e:
            print(f"  ✗ illegal in current state: {e}")
            continue

        print(_render(speller, "".join(typed)))


if __name__ == "__main__":
    sys.exit(main())
