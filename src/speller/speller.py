from typing import Callable, Optional

from src.speller.layout import SpellerLayout
from src.speller.state import Direction, SpellerState, SpellerStateIdle
from src.speller.state import UnsupportedTransitionError
from src.speller.static_layout import StaticGridLayout


class Speller:
    """BCI keyboard façade — drives a state machine over a pluggable layout.

    The layout decides what letters are reachable at any moment; the state
    machine decides how navigation proceeds. The state machine code itself
    is fully agnostic of layout details.

    Parameters
    ----------
    layout : SpellerLayout, optional
        How letters are organized. Defaults to StaticGridLayout (5×6).
        Pass BigramAdaptiveLayout for context-sensitive Polish keyboard.
    on_letter_select : callable, optional
        Called whenever a letter is committed.
    """

    def __init__(
        self,
        layout: Optional[SpellerLayout] = None,
        on_letter_select: Callable[[str], None] | None = None,
    ):
        self.state: SpellerState = SpellerStateIdle()
        self.layout: SpellerLayout = layout if layout is not None else StaticGridLayout()

        if on_letter_select is not None:
            self.on_letter_select_hook = on_letter_select
        else:
            self.on_letter_select_hook = lambda letter: print(f"Selected letter: {letter}")

    # State machine reads .map for backwards compatibility — proxy through
    # the layout so existing state.py code keeps working.
    @property
    def map(self) -> dict[int, list[str]]:
        return {i: self.layout.letters_in_sector(i) for i in range(self.layout.n_sectors)}

    def on_letter_select(self, letter: str) -> None:
        """Called by SpellerStateLetterNavigation.select(). Updates layout context, then fires the hook."""
        self.layout.on_letter_committed(letter)
        self.on_letter_select_hook(letter)

    def back(self) -> SpellerState:
        self.state.back(self)
        # Returning fully to Idle resets layout context (start a new word).
        if isinstance(self.state, SpellerStateIdle):
            self.layout.reset()
        return self.state

    def select(self) -> SpellerState:
        self.state.select(self)
        return self.state

    def move(self, direction: Direction) -> SpellerState:
        self.state.move(self, direction)
        return self.state


if __name__ == "__main__":
    speller = Speller()

    while True:
        try:
            command = (
                input("AD - Move left/move right, W - Select, S - Back: ")
                .strip()
                .upper()
            )
            match command:
                case "A":
                    speller.move(Direction.LEFT)
                case "D":
                    speller.move(Direction.RIGHT)
                case "W":
                    speller.select()
                case "S":
                    speller.back()
            print(speller.state)
        except UnsupportedTransitionError as e:
            print(e)
