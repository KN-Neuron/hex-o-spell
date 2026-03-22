# So that the type "Speller" doesn't have to be in quotes everywhere
from __future__ import annotations

import enum
from typing import TYPE_CHECKING

# Only needed when actually type checking, to avoid circular imports
if TYPE_CHECKING:
    from src.speller.speller import Speller


class Direction(enum.Enum):
    LEFT = 1
    RIGHT = 2


class SpellerState:
    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def back(self, _: Speller) -> None:
        raise UnsupportedTransitionError("Current state does not support 'back'.")

    def select(self, _: Speller) -> None:
        raise UnsupportedTransitionError("Current state does not support 'select'.")

    def move(self, _: Speller, _dir: Direction) -> None:
        raise UnsupportedTransitionError("Current state does not support 'move'.")


class UnsupportedTransitionError(Exception):
    pass


class SpellerStateIdle(SpellerState):
    def select(self, speller: Speller) -> None:
        speller.state = SpellerStateWriting()


class SpellerStateWriting(SpellerState):
    def __init__(self) -> None:
        self.cursor = 0

    def back(self, speller: Speller) -> None:
        speller.state = SpellerStateIdle()

    def select(self, speller: Speller) -> None:
        speller.state = SpellerStateSectorNavigation()

    def move(self, speller: Speller, direction: Direction) -> None:
        raise UnsupportedTransitionError("Moving in writing state not implemented yet")


class SpellerStateSectorNavigation(SpellerState):
    def __init__(self) -> None:
        self.cursor = 0

    def back(self, speller: Speller) -> None:
        speller.state = SpellerStateWriting()

    def select(self, speller: Speller) -> None:
        speller.state = SpellerStateLetterNavigation(self.cursor)

    def move(self, speller: Speller, direction: Direction) -> None:
        match direction:
            case Direction.LEFT:
                self.cursor = (self.cursor - 1) % len(speller.map)
            case Direction.RIGHT:
                self.cursor = (self.cursor + 1) % len(speller.map)


class SpellerStateLetterNavigation(SpellerState):
    def __init__(self, selected_sector: int) -> None:
        self.selected_sector = selected_sector
        self.cursor = 0

    def back(self, speller: Speller) -> None:
        speller.state = SpellerStateSectorNavigation()

    def select(self, speller: Speller) -> None:
        speller.on_letter_select(speller.map[self.selected_sector][self.cursor])

        speller.state = SpellerStateWriting()

    def move(self, speller: Speller, direction: Direction) -> None:
        letters = speller.map[self.selected_sector]

        match direction:
            case Direction.LEFT:
                self.cursor = (self.cursor - 1) % len(letters)
            case Direction.RIGHT:
                self.cursor = (self.cursor + 1) % len(letters)
