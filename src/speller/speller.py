from typing import Callable

from src.speller.state import Direction, SpellerState, SpellerStateIdle
from src.speller.state import UnsupportedTransitionError


SECTOR_DICTIONARY = {
    0: ["a", "b", "c", "d", "e", "f"],
    1: ["g", "h", "i", "j", "k", "l"],
    2: ["m", "n", "o", "p", "q", "r"],
    3: ["s", "t", "u", "v", "w", "x"],
    4: ["y", "z", " ", ".", ",", "?"],
}


class Speller:
    def __init__(self, on_letter_select: Callable[[str], None] | None = None):
        self.state: SpellerState = SpellerStateIdle()
        self.map = SECTOR_DICTIONARY

        if on_letter_select is not None:
            self.on_letter_select = on_letter_select
        else:
            self.on_letter_select = lambda letter: print(f"Selected letter: {letter}")

    def back(self) -> SpellerState:
        self.state.back(self)
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
