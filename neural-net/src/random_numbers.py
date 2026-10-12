from typing import Callable


# A seedable random number generator: returns a function that gives a new number in the
# range 0 to 1 each time it is called. The same seed always gives the same sequence, and the
# same sequence as the TypeScript version of mulberry32.
#
# Example:  rand = mulberry32(42);  rand() -> 0.601104…,  rand() -> 0.448291…
def mulberry32(seed: int) -> Callable[[], float]:
    MASK = 0xFFFFFFFF  # keep values to 32 bits, like JavaScript
    state = int(seed) & MASK

    def next_random() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK
        t = ((state ^ (state >> 15)) * (1 | state)) & MASK
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return next_random
