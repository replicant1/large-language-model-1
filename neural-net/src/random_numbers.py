class Mulberry32:
    """A seedable random number generator. Calling it gives a new number in the range 0 to 1
    each time. The same seed always gives the same sequence, and the same sequence as the
    TypeScript version of mulberry32.

    Example:  rand = Mulberry32(42);  rand() -> 0.601104…,  rand() -> 0.448291…
    """

    MASK = 0xFFFFFFFF  # keep values to 32 bits, like JavaScript

    def __init__(self, seed: int):
        self.state = int(seed) & Mulberry32.MASK

    # The next number in the sequence, from 0 up to (but not including) 1
    def __call__(self) -> float:
        MASK = Mulberry32.MASK
        self.state = (self.state + 0x6D2B79F5) & MASK
        t = ((self.state ^ (self.state >> 15)) * (1 | self.state)) & MASK
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296
