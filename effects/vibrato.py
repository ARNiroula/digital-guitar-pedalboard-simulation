import math


def traingle_lfo(phase: float) -> float:
    # Normalize phase to be [0, 2*pi]
    normalized = phase % (2 * math.pi)

    # Triangle wave
    if normalized < math.pi:
        # Rising edge: -1 to +1
        return -1 + 2 * normalized / math.pi
    else:
        # Falling edge: +1 to -1
        return 3 - 2 * normalized / math.pi


def square_lfo(phase: float) -> float:
    # Normalize phase to be [0, 2*pi]
    normalized = phase % (2 * math.pi)
    return 1.0 if normalized < math.pi else -1.0
