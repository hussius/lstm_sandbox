import numpy as np

from data import generate_sequence


def test_generate_sequence():
    sequence = generate_sequence(5, 50)
    assert(len(sequence) == 5)
    assert(np.array([s >= 0 for s in sequence]).all())
    assert(np.array([s < 50 for s in sequence]).all())
