def halton_sequence(base):
    """Returns a halton sequence for obstacle placement."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= base
        else:
            y = d // base
            while x <= y:
                y //= base
            n = (base + 1) * y - x
        yield n / d
