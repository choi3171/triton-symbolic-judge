"""A cap on how many terms a reduction contributes -- the other way to bound.

Two published approaches to the same problem bound it in opposite directions.
Gimlet Labs' tensor algebra equivalence checker keeps the declared shape and
truncates the summation ("if you have say 256 dot product terms ... we cap the
summation bound to say two or four ... so we cut the product terms"); this
project keeps the whole reduction and shrinks the shape, choosing shapes from
the kernel's own contract (`shapes.py`).

Neither dominates, and they are blind in different directions:

  truncated at the true shape   the ADDRESS arithmetic is evaluated at production
                                strides, so an indexing bug that only appears at
                                K = 256 is exposed in the first two terms.  What
                                is lost is the tail: a value defect anywhere past
                                the cap is invisible, as its authors say.

  whole at a small shape        every contribution is present, so a value defect
                                cannot hide -- but the strides are the small
                                shape's, and a bug that lives in the real ones is
                                gone with them.

So this is a MODE, not a competitor: with a cap set, this judge samples the other
point of the trade-off.  It is not sound in the way the uncapped path is.  A
truncated reduction corresponds to no real execution -- a small shape is a real
instance of the kernel's contract, a fragment of a large one is not -- so a
disagreement found here is a lead to be confirmed, not a verdict.
"""
CAP = None          # None = keep every term (the default, and the sound one)


def limit(n):
    """How many of `n` reduction terms to build."""
    return n if CAP is None else min(n, CAP)


class capped:
    """`with capped(4): ...` -- both the interpreter and the spec front-end read
    the same global, so the two sides are always truncated identically."""
    def __init__(self, n): self.n = n
    def __enter__(self):
        global CAP
        self.prev, CAP = CAP, self.n
        # a delegated matmul stands for the WHOLE product; with a cap in force it
        # would silently reintroduce the terms the cap is meant to remove
        from tvj.decide import delegate
        self.prev_exp, delegate.EXPAND_BELOW = delegate.EXPAND_BELOW, float("inf")
        return self
    def __exit__(self, *a):
        global CAP
        CAP = self.prev
        from tvj.decide import delegate
        delegate.EXPAND_BELOW = self.prev_exp
        return False
