"""Is our own cost an axis a generator can aim at?

The Limits table says yes, and this is the demonstration behind that cell.

Three formulations of attention -- naive, max-subtracted, online (flash) -- are
all equal over the reals, and the judge proves all three pairs equal when it is
given room.  What separates them is what it COSTS to prove: the pair whose two
sides share no common atom expands where the others do not.  Hold the reference
fixed and swap only the kernel, and the price of judging it moves by an order of
magnitude.

So at a cap between the cheap pair and the dear one, the same judge decides one
correct kernel and returns UNDECIDED on another.  That is the whole claim: a
kernel does not have to use an operation we cannot model to land outside the
judge's reach, it only has to be shaped so that canonicalising it is expensive --
and the expensive one here is also the FASTER one to run, which is why a policy
rewarded for speed drifts towards it without being told to.

A row that lands there is reported UNKNOWN, not FAIL.

    python3 -m tvj.measure.steerable [L] [headroom_gb]
"""
import math, os, sys, time
from tvj.core import terms as T
from tvj.core import ttir as P
from tvj.core import sexec as X
from tvj.decide import volta_bridge as V
from tvj.fixtures import attn
from tvj.checks.check import to_ttir

L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
HEADROOM = int(sys.argv[2]) if len(sys.argv) > 2 else 3     # the cap for the measuring pass
D, SPREAD = 16, 5.0     # SPREAD: how much dearer the worst pair must be to call it an axis

ASIG = {"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32", "o_ptr": "*fp32",
        "L": "i32", "D": "i32", "BM": "constexpr", "BD": "constexpr", "BL": "constexpr"}
FSIG = {**{k: v for k, v in ASIG.items() if k != "BL"}, "BN": "constexpr"}
PAIRS = [("ref", "safe"), ("safe", "flash"), ("ref", "flash")]


def build():
    bufs = {"q_ptr": L*D, "k_ptr": L*D, "v_ptr": L*D, "o_ptr": L*D}
    args = [X.Ptr("q_ptr", 0), X.Ptr("k_ptr", 0), X.Ptr("v_ptr", 0), X.Ptr("o_ptr", 0), L, D]
    out = {}
    for name, fn, sig, cst in (("ref",   attn.attn_ref,   ASIG, {"BM": 16, "BD": 16, "BL": L}),
                               ("safe",  attn.attn_safe,  ASIG, {"BM": 16, "BD": 16, "BL": L}),
                               ("flash", attn.attn_flash, FSIG, {"BM": 16, "BD": 16, "BN": 16})):
        f = P.parse(to_ttir(fn, sig, cst))
        it = X.Interp(f, None, (L // 16,), bufs); it.argvals = args
        it.run_all(); out[name] = it.g.store
    return out


def decide(stores, a, b, cap):
    """(verdict, peak GB, seconds).  verdict: 'equal', 'not equal', or None for
    'the cap stopped it' -- which is the outcome this script exists to produce."""
    os.environ["VOLTA_MEM_GB"] = str(cap)
    keys = sorted(stores["ref"])
    t0 = time.time()
    try:
        res, st = V.equivalent([(stores[a][k], stores[b][k]) for k in keys])
    except V.Unsupported:
        return None, None, time.time() - t0
    ok = sum(r is True for r in res)
    return ("equal" if ok == len(keys) else f"{ok}/{len(keys)} equal"), \
           st["peak_rss_mb"] / 1024, time.time() - t0


if __name__ == "__main__":
    stores = build()
    n = len(stores["ref"])
    print(f"attention L={L} D={D}, BM=16, flash BN=16 -- {n} outputs, three formulations\n")

    print(f"what it costs to decide each pair, with {HEADROOM} GB of room")
    peak = {}
    for a, b in PAIRS:
        v, gb, dt = decide(stores, a, b, HEADROOM)
        if v is None:
            sys.exit(f"steerable: {a} vs {b} did not fit in {HEADROOM} GB either, so there is no\n"
                     f"  cap between the pairs to demonstrate with.  Run at a smaller L, or give\n"
                     f"  more headroom: `python3 -m tvj.measure.steerable {L} {HEADROOM*4}`.")
        peak[(a, b)] = gb
        print(f"  {a:<5} vs {b:<5}  {gb:>5.2f} GB  {dt:>5.1f}s   {v}")

    lo, hi = min(peak.values()), max(peak.values())
    cheap = min(peak, key=peak.get); dear = max(peak, key=peak.get)
    print(f"\n  the dearest pair costs {hi/lo:.1f}x the cheapest, and every pair is equal over "
          f"the reals")
    print(f"  an order of magnitude between two CORRECT kernels against the same reference: "
          f"{'ok' if hi/lo >= SPREAD else 'NO'}")

    # Derived, not chosen: the smallest whole-GB cap that still holds the cheap pair.
    # Volta's cap is read as an integer number of gigabytes, so there is no finer one.
    cap = max(1, math.ceil(lo))
    if cap >= hi:
        sys.exit(f"\nsteerable: no whole-GB cap separates {lo:.2f} GB from {hi:.2f} GB at L={L}.\n"
                 f"  The spread is real but too narrow to demonstrate with an integer cap --\n"
                 f"  run at a larger L, where it widens: `python3 -m tvj.measure.steerable {L*2}`.")

    print(f"\nthe same three pairs at a {cap} GB cap -- between the two, derived from the row above")
    verdicts = {}
    for a, b in PAIRS:
        v, gb, dt = decide(stores, a, b, cap)
        verdicts[(a, b)] = v
        print(f"  {a:<5} vs {b:<5}  {dt:>5.1f}s   " +
              (f"decided, {v}" if v else "UNDECIDED -- the cap stopped it, so the verdict is UNKNOWN"))

    got, lost = [p for p in PAIRS if verdicts[p]], [p for p in PAIRS if not verdicts[p]]
    print(f"\n  at one cap the judge decides {len(got)} of these correct kernels and not {len(lost)}: "
          f"{'ok' if got and lost else 'NO'}")
    print(f"  what moved is the shape of the kernel, not the reference and not the cap: "
          f"{'ok' if verdicts[cheap] and not verdicts[dear] else 'NO'}")
