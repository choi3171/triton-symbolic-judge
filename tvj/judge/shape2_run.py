"""The LLM corpus judged a second time, at a shape with at least two blocks and a tail.

The first input of every row whose get_inputs() shape could be read (155 of 155)
has at most 1024 elements, and 95 of the 156 kernels take the element count at
runtime.  So most PASS rows in results/triton_traces.jsonl are proofs at one
partial block; for a 1-D kernel with BLOCK >= 1024 the grid is 1, pid is 0 in
every program, and the pid arithmetic is not exercised.  This run tiles every tensor input's leading
dimension to the smallest odd m with m * inner > 2048 -- for [4,4,4,4] that is
m = 33, numel 2112 = 8 * 256 + 64, at least two blocks and a tail for any BLOCK
from 128 to 2048 -- and judges the same kernel against the same reference there.

    ./run_shape2.sh                      (resumes; one process; results/tr_shape2.log)
    python3 -m tvj.judge.shape2_run --smoke 0,5      two rows, nothing recorded

Records go to record.scratch("traces_shape2"), published like any corpus.  Per
row: k, numel2, shapes2, and the judge's usual fields including `grids`.  Rows
the reference itself rejects at the new shape (a hard-coded view, a batch baked
into __init__) are SHAPE2-NA: nothing to hold the kernel to.  The four rows that
took more than 10 s at the corpus shape are SHAPE2-SKIPPED: at 33x they are the
alarm and, for row 92's 418k-node spec term, the machine.

Safety, in order: the reference is tried on a CPU copy before any kernel runs at
the new shape (a device-side assert is unrecoverable); a watchdog thread ends the
process at TVJ_MEM_CAP_GB of RSS (default 5) after recording MEM-CAP for the row;
each row is under the judge's alarm; a STARTED line is written before a row and
a row still STARTED on resume is recorded HANG and stepped over; the process
retires itself after 1200 s so the shell's timeout is never what stops it.
"""
import copy, json, os, signal, sys, threading, time
import torch
from tvj.judge import record
from tvj.judge.traces_run import build, ROW_TIMEOUT
from tvj.judge.judge import judge, Timeout

CORPUS = "traces_shape2"
ROWS = "data/triton_traces.json"
HEAVY = {128: "84 s at the corpus shape", 51: "52 s", 92: "29 s, a 418k-node spec term", 129: "12 s"}
MEM_CAP = float(os.environ.get("TVJ_MEM_CAP_GB", 5)) * 2**30
RETIRE_AFTER = 1200


def leading(inner):
    """The new leading dimension: the smallest ODD m with m * inner > 2048, where
    inner is the element count below the leading dim.  Odd matters: tiling the
    original leading dim of 4 gives multiples of 4 * inner = 256 for [4,4,4,4],
    and a BLOCK of 256 then sees no tail.  With m odd, m * inner is not a multiple
    of any BLOCK above inner: [4,4,4,4] -> m = 33, numel 2112 = 8 * 256 + 64."""
    m = 3
    while m * inner <= 2048: m += 2
    return m


def scale_leading(inputs, m):
    """Set every tensor input's leading dimension to m by repeating its first
    slice.  All 156 rows agree on the leading dim across their tensor inputs
    (checked), so this is the batch.  Values repeat, which is irrelevant: the
    judge symbolises the reals and draws every probe from the shapes; integer
    inputs keep their first row's in-domain values."""
    return [x[:1].repeat((m,) + (1,) * (x.dim() - 1)).contiguous()
            if torch.is_tensor(x) and x.dim() >= 1 else x for x in inputs]


def reference_accepts(model, xs):
    """Run the reference on a CPU copy at the new shape.  A device-side assert
    on CUDA would poison the context for every row after this one."""
    try:
        cpu = copy.deepcopy(model).cpu()
        with torch.no_grad():
            cpu(*[x.detach().cpu() if torch.is_tensor(x) else x for x in xs])
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:100]}"


def judge_row(r, i):
    info = {}
    def transform(xs):
        t0 = next((x for x in xs if torch.is_tensor(x) and x.dim() >= 1), None)
        if t0 is None: return xs
        info["k"] = k = leading(t0.numel() // t0.shape[0])       # k is the new leading dim
        ys = scale_leading(xs, k)
        info["numel2"] = next(y for y in ys if torch.is_tensor(y)).numel()
        info["shapes2"] = [list(y.shape) for y in ys if torch.is_tensor(y)]
        return ys
    base = {"key": r["sample_key"], "label": bool(r["result_correctness"])}
    signal.alarm(60)
    try:
        cand, bucket = build(r, transform=transform)
    except Timeout:
        return dict(base, verdict="TIMEOUT", reason="building the candidate")
    except Exception as e:
        return dict(base, verdict="ERROR", reason=f"{type(e).__name__}: {str(e)[:80]}")
    finally:
        signal.alarm(0)
    if cand is None: return dict(bucket, **info)
    why = reference_accepts(cand.model, cand.inputs)
    if why: return dict(base, **info, verdict="SHAPE2-NA", reason=why)
    rec = judge(cand, timeout=ROW_TIMEOUT)
    rec.update(info)
    return rec


def _rss():
    with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def watchdog(path, current):
    def run():
        while True:
            time.sleep(0.5)
            if _rss() > MEM_CAP:
                with open(path, "a") as f:
                    f.write(json.dumps({"i": current["i"], "key": current.get("key"), "verdict": "MEM-CAP",
                                        "reason": f"RSS above {MEM_CAP / 2**30:.0f} GB"}) + "\n")
                os._exit(3)
    threading.Thread(target=run, daemon=True).start()


def last_verdicts(path):
    seen = {}
    if os.path.exists(path):
        for ln in open(path):
            if ln.strip():
                rec = json.loads(ln); seen[rec["i"]] = rec["verdict"]
    return seen


def line(i, rec):
    return (f"[{i:3d}] {rec['verdict']:<18} label={str(rec.get('label')):<5} tol={str(rec.get('tol')):<5} "
            f"k={rec.get('k', '-'):<3} numel2={rec.get('numel2', '-'):<6} grids={str(rec.get('grids', ''))[:24]:<24} "
            f"{rec.get('model', '?')[:20]:<20} {str(rec.get('reason', rec.get('prec', '')))[:56]}")


if __name__ == "__main__":
    rows = [r for r in json.load(open(ROWS)) if r["source"] == "kernelbook"]
    t_start = time.time()
    if len(sys.argv) > 2 and sys.argv[1] == "--smoke":
        for i in (int(x) for x in sys.argv[2].split(",")):
            rec = judge_row(rows[i], i); print(line(i, rec), flush=True)
        sys.exit(0)
    path = record.scratch(CORPUS)
    seen = last_verdicts(path)
    current = {"i": -1}
    watchdog(path, current)
    out = open(path, "a")
    def emit(rec): out.write(json.dumps(rec) + "\n"); out.flush()
    for i, r in enumerate(rows):
        if seen.get(i) == "STARTED":
            emit({"i": i, "key": r["sample_key"], "verdict": "HANG", "reason": "no record after STARTED: killed or hung"})
            print(f"[{i:3d}] HANG               stepped over", flush=True); continue
        if i in seen: continue
        if i in HEAVY:
            emit({"i": i, "key": r["sample_key"], "verdict": "SHAPE2-SKIPPED", "reason": HEAVY[i]}); continue
        if time.time() - t_start > RETIRE_AFTER:
            print(f"retiring after {RETIRE_AFTER} s; next row {i}", flush=True); sys.exit(4)
        current.update(i=i, key=r["sample_key"])
        emit({"i": i, "key": r["sample_key"], "verdict": "STARTED"})
        rec = judge_row(r, i); rec["i"] = i
        emit(rec); print(line(i, rec), flush=True)
        reason = rec.get("reason") or ""
        if "device-side assert" in reason or "CUDA error" in reason or "illegal memory access" in reason:
            print(f"!! row {i} poisoned the CUDA context -- exiting so the shell restarts", flush=True); sys.exit(2)
    print("complete", flush=True); sys.exit(0)
