#!/bin/sh
# Fetch the third-party pieces this project reads but does not vendor.
set -e
cd "$(dirname "$0")"
. ./_python.sh

# Preflight.  Everything below assumes torch, triton, numpy, z3 and datasets are
# importable and that cargo is on PATH.  Without this the first sign of a missing
# dependency is a traceback out of a HuggingFace loader, two git clones and
# several minutes in, and the README's two-line quickstart never said to install
# anything at all.
"$PY" - <<'CHECK'
import importlib.util, sys
need = [("torch", "torch"), ("triton", "triton"), ("numpy", "numpy"),
        ("z3", "z3-solver"), ("datasets", "datasets")]
miss = [pkg for mod, pkg in need if importlib.util.find_spec(mod) is None]
if miss:
    sys.exit("setup: missing Python packages: " + " ".join(miss) +
             "\n  pip install -r requirements.txt"
             "\n  (see that file: torch has to be a CUDA build)")
import torch, triton
print(f"  torch {torch.__version__}  triton {triton.__version__}  cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("setup: no CUDA device.  The judge runs every candidate on the GPU: the"
             "\n  hardware gate that decides whether a FAIL is believed is not optional.")
cc = torch.cuda.get_device_capability()
print(f"  device {torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}")
if cc != (7, 5):
    print(f"  note: the [sm_75] claims were measured on Turing.  On sm_{cc[0]}{cc[1]} some of them"
          "\n  are expected to answer differently; verify.py reports those apart from the"
          "\n  count rather than as failures.")
CHECK
command -v cargo >/dev/null || {
  echo "setup: cargo is not on PATH -- the Volta bridge is Rust (see requirements.txt)"; exit 1; }
# The bridge depends on gmp-mpfr-sys, which builds GMP from source, and GMP's
# configure wants a C toolchain and m4.  Without m4 cargo gets all the way through
# fetching and compiling the Rust half before dying inside a shell script with
# "No usable m4 in $PATH", which reads like a Rust problem and is not one.
for tool in cc make m4; do
  command -v "$tool" >/dev/null || {
    echo "setup: $tool is not on PATH.  The Volta bridge builds GMP from source:"
    echo "  sudo apt install build-essential m4        # Debian/Ubuntu"; exit 1; }
done

# Volta's decision procedure (MIT).  We use only volta_analysis::canon; its PTX
# frontend is never invoked -- see PIPELINE.md.
[ -d ../volta ] || git clone https://github.com/willtunnels/volta.git ../volta
( cd bridge && cargo build --release )

mkdir -p data
[ -d data/KernelBench ] || git clone --depth 1 \
  https://github.com/ScalingIntelligence/KernelBench.git data/KernelBench

"$PY" - <<'PY'
import json, itertools, os
from datasets import load_dataset
if not os.path.exists("data/kernelbook_400.json"):
    ds = load_dataset("GPUMODE/KernelBook", split="train", streaming=True)
    keep = ["entry_point","module_name","python_code","triton_code","synthetic","repo_name"]
    json.dump([{k: r[k] for k in keep} for r in itertools.islice(ds, 400)],
              open("data/kernelbook_400.json","w"))
if not os.path.exists("data/triton_multiturn.json"):
    # the same author's MULTI-TURN traces: the model was told its kernel failed and
    # tried again, up to four times.  `num_turns` is how much selection pressure a
    # row survived, which is the closest thing to an adversary available without
    # running RL -- see `multiturn.py`.
    ds = load_dataset("ppbhatt500/kernelbook-triton-multiturn-reasoning-traces", split="train")
    rows = []
    for r in ds:
        fr = r["final_result"] or {}
        rows.append({"sample_key": r["sample_key"], "source": r["source"],
                     "pytorch_code": r["pytorch_code"], "triton_code": r["final_triton_code"],
                     "num_turns": r["num_turns"], "stop_reason": r["stop_reason"],
                     "result_correctness": bool(fr.get("correctness")),
                     "result_speedup": fr.get("speedup")})
    json.dump(rows, open("data/triton_multiturn.json", "w"))
if not os.path.exists("data/triton_traces.json"):
    ds = load_dataset("ppbhatt500/kernelbook-triton-reasoning-traces", split="train")
    keep = ["sample_key","source","level","name","problem_id","pytorch_code","triton_code",
            "result_correctness","result_speedup"]
    json.dump([{k: r[k] for k in keep} for r in ds], open("data/triton_traces.json","w"))
print("corpora ready")
PY

# KernelBench-Verified: its hidden_tests are read by kbv_blindspot.py
[ -d data/KBV ] || git clone --depth 1 \
  https://github.com/facebookresearch/kernel_bench_verified.git data/KBV
