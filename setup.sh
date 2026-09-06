#!/bin/sh
# Fetch the third-party pieces this project reads but does not vendor.
set -e
cd "$(dirname "$0")"

# Volta's decision procedure (MIT).  We use only volta_analysis::canon; its PTX
# frontend is never invoked -- see PIPELINE.md.
[ -d ../volta ] || git clone https://github.com/willtunnels/volta.git ../volta
( cd bridge && cargo build --release )

mkdir -p data
[ -d data/KernelBench ] || git clone --depth 1 \
  https://github.com/ScalingIntelligence/KernelBench.git data/KernelBench

python3 - <<'PY'
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
