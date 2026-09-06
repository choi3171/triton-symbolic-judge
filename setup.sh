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
if not os.path.exists("data/triton_traces.json"):
    ds = load_dataset("ppbhatt500/kernelbook-triton-reasoning-traces", split="train")
    keep = ["sample_key","source","level","name","problem_id","pytorch_code","triton_code",
            "result_correctness","result_speedup"]
    json.dump([{k: r[k] for k in keep} for r in ds], open("data/triton_traces.json","w"))
print("corpora ready")
PY
