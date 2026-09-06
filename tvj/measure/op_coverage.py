"""How many op definitions buy how much of the corpus?

The reference side of this project is PyTorch, and it is PyTorch for one reason:
coverage.  Nothing else spans the ops a corpus module is actually written in.
The price is that the baseline is an *implementation*, not a definition -- so
"is this kernel correct" reduces to "does it agree with whatever torch does",
and every one-sided obligation inherits torch's numerical choices.

The alternative is to write the reference as a definition in the term algebra
(`spec_test.py` already does this for mm / softmax / attention, as test fixtures).
That buys a clean semantics and loses coverage.  The trade is only worth arguing
about with the actual distribution in hand, so: for each reference module in both
corpora, which torch ops does it use, and what fraction of modules is fully
covered by the K most common ops?

Static extraction -- `F.x(`, `torch.x(`, `nn.X(`, `.x(` -- so it over-counts
method names that are not ops and under-counts ops reached indirectly.  The shape
of the curve is the point, not the third digit.
"""
import collections, json, re, sys

# nn.Module -> the functional op its forward is defined by
NN = {
    "Linear": "linear", "Conv1d": "conv1d", "Conv2d": "conv2d", "Conv3d": "conv3d",
    "ConvTranspose2d": "conv_transpose2d", "BatchNorm1d": "batch_norm",
    "BatchNorm2d": "batch_norm", "BatchNorm3d": "batch_norm", "LayerNorm": "layer_norm",
    "GroupNorm": "group_norm", "InstanceNorm2d": "instance_norm", "ReLU": "relu",
    "ReLU6": "relu6", "LeakyReLU": "leaky_relu", "ELU": "elu", "GELU": "gelu",
    "SiLU": "silu", "Mish": "mish", "Sigmoid": "sigmoid", "Tanh": "tanh",
    "Softmax": "softmax", "LogSoftmax": "log_softmax", "Softplus": "softplus",
    "Hardtanh": "hardtanh", "Hardsigmoid": "hardsigmoid", "Hardswish": "hardswish",
    "PReLU": "prelu", "MaxPool1d": "max_pool1d", "MaxPool2d": "max_pool2d",
    "MaxPool3d": "max_pool3d", "AvgPool1d": "avg_pool1d", "AvgPool2d": "avg_pool2d",
    "AvgPool3d": "avg_pool3d", "AdaptiveAvgPool1d": "adaptive_avg_pool1d",
    "AdaptiveAvgPool2d": "adaptive_avg_pool2d", "AdaptiveMaxPool2d": "adaptive_max_pool2d",
    "Dropout": "dropout", "Dropout2d": "dropout", "Embedding": "embedding",
    "MSELoss": "mse_loss", "L1Loss": "l1_loss", "CrossEntropyLoss": "cross_entropy",
    "BCELoss": "binary_cross_entropy", "BCEWithLogitsLoss": "binary_cross_entropy_with_logits",
    "SmoothL1Loss": "smooth_l1_loss", "HuberLoss": "huber_loss", "Flatten": "flatten",
    "Identity": "clone", "Upsample": "interpolate", "Unfold": "unfold", "Softmin": "softmin",
}
# method / operator names that are not ops we would have to define
NOISE = {"super", "self", "append", "format", "join", "size", "item", "to", "cuda",
         "cpu", "float", "type", "device", "shape", "range", "len", "int", "str",
         "print", "isinstance", "getattr", "setattr", "named_parameters", "parameters",
         "state_dict", "eval", "train", "modules", "children", "requires_grad_",
         "manual_seed", "no_grad", "Parameter", "Module", "ModuleList", "Sequential",
         "randn", "rand", "zeros", "ones", "empty", "tensor", "arange", "Tensor",
         "get_inputs", "get_init_inputs", "forward", "__init__", "detach", "numpy",
         "contiguous", "data", "grad", "backward", "step", "zero_", "clone"}

# `self.fc1(x)` is a submodule call, not an op: the receiver has to be captured
# so those can be dropped, or the vocabulary fills up with attribute names.
PAT = re.compile(r"(\w+)?\s*\.\s*([A-Za-z_]\w*)\s*\(|\btorch\.([A-Za-z_]\w*)\s*\(")
QUALIFIED = re.compile(r"\b(?:F|functional|torch\.nn\.functional|torch|nn|linalg)\.([A-Za-z_]\w*)\s*\(")


def forwards(src):
    """Only the forward bodies matter: `__init__` and `reset_parameters` run
    before the judge ever symbolises anything, so their ops are not a coverage
    requirement.  Counting them was what made `uniform_` look important."""
    out, lines = [], src.split("\n")
    for i, l in enumerate(lines):
        m = re.match(r"(\s*)def forward\b", l)
        if not m: continue
        ind, body = len(m.group(1)), []
        for l2 in lines[i + 1:]:
            if l2.strip() and len(l2) - len(l2.lstrip()) <= ind: break
            body.append(l2)
        out.append("\n".join(body))
    return "\n".join(out)


def ops_of(whole):
    src = forwards(whole) or whole
    out, sub = set(), set(re.findall(r"self\.(\w+)\s*=", whole))
    for m in QUALIFIED.finditer(src):
        n = m.group(1)
        if n in NN: out.add(NN[n])
        elif n not in NOISE and not n.startswith("_"): out.add(n)
    for m in re.finditer(r"(\w+)\.([A-Za-z_]\w*)\s*\(", src):
        recv, n = m.group(1), m.group(2)
        if recv in ("F", "torch", "nn", "functional", "linalg", "self"): continue
        if n in sub or n in NOISE or n.startswith("_") or n[0].isupper(): continue
        out.add(NN.get(n, n))
    # `self.<attr>(...)` where <attr> is an nn.Module assigned in __init__ is a
    # submodule: resolve it to the op its class defines, when we can see the class
    for attr in sub:
        m = re.search(rf"self\.{attr}\s*=\s*(?:torch\.)?nn\.(\w+)", whole)
        if m and m.group(1) in NN and re.search(rf"self\.{attr}\s*\(", src):
            out.add(NN[m.group(1)])
    return out


def load():
    rows = []
    for r in json.load(open("data/kernelbook_400.json")):
        rows.append(("kb", r["python_code"]))
    for r in json.load(open("data/triton_traces.json")):
        if r.get("source") == "kernelbook": rows.append(("tr", r["pytorch_code"]))
    return rows


if __name__ == "__main__":
    rows = load()
    per = [(c, ops_of(s)) for c, s in rows]
    freq = collections.Counter(o for _, ops in per for o in ops)
    n = len(per)
    print(f"{n} reference modules, {len(freq)} distinct ops\n")
    print("the 25 most common ops")
    for o, k in freq.most_common(25): print(f"  {o:<26} {k:4d}  {100*k/n:5.1f}% of modules")

    print("\nmodules FULLY covered by the K most common ops")
    order = [o for o, _ in freq.most_common()]
    print(f"  {'K':>4}  {'covered':>8}  {'share':>7}")
    for K in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 131, 150, 200, len(order)):
        if K > len(order): continue
        top = set(order[:K])
        c = sum(1 for _, ops in per if ops <= top)
        print(f"  {K:>4}  {c:>8}  {100*c/n:6.1f}%")

    # what the long tail costs: ops that appear in exactly one module
    once = [o for o, k in freq.items() if k == 1]
    print(f"\nops appearing in exactly one module: {len(once)} of {len(freq)} "
          f"({100*len(once)/len(freq):.0f}% of the vocabulary, "
          f"{100*sum(1 for _, ops in per if ops & set(once))/n:.0f}% of the modules touch one)")

    from tvj.front import spec
    have = set(spec._TORCH) | {"relu6", "silu", "mish", "softplus", "sigmoid", "tanh"}
    miss = collections.Counter({o: k for o, k in freq.items() if o not in have})
    blocked = sum(1 for _, ops in per if ops - have)
    print(f"\nwith the {len(spec._TORCH)} handlers we have: {n - blocked} modules fully covered "
          f"({100*(n-blocked)/n:.0f}%), {blocked} blocked by at least one missing op")
    print("  most valuable missing ops (modules unblocked if added alone):")
    for o, k in miss.most_common(12):
        gain = sum(1 for _, ops in per if (ops - have) == {o})
        print(f"    {o:<26} appears in {k:3d}, would fully unblock {gain:3d}")
