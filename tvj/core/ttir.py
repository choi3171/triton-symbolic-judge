"""Minimal parser for the textual TTIR that Triton emits.

A production tool would bind MLIR's own parser (as mlir-tv does); parsing text
keeps this prototype self-contained and makes the covered op surface explicit.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Op:
    name: str
    results: List[str]
    operands: List[str]
    attrs: dict
    rtype: str
    body: List["Op"] = field(default_factory=list)
    else_body: List["Op"] = field(default_factory=list)
    raw: str = ""

@dataclass
class Func:
    name: str
    args: List[str]
    argtypes: List[str]
    body: List[Op]

def split_top(s, sep=","):
    """Split on `sep` at bracket depth 0."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<([{": depth += 1
        elif ch in ">)]}": depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def shape_of(ty):
    m = re.match(r"^tensor<((?:\d+x)+)(.*)>$", ty.strip())
    if not m: return ()
    return tuple(int(d) for d in m.group(1).rstrip("x").split("x"))

def elem_of(ty):
    m = re.match(r"^tensor<((?:\d+x)+)(.*)>$", ty.strip())
    return m.group(2) if m else ty.strip()

def result_type(line):
    """Type of the op's (first) result."""
    body = line
    if "->" in body:
        return split_top(body.rsplit("->", 1)[1])[0].strip().strip("()")
    parts = split_top(body.rsplit(" : ", 1)[1]) if " : " in body else [""]
    ty = parts[0].strip()
    if " to " in ty: ty = ty.split(" to ", 1)[1].strip()      # arith.extsi %x : i32 to i64
    return ty

_ATTR = re.compile(r"(\w+)\s*=\s*([^,}]+)")

def parse_attrs(line):
    m = re.search(r"\{([^}]*)\}", line)
    if not m: return {}
    out = {}
    for k, v in _ATTR.findall(m.group(1)):
        v = v.strip()
        mm = re.match(r"^(-?\d+)\s*:", v)
        out[k] = int(mm.group(1)) if mm else v
    return out

def parse(text: str) -> Func:
    lines = [l.rstrip() for l in text.split("\n")]
    lines = [l for l in lines if l.strip() and not l.strip().startswith("#loc")]
    fn = None
    stack = []           # stack of op-lists we are appending into
    for raw in lines:
        s = raw.strip()
        if s.startswith("module"): continue
        if s.startswith("tt.func"):
            m = re.search(r"@(\w+)\((.*?)\)\s*attributes", s)
            argstr = m.group(2)
            names, types = [], []
            for a in split_top(argstr):
                am = re.match(r"%(\w+)\s*:\s*(.+?)(?:\s+loc\(.*)?$", a)
                names.append(am.group(1)); types.append(am.group(2).strip())
            fn = Func(m.group(1), names, types, [])
            stack = [fn.body]
            continue
        if s.startswith("}") and s.endswith("else {"):
            stack.pop()
            owner = stack[-1][-1]
            stack.append(owner.else_body)
            continue
        if s.startswith("}"):
            if len(stack) > 1:
                stack.pop()
                # a region-carrying op puts its result type on the closing line:
                #   }) : (tensor<4xf32>) -> f32
                if " : " in s or "->" in s:
                    owner = stack[-1][-1]
                    owner.rtype = result_type(s)
            continue
        if s.startswith("^bb"):
            names = re.findall(r"(%\w+)\s*:", s)
            stack[-1].append(Op("^block", names, [], {}, "", raw=s))
            continue
        # op line
        results, rest = [], s
        m = re.match(r"^(%[\w#:]+(?:\s*,\s*%[\w#:]+)*)\s*=\s*(.*)$", s)
        if m:
            lhs, rest = m.group(1), m.group(2)
            for r in split_top(lhs):
                r = r.strip()
                if ":" in r:                      # %acc:3  -> %acc#0..#2
                    base, n = r.split(":")
                    results += [f"{base}#{i}" for i in range(int(n))]
                else:
                    results.append(r)
        opname = re.match(r'"?([A-Za-z_][\w.]*)"?', rest).group(1)
        operands = re.findall(r"%[\w#]+", rest.split(" : ")[0] if " : " in rest else rest)
        op = Op(opname, results, operands, parse_attrs(rest), result_type(rest), raw=s)
        stack[-1].append(op)
        if s.endswith("{"):
            stack.append(op.body)
    return fn
