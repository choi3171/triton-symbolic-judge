"""Block arguments across a `cf` branch are bound.

`^bb1(%5: tensor<4xf32>)` DECLARES `%5`; no op assigns it.  The parser puts the
declaration in the block op's `results` and the branch's arguments in the raw
text, mixed in with the condition, so binding them is the interpreter's job.
Until it did, any block that carried a value died on a KeyError and the row was
reported as ERROR -- a judge bug wearing a candidate's clothes.

Triton 3.x keeps structured control flow at the TTIR stage, so the corpora only
exercise the argument-LESS form (an early `return` lowers to `cf.cond_br %c,
^bb1, ^bb2`).  The carrying form is written out by hand here rather than left
untested because it cannot yet be reached from a corpus row.
"""
import sys
from tvj.core import ttir as P, sexec as X, terms as T

HEAD = """module {
  tt.func public @carry(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %two = arith.constant dense<2.000000e+00> : tensor<4xf32>
    %three = arith.constant dense<3.000000e+00> : tensor<4xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %pid = tt.get_program_id x : i32
    %p = arith.cmpi sgt, %pid, %c0_i32 : i32
"""
TAIL = """  ^bb3(%s: tensor<4xf32>):
    %base = arith.muli %pid, %c4_i32 : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %bs = tt.splat %base : i32 -> tensor<4xi32>
    %o = arith.addi %bs, %r : tensor<4xi32>
    %xp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %o : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %xa : tensor<4x!tt.ptr<f32>>
    %m = arith.mulf %v, %s : tensor<4xf32>
    %yp = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ya = tt.addptr %yp, %o : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %ya, %m : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}"""

# the true branch scales by 2, the false branch by 3, and the choice is carried
# to the merge block as an argument -- nothing else distinguishes the two paths
CARRY = HEAD + """    cf.cond_br %p, ^bb1(%two : tensor<4xf32>), ^bb2(%three : tensor<4xf32>)
  ^bb1(%a: tensor<4xf32>):
    cf.br ^bb3(%a : tensor<4xf32>)
  ^bb2(%b: tensor<4xf32>):
    cf.br ^bb3(%b : tensor<4xf32>)
""" + TAIL

# the branch passes nothing where the block declares one parameter
MISMATCH = HEAD + """    cf.cond_br %p, ^bb1(%two : tensor<4xf32>), ^bb2(%three : tensor<4xf32>)
  ^bb1(%a: tensor<4xf32>):
    cf.br ^bb3
  ^bb2(%b: tensor<4xf32>):
    cf.br ^bb3(%b : tensor<4xf32>)
""" + TAIL


def run(src):
    T.reset()
    f = P.parse(src)
    it = X.Interp(f, None, (2,), {"x_ptr": 8, "y_ptr": 8})
    it.argvals = [X.Ptr("x_ptr", 0), X.Ptr("y_ptr", 0)]
    g = it.run_all()
    return {k[1]: repr(v) for k, v in g.store.items() if k[0] == "y_ptr"}


if __name__ == "__main__":
    bad = 0
    print("block arguments carried across cf.br / cf.cond_br")
    try:
        got = run(CARRY)
    except Exception as e:
        print(f"  carry            {type(e).__name__}: {str(e)[:70]}"); sys.exit(1)
    # program 0 takes the false edge (scale 3), program 1 the true edge (scale 2)
    want = {**{i: f"(3.0*x_ptr[{i}])" for i in range(4)},
            **{i: f"(2.0*x_ptr[{i}])" for i in range(4, 8)}}
    for i in range(8):
        ok = got.get(i) == want[i]
        bad += not ok
        print(f"  y[{i}]  {got.get(i, '<unwritten>'):<22} want {want[i]:<22} {'ok' if ok else 'WRONG'}")

    print("\na branch whose argument count disagrees with the block is refused")
    try:
        run(MISMATCH); print("  MISMATCH DID NOT REFUSE"); bad += 1
    except X.Unsupported as e:
        print(f"  refused: {str(e)[:70]}")
    except Exception as e:
        print(f"  wrong error {type(e).__name__}: {str(e)[:60]}"); bad += 1

    print(f"\n{'8/8 carried values correct, 1/1 refusal' if not bad else f'{bad} problem(s)'}")
    sys.exit(1 if bad else 0)
