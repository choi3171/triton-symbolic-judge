//! Bridge: feed term DAGs produced by the TTIR symbolic executor to Volta's
//! decision procedure (`volta_analysis::canon::Session`) and report equality
//! over the reals.  Volta's frontend is PTX; this bypasses it and uses only
//! the canonicalizer, so every verdict here is Volta's, not a reimplementation.
//!
//! The input is a binary node stream (`tvj/decide/volta_bridge.py` writes it),
//! decoded in one pass straight into the arena.  It used to arrive as JSON and be
//! deserialised into a `Vec<Node>` first: at a measured 245 bytes of Python dict
//! and 80 bytes of JSON text per node, that hop cost more than the term graph it
//! described, and its Rust half came out of the same address-space cap this
//! process canonicalises under.
use serde::Serialize;
use std::io::Read;
use std::time::Instant;
use volta_analysis::canon::Session;
use volta_analysis::symbolic::{ExprArena, ExprId};

const MAGIC: &[u8; 4] = b"TVJB";
const WIRE_VERSION: u8 = 1;

/// A read cursor over the message.  Every `take` is bounds-checked by the slice
/// index, so a truncated or malformed stream panics here rather than being
/// half-interpreted into an arena and decided.
struct Cur<'a> { b: &'a [u8], i: usize }

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        s
    }
    fn u8(&mut self) -> u8 { let v = self.b[self.i]; self.i += 1; v }
    fn u32(&mut self) -> u32 { u32::from_le_bytes(self.take(4).try_into().unwrap()) }
    fn u64(&mut self) -> u64 { u64::from_le_bytes(self.take(8).try_into().unwrap()) }
    fn i64(&mut self) -> i64 { i64::from_le_bytes(self.take(8).try_into().unwrap()) }
    fn f64(&mut self) -> f64 { f64::from_le_bytes(self.take(8).try_into().unwrap()) }
}

#[derive(Serialize)]
struct Output {
    results: Vec<String>,   // "true" | "false" | "error: ..."
    ops_used: u64,
    interned_terms: usize,
    secs: f64,
    peak_rss_mb: f64,
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:"))
            .and_then(|l| l.split_whitespace().nth(1)).and_then(|k| k.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0).unwrap_or(-1.0)
}

/// One side: a string table, then the node stream.  Nodes are children-first, so
/// an operand index always names a node already built and one pass suffices.
fn build(c: &mut Cur) -> (ExprArena, Vec<ExprId>) {
    let mut ar = ExprArena::new();
    let n_str = c.u32() as usize;
    let mut strings = Vec::with_capacity(n_str);
    for _ in 0..n_str {
        let len = c.u32() as usize;
        let s = std::str::from_utf8(c.take(len)).expect("bad utf-8 in the string table");
        strings.push(ar.intern_string(s.to_string()));
    }
    let n = c.u32() as usize;
    let nbytes = c.u32() as usize;
    let start = c.i;
    let mut ids: Vec<ExprId> = Vec::with_capacity(n);
    for _ in 0..n {
        let op = c.u8();
        let id = match op {
            0 => { let s = c.u32() as usize; let idx = c.u64(); ar.input_element(strings[s], idx) }
            1 => { let v = c.f64(); ar.float_from_f64(v).expect("NaN constant") }
            2 => {
                let num = c.i64(); let den = c.i64();
                ar.real(volta_analysis::symbolic::Real::from_rational(rug::Rational::from((num, den))))
            }
            3..=9 | 16..=21 => {
                let a = ids[c.u32() as usize];
                let b = ids[c.u32() as usize];
                match op {
                    3 => ar.add(a, b), 4 => ar.mul(a, b), 5 => ar.div(a, b),
                    6 => ar.max(a, b), 7 => ar.min(a, b), 8 => ar.and(a, b), 9 => ar.or(a, b),
                    16 => ar.lt(a, b), 17 => ar.le(a, b), 18 => ar.gt(a, b),
                    19 => ar.ge(a, b), 20 => ar.eq(a, b), 21 => ar.ne(a, b),
                    _ => unreachable!(),
                }
            }
            10..=14 => {
                let a = ids[c.u32() as usize];
                match op {
                    10 => ar.exp(a), 11 => ar.sqrt(a), 12 => ar.log(a),
                    13 => ar.abs(a), 14 => ar.not(a), _ => unreachable!(),
                }
            }
            15 => {
                let cond = ids[c.u32() as usize];
                let t = ids[c.u32() as usize];
                let f = ids[c.u32() as usize];
                ar.select(cond, t, f)
            }
            other => panic!("unknown wire op {other}"),
        };
        ids.push(id);
    }
    assert_eq!(c.i - start, nbytes, "node stream is {} bytes, header says {}", c.i - start, nbytes);
    (ar, ids)
}

/// Volta's budget counts term operations, not bytes, and canonicalising two
/// differently-shaped kernels can reach tens of GB before it trips.  An address
/// space cap turns that into a clean allocation failure in this child process
/// instead of an OOM kill that takes the session with it.
fn cap_address_space(gb: u64) {
    unsafe {
        let bytes = gb * 1024 * 1024 * 1024;
        let lim = libc::rlimit { rlim_cur: bytes, rlim_max: bytes };
        libc::setrlimit(libc::RLIMIT_AS, &lim);
    }
}

fn main() {
    let cap: u64 = std::env::var("VOLTA_MEM_GB").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    cap_address_space(cap);
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let mut c = Cur { b: &buf, i: 0 };
    assert_eq!(c.take(4), MAGIC, "input is not a tvj bridge stream");
    let v = c.u8();
    assert_eq!(v, WIRE_VERSION, "stream is wire version {v}; this binary speaks {WIRE_VERSION}");
    let has_budget = c.u8();
    let budget = c.u64();

    let t0 = Instant::now();
    let (ar_a, ids_a) = build(&mut c);
    let (ar_b, ids_b) = build(&mut c);
    let n_pairs = c.u32() as usize;
    let mut sess = if has_budget != 0 { Session::with_budget(budget) } else { Session::new() };
    let mut results = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let ia = c.u32() as usize;
        let ib = c.u32() as usize;
        let r = sess.check_equivalent(&ar_a, ids_a[ia], &ar_b, ids_b[ib]);
        results.push(match r {
            Ok(true) => "true".to_string(),
            Ok(false) => "false".to_string(),
            Err(e) => format!("error: {:?}", e),
        });
    }
    let out = Output {
        results,
        ops_used: sess.ops_used(),
        interned_terms: sess.interned_terms(),
        secs: t0.elapsed().as_secs_f64(),
        peak_rss_mb: peak_rss_mb(),
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
