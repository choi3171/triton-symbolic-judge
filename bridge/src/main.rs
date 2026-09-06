//! Bridge: feed term DAGs produced by the TTIR symbolic executor to Volta's
//! decision procedure (`volta_analysis::canon::Session`) and report equality
//! over the reals.  Volta's frontend is PTX; this bypasses it and uses only
//! the canonicalizer, so every verdict here is Volta's, not a reimplementation.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;
use volta_analysis::canon::Session;
use volta_analysis::symbolic::{ExprArena, ExprId};

#[derive(Deserialize)]
#[serde(tag = "op")]
enum Node {
    #[serde(rename = "sym")]   Sym { buf: String, idx: u64 },
    #[serde(rename = "const")] Const { v: f64 },
    #[serde(rename = "rat")]   Rat { num: i64, den: i64 },
    #[serde(rename = "add")]   Add { a: usize, b: usize },
    #[serde(rename = "mul")]   Mul { a: usize, b: usize },
    #[serde(rename = "div")]   Div { a: usize, b: usize },
    #[serde(rename = "max")]   Max { a: usize, b: usize },
    #[serde(rename = "min")]   Min { a: usize, b: usize },
    #[serde(rename = "exp")]   Exp { a: usize },
    #[serde(rename = "sqrt")]  Sqrt { a: usize },
    #[serde(rename = "log")]   Log { a: usize },
    #[serde(rename = "abs")]   Abs { a: usize },
    #[serde(rename = "select")] Select { c: usize, t: usize, f: usize },
    #[serde(rename = "cmp")]   Cmp { kind: String, a: usize, b: usize },
    #[serde(rename = "not")]   Not { a: usize },
    #[serde(rename = "and")]   And { a: usize, b: usize },
    #[serde(rename = "or")]    Or { a: usize, b: usize },
}

#[derive(Deserialize)]
struct Input {
    nodes_a: Vec<Node>,
    nodes_b: Vec<Node>,
    pairs: Vec<(usize, usize)>,
    budget: Option<u64>,
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

fn build(nodes: &[Node]) -> (ExprArena, Vec<ExprId>) {
    let mut ar = ExprArena::new();
    let mut ids: Vec<ExprId> = Vec::with_capacity(nodes.len());
    let mut strings: HashMap<String, _> = HashMap::new();
    for n in nodes {
        let id = match n {
            Node::Sym { buf, idx } => {
                let s = match strings.get(buf) {
                    Some(s) => *s,
                    None => { let s = ar.intern_string(buf.clone()); strings.insert(buf.clone(), s); s }
                };
                ar.input_element(s, *idx)
            }
            Node::Const { v } => ar.float_from_f64(*v).expect("NaN constant"),
            Node::Rat { num, den } => ar.real(volta_analysis::symbolic::Real::from_rational(rug::Rational::from((*num, *den)))),
            Node::Add { a, b } => ar.add(ids[*a], ids[*b]),
            Node::Mul { a, b } => ar.mul(ids[*a], ids[*b]),
            Node::Div { a, b } => ar.div(ids[*a], ids[*b]),
            Node::Max { a, b } => ar.max(ids[*a], ids[*b]),
            Node::Min { a, b } => ar.min(ids[*a], ids[*b]),
            Node::Exp { a }    => ar.exp(ids[*a]),
            Node::Sqrt { a }   => ar.sqrt(ids[*a]),
            Node::Log { a }    => ar.log(ids[*a]),
            Node::Abs { a }    => ar.abs(ids[*a]),
            Node::Select { c, t, f } => ar.select(ids[*c], ids[*t], ids[*f]),
            Node::Not { a }    => ar.not(ids[*a]),
            Node::And { a, b } => ar.and(ids[*a], ids[*b]),
            Node::Or  { a, b } => ar.or(ids[*a], ids[*b]),
            Node::Cmp { kind, a, b } => match kind.as_str() {
                "lt" => ar.lt(ids[*a], ids[*b]), "le" => ar.le(ids[*a], ids[*b]),
                "gt" => ar.gt(ids[*a], ids[*b]), "ge" => ar.ge(ids[*a], ids[*b]),
                "eq" => ar.eq(ids[*a], ids[*b]), "ne" => ar.ne(ids[*a], ids[*b]),
                other => panic!("unknown comparison {other}"),
            },
        };
        ids.push(id);
    }
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
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let inp: Input = serde_json::from_str(&s).expect("bad input json");
    let t0 = Instant::now();
    let (ar_a, ids_a) = build(&inp.nodes_a);
    let (ar_b, ids_b) = build(&inp.nodes_b);
    let mut sess = match inp.budget { Some(b) => Session::with_budget(b), None => Session::new() };
    let mut results = Vec::with_capacity(inp.pairs.len());
    for (ia, ib) in &inp.pairs {
        let r = sess.check_equivalent(&ar_a, ids_a[*ia], &ar_b, ids_b[*ib]);
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
