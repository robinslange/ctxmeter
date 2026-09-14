use crate::transcript::{Block, Kind, Role, Session};
use std::collections::HashMap;

pub struct Probe {
    pub tok: u32,
    pub origin: usize,
    pub use_at: usize,
}

#[derive(Clone, Copy)]
pub enum Policy {
    /// Keep the N most recent tool results, mask the rest.
    KeepLast(usize),
    /// Keep as many recent tool results as fit in a token budget.
    TailBudget(u32),
}

impl Policy {
    pub fn label(&self) -> String {
        match self {
            Policy::KeepLast(n) => format!("keep_last_{n}"),
            Policy::TailBudget(b) => format!("tail_budget_{}k", b / 1000),
        }
    }

    /// Lowest position in `live` that survives this policy.
    fn survives_from(&self, live: &[usize], blocks: &[Block]) -> usize {
        match *self {
            Policy::KeepLast(n) => live.len().saturating_sub(n),
            Policy::TailBudget(budget) => {
                let mut run: u64 = 0;
                for (p, &i) in live.iter().enumerate().rev() {
                    run += blocks[i].tokens as u64;
                    if run > budget as u64 {
                        return p + 1;
                    }
                }
                0
            }
        }
    }
}

pub fn default_policies() -> Vec<Policy> {
    vec![
        Policy::KeepLast(1),
        Policy::KeepLast(3),
        Policy::KeepLast(5),
        Policy::KeepLast(10),
        Policy::KeepLast(25),
        Policy::KeepLast(50),
        Policy::TailBudget(10_000),
        Policy::TailBudget(40_000),
        Policy::TailBudget(100_000),
        Policy::TailBudget(200_000),
    ]
}

fn has(b: &Block, tok: u32) -> bool {
    b.toks.binary_search(&tok).is_ok()
}

/// Probes derived from traces, never authored: a fact a tool result established,
/// that the agent demonstrably reused far later.
/// Sessions long enough to plausibly establish a fact and reuse it later.
pub const MIN_MSGS: usize = 6;

pub fn eligible(sessions: &[Session]) -> Vec<&Session> {
    sessions.iter().filter(|s| s.msgs >= MIN_MSGS).collect()
}

pub fn harvest(sessions: &[&Session], max_df: usize, min_gap: usize) -> Vec<Vec<Probe>> {
    let mut df: HashMap<u32, u32> = HashMap::new();
    for s in sessions {
        let mut here: Vec<u32> = Vec::new();
        for b in &s.blocks {
            if b.kind == Kind::ToolResult {
                here.extend_from_slice(&b.toks);
            }
        }
        here.sort_unstable();
        here.dedup();
        for t in here {
            *df.entry(t).or_insert(0) += 1;
        }
    }

    sessions
        .iter()
        .map(|s| {
            let mut origin: HashMap<u32, usize> = HashMap::new();
            let mut excluded: HashMap<u32, ()> = HashMap::new();

            for (i, b) in s.blocks.iter().enumerate() {
                match (b.role, b.kind) {
                    (_, Kind::ToolResult) => {
                        for &t in &b.toks {
                            if df.get(&t).copied().unwrap_or(0) as usize <= max_df {
                                origin.entry(t).or_insert(i);
                            }
                        }
                    }
                    (Role::User, Kind::Text) => {
                        for &t in &b.toks {
                            excluded.insert(t, ());
                        }
                    }
                    _ => {}
                }
            }

            let mut out = Vec::new();
            for (&tok, &o) in &origin {
                if excluded.contains_key(&tok) {
                    continue;
                }
                // first reuse at least min_gap blocks after the fact was established
                let mut first = usize::MAX;
                for (i, b) in s.blocks.iter().enumerate().skip(o + min_gap) {
                    if b.role == Role::Assistant
                        && matches!(b.kind, Kind::Text | Kind::ToolUse)
                        && has(b, tok)
                    {
                        first = i;
                        break;
                    }
                }
                if first != usize::MAX {
                    out.push(Probe {
                        tok,
                        origin: o,
                        use_at: first,
                    });
                }
            }
            out
        })
        .collect()
}

/// (facts still present when needed, facts tested) under one policy.
pub fn retention(sessions: &[&Session], probes: &[Vec<Probe>], pol: Policy) -> (u64, u64) {
    let mut kept = 0u64;
    let mut total = 0u64;
    for (s, ps) in sessions.iter().zip(probes) {
        if ps.is_empty() {
            continue;
        }
        let tr: Vec<usize> = s
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.kind == Kind::ToolResult)
            .map(|(i, _)| i)
            .collect();
        for p in ps {
            let cut = tr.partition_point(|&i| i < p.use_at);
            let live = &tr[..cut];
            let holders: Vec<usize> = live
                .iter()
                .enumerate()
                .filter(|(_, &i)| has(&s.blocks[i], p.tok))
                .map(|(pos, _)| pos)
                .collect();
            if holders.is_empty() {
                continue;
            }
            total += 1;
            let from = pol.survives_from(live, &s.blocks);
            if holders.iter().any(|&pos| pos >= from) {
                kept += 1;
            }
        }
    }
    (kept, total)
}

/// A masked tool result still costs its call header and a placeholder.
const PLACEHOLDER: u32 = 25;
const W: f64 = 1.25;
const R: f64 = 0.10;

/// Billed input cost under a policy, in base-input-token equivalents, grounded in
/// the real prefix sizes from the usage records. The invisible remainder of each
/// prefix (system prompt and tool definitions) is carried unchanged, because no
/// context policy can touch it.
pub fn billed_cost(sessions: &[&Session], pol: Option<Policy>) -> f64 {
    let mut cost = 0.0;
    for s in sessions {
        let tr: Vec<usize> = s
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.kind == Kind::ToolResult)
            .map(|(i, _)| i)
            .collect();
        let mut prev: Vec<(usize, u32)> = Vec::new();
        for &(cut, real) in &s.turns {
            let cut = cut.min(s.blocks.len());
            let raw: u64 = s.blocks[..cut].iter().map(|b| b.tokens as u64).sum();
            let invisible = real.saturating_sub(raw);

            let live_end = tr.partition_point(|&i| i < cut);
            let live = &tr[..live_end];
            let from = pol.map(|p| p.survives_from(live, &s.blocks)).unwrap_or(0);
            let masked: std::collections::HashSet<usize> =
                live[..from.min(live.len())].iter().copied().collect();

            let cur: Vec<(usize, u32)> = s.blocks[..cut]
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let t = if masked.contains(&i) && b.tokens > PLACEHOLDER {
                        PLACEHOLDER
                    } else {
                        b.tokens
                    };
                    (i, t)
                })
                .collect();

            let mut shared: u64 = 0;
            for (a, b) in cur.iter().zip(prev.iter()) {
                if a == b {
                    shared += a.1 as u64;
                } else {
                    break;
                }
            }
            if !prev.is_empty() {
                shared += invisible;
            }
            let total: u64 = cur.iter().map(|x| x.1 as u64).sum::<u64>() + invisible;
            cost += R * shared as f64 + W * total.saturating_sub(shared) as f64;
            prev = cur;
        }
    }
    cost
}
