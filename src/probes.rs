use crate::transcript::{Block, Kind, Role, Session};
use std::collections::HashMap;

#[derive(Clone, Copy)]
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

    /// Same decision, but against sizes supplied directly rather than looked up
    /// in a session. Tier two rebuilds requests from raw JSON and has no blocks.
    pub fn survives_from_indexed(
        &self,
        live: &[usize],
        sizes: &std::collections::HashMap<usize, u32>,
    ) -> usize {
        match *self {
            Policy::KeepLast(n) => live.len().saturating_sub(n),
            Policy::TailBudget(budget) => {
                let mut run: u64 = 0;
                for (p, i) in live.iter().enumerate().rev() {
                    run += *sizes.get(i).unwrap_or(&0) as u64;
                    if run > budget as u64 {
                        return p + 1;
                    }
                }
                0
            }
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

/// Rarity is counted per family, not per literal: tokens that differ only in their
/// digits are one family, and a numbered family is guessable from any one member
/// even when each member sits in a single session.
pub fn harvest(
    sessions: &[&Session],
    family: &[u32],
    max_df: usize,
    min_gap: usize,
) -> Vec<Vec<Probe>> {
    let mut df: HashMap<u32, u32> = HashMap::new();
    for s in sessions {
        let mut here: Vec<u32> = Vec::new();
        for b in &s.blocks {
            if b.kind == Kind::ToolResult {
                here.extend_from_slice(&b.toks);
            }
        }
        let mut here: Vec<u32> = here.into_iter().map(|t| family[t as usize]).collect();
        here.sort_unstable();
        here.dedup();
        for f in here {
            *df.entry(f).or_insert(0) += 1;
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
                            if df.get(&family[t as usize]).copied().unwrap_or(0) as usize <= max_df
                            {
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
#[derive(Clone, Copy, PartialEq)]
pub enum CacheModel {
    /// Credit every unchanged leading block as a cache read. Generous to masking.
    LongestPrefix,
    /// Matching is anchored at breakpoints with a 20-block lookback, and measured
    /// invalidation on real traces is all-or-nothing. So any change in the message
    /// region forfeits the whole region; only system and tools stay cached.
    AllOrNothing,
}

/// Multiplies the per-block token estimate, to test how far the conclusion depends
/// on the estimator rather than on the mechanism.
pub struct CostOpts {
    pub model: CacheModel,
    pub scale: f64,
}

pub fn billed_cost(sessions: &[&Session], pol: Option<Policy>) -> f64 {
    billed_cost_with(
        sessions,
        pol,
        &CostOpts {
            model: CacheModel::LongestPrefix,
            scale: 1.0,
        },
    )
}

pub fn billed_cost_with(sessions: &[&Session], pol: Option<Policy>, o: &CostOpts) -> f64 {
    let mut cost = 0.0;
    for s in sessions {
        let tr: Vec<usize> = s
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.kind == Kind::ToolResult)
            .map(|(i, _)| i)
            .collect();
        let mut prev: Vec<(usize, u64)> = Vec::new();
        for &(cut, real) in &s.turns {
            let cut = cut.min(s.blocks.len());
            let sz = |b: &Block| (b.tokens as f64 * o.scale) as u64;
            let raw: u64 = s.blocks[..cut].iter().map(sz).sum();
            let invisible = real.saturating_sub(raw);

            let live_end = tr.partition_point(|&i| i < cut);
            let live = &tr[..live_end];
            let from = pol.map(|p| p.survives_from(live, &s.blocks)).unwrap_or(0);
            let masked: std::collections::HashSet<usize> =
                live[..from.min(live.len())].iter().copied().collect();

            let cur: Vec<(usize, u64)> = s.blocks[..cut]
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let full = sz(b);
                    let t = if masked.contains(&i) && full > PLACEHOLDER as u64 {
                        PLACEHOLDER as u64
                    } else {
                        full
                    };
                    (i, t)
                })
                .collect();

            let mut lcp: u64 = 0;
            let mut identical = cur.len() == prev.len();
            for (a, b) in cur.iter().zip(prev.iter()) {
                if a == b {
                    lcp += a.1;
                } else {
                    identical = false;
                    break;
                }
            }
            let mut shared = match o.model {
                CacheModel::LongestPrefix => lcp,
                // the appended tail is new either way; what differs is whether an
                // edit inside the region forfeits the blocks before it
                CacheModel::AllOrNothing => {
                    let unchanged_prefix =
                        identical || lcp >= prev.iter().map(|x| x.1).sum::<u64>();
                    if unchanged_prefix {
                        lcp
                    } else {
                        0
                    }
                }
            };
            if !prev.is_empty() {
                shared += invisible;
            }
            let total: u64 = cur.iter().map(|x| x.1).sum::<u64>() + invisible;
            cost += R * shared as f64 + W * total.saturating_sub(shared) as f64;
            prev = cur;
        }
    }
    cost
}

/// Retention counted per session, so uncertainty can be clustered correctly:
/// probes inside one session share a trajectory and a mask boundary, so they
/// are not independent observations.
pub fn retention_by_session(
    sessions: &[&Session],
    probes: &[Vec<Probe>],
    pol: Policy,
) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
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
        let mut kept = 0u64;
        let mut total = 0u64;
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
        if total > 0 {
            out.push((kept, total));
        }
    }
    out
}

/// Cluster bootstrap: resample whole sessions, not individual probes.
pub fn bootstrap_ci(per_session: &[(u64, u64)], iters: usize) -> (f64, f64) {
    if per_session.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = per_session.len();
    let mut rs: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut k = 0u64;
        let mut t = 0u64;
        for _ in 0..n {
            let (a, b) = per_session[(next() as usize) % n];
            k += a;
            t += b;
        }
        if t > 0 {
            rs.push(k as f64 / t as f64);
        }
    }
    rs.sort_by(f64::total_cmp);
    if rs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    (rs[rs.len() * 25 / 1000], rs[rs.len() * 975 / 1000])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{Interner, Kind, Role};

    fn session(blocks: Vec<Block>) -> Session {
        Session {
            path: "s.jsonl".into(),
            msgs: blocks.len() + 1,
            blocks,
            turns: vec![],
            usage: vec![],
        }
    }

    fn block(i: u32, role: Role, kind: Kind, toks: Vec<u32>) -> Block {
        Block {
            msg: i,
            kind,
            role,
            tokens: 0,
            toks,
        }
    }

    /// A tool result establishes the token, the assistant names it six blocks on.
    fn establish_then_reuse(tok: u32) -> Session {
        let mut blocks = vec![block(0, Role::Assistant, Kind::ToolResult, vec![tok])];
        for i in 1..6 {
            blocks.push(block(i, Role::Assistant, Kind::Text, vec![]));
        }
        blocks.push(block(6, Role::Assistant, Kind::ToolUse, vec![tok]));
        session(blocks)
    }

    /// `/tmp/t3.txt` passes the length, digit and letter floors, and each literal
    /// sits in one session, but the digit is a task counter: a model that has seen
    /// `/tmp/t2.txt` can produce `/tmp/t3.txt` without having read it. Rarity is
    /// counted over the pattern the digits fill, so the family is common even when
    /// each member is rare. A hash has no siblings and stays a probe.
    #[test]
    fn a_numbered_family_is_not_rare_even_when_each_member_is() {
        let mut it = Interner::default();
        let counters: Vec<u32> = (1..=4)
            .map(|n| it.intern(&format!("/tmp/t{n}.txt")))
            .collect();
        let hash = it.intern("a7f3c9e21b84");
        let mut all: Vec<Session> = counters.iter().map(|&t| establish_then_reuse(t)).collect();
        all.push(establish_then_reuse(hash));
        let sessions: Vec<&Session> = all.iter().collect();

        let probes = harvest(&sessions, &it.family, 3, 5);

        let from_counters: usize = probes[..4].iter().map(Vec::len).sum();
        assert_eq!(from_counters, 0, "a numbered family counted as rare");
        assert_eq!(probes[4].len(), 1, "a token with no siblings was dropped");
    }
}
