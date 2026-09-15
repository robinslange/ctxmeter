mod counterfactual;
mod probes;
mod transcript;

use clap::{Parser, Subcommand};
use probes::{
    billed_cost, billed_cost_with, bootstrap_ci, default_policies, eligible, harvest, retention,
    retention_by_session, CacheModel, CostOpts, Policy,
};
use std::path::PathBuf;
use transcript::{Interner, Session};

#[derive(Parser)]
#[command(
    name = "ctxmeter",
    about = "Measure what an LLM coding agent bills you, and what compaction destroys, from your own transcripts.",
    version
)]
struct Cli {
    /// Directory of .jsonl transcripts (default: ~/.claude/projects)
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// Print one JSON document instead of the report.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Cache hit rate, token classes, and billed equivalents.
    Summary,
    /// System prompt plus tool definitions, by month. Re-read every turn.
    Floor,
    /// How much later-needed information each retention policy destroys.
    Probes {
        /// A probe token may appear in at most this many sessions.
        #[arg(long, default_value_t = 3)]
        max_df: usize,
        /// Minimum blocks between a fact being established and reused.
        #[arg(long, default_value_t = 5)]
        min_gap: usize,
    },
    /// Widen the sample and check the policy ranking does not move.
    Sensitivity,
    /// Does the conclusion survive its own assumptions? Cache model, estimator
    /// scale, and uncertainty clustered by session.
    Robustness {
        #[arg(long, default_value_t = 3)]
        max_df: usize,
        #[arg(long, default_value_t = 5)]
        min_gap: usize,
    },
    /// Tier two. Replay the turn the agent actually took, with the context
    /// intact and with the policy applied, and see whether the fact comes back.
    /// Costs real money: dry-run first.
    Counterfactual {
        /// Tool results the policy keeps. 3 is a shipped default worth testing.
        #[arg(long, default_value_t = 3)]
        keep_last: usize,
        /// Only use traces from models matching this substring. One family per
        /// run: pooling two into one retention figure conflates them.
        #[arg(long, default_value = "sonnet")]
        model: String,
        /// Turns to replay. A turn is one observation; several facts destroyed in
        /// the same turn are scored against its one response, not re-requested.
        #[arg(long, default_value_t = 40)]
        sample: usize,
        /// Built turns to pass over before the sample starts. Selection is
        /// deterministic over a given corpus, so without it a smaller sample
        /// replays the first turns of a larger one. The order moves as transcripts
        /// are added, so a skip lines up with an earlier run only over the same
        /// corpus.
        #[arg(long, default_value_t = 0)]
        skip: usize,
        /// Build every request and price it, without calling the API.
        #[arg(long)]
        dry_run: bool,
        /// Required to spend money.
        #[arg(long)]
        yes: bool,
        /// Send every arm to the free token-counting endpoint and stop. Proves the API
        /// accepts what was rebuilt, and prices the run from the tokenizer, for nothing.
        #[arg(long)]
        preflight: bool,
        /// Print the fact under test and each raw response, so the verdict can be
        /// checked by eye instead of taken on trust. This prints probe text,
        /// which every other command deliberately does not.
        #[arg(long)]
        show_raw: bool,
        #[arg(long, default_value_t = 3)]
        max_df: usize,
        #[arg(long, default_value_t = 5)]
        min_gap: usize,
    },
    /// What each policy saves against what it destroys. The same knob does both.
    Tradeoff {
        #[arg(long, default_value_t = 3)]
        max_df: usize,
        #[arg(long, default_value_t = 5)]
        min_gap: usize,
    },
}

fn default_root() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".claude/projects")
}

fn pct(x: f64) -> String {
    format!("{:.2}%", x * 100.0)
}

fn emit(doc: serde_json::Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).expect("a Value always serializes")
    );
}

fn by_policy(rows: impl IntoIterator<Item = (Policy, serde_json::Value)>) -> serde_json::Value {
    rows.into_iter()
        .map(|(pol, fields)| {
            let mut doc = match pol {
                Policy::KeepLast(n) => serde_json::json!({ "kind": "keep_last", "keep_last": n }),
                Policy::TailBudget(b) => {
                    serde_json::json!({ "kind": "tail_budget", "budget_tokens": b })
                }
            };
            if let (Some(d), serde_json::Value::Object(f)) = (doc.as_object_mut(), fields) {
                d.extend(f);
            }
            (pol.label(), doc)
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn cmd_summary(sessions: &[Session], json: bool) {
    let (mut read, mut write, mut fresh, mut out, mut turns) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut models: Vec<(String, u64)> = Vec::new();
    let mut n_sessions = 0;
    for s in sessions {
        if s.usage.is_empty() {
            continue;
        }
        n_sessions += 1;
        for u in &s.usage {
            turns += 1;
            read += u.read;
            write += u.write;
            fresh += u.fresh;
            out += u.out;
            match models.iter_mut().find(|(m, _)| *m == u.model) {
                Some(e) => e.1 += 1,
                None => models.push((u.model.clone(), 1)),
            }
        }
    }
    let tot = (read + write + fresh) as f64;
    let cost = fresh as f64 + 1.25 * write as f64 + 0.10 * read as f64;
    models.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    if json {
        let share = |v: u64| v as f64 / tot * 100.0;
        emit(serde_json::json!({
            "turns": turns,
            "sessions": n_sessions,
            "tokens": { "fresh": fresh, "write": write, "read": read, "output": out },
            "share": { "fresh": share(fresh), "write": share(write), "read": share(read) },
            "billed": {
                "fresh": fresh as f64,
                "write": 1.25 * write as f64,
                "read": 0.10 * read as f64,
                "total": cost,
            },
            "no_cache_total": tot,
            "cache_hit_rate": share(read),
            "caching_saves": (1.0 - cost / tot) * 100.0,
            "models": models
                .iter()
                .map(|(m, n)| (m.clone(), serde_json::json!(n)))
                .collect::<serde_json::Map<_, _>>(),
        }));
        return;
    }
    println!("turns {turns}   sessions {n_sessions}");
    println!(
        "\n{:<10}{:>18}{:>10}{:>18}",
        "", "tokens", "share", "billed equiv"
    );
    for (name, v, mult) in [
        ("fresh", fresh, 1.0),
        ("write", write, 1.25),
        ("read", read, 0.10),
    ] {
        println!(
            "{:<10}{:>18}{:>10}{:>18.0}",
            name,
            v,
            pct(v as f64 / tot),
            mult * v as f64
        );
    }
    println!("{:<10}{:>18}", "output", out);
    println!("\ncache hit rate      {}", pct(read as f64 / tot));
    println!("billed equivalents  {cost:.0}  (no-cache counterfactual {tot:.0})");
    println!("caching already saves {}", pct(1.0 - cost / tot));
    let list: Vec<String> = models
        .iter()
        .take(4)
        .map(|(m, n)| format!("{m} {n}"))
        .collect();
    println!("\nmodels: {}", list.join(", "));
}

fn cmd_floor(sessions: &[Session], json: bool) {
    let mut all: Vec<u64> = Vec::new();
    let mut by: Vec<(String, Vec<u64>)> = Vec::new();
    for s in sessions {
        let Some(u) = s.usage.first() else { continue };
        let p = u.read + u.write + u.fresh;
        if p == 0 {
            continue;
        }
        all.push(p);
        if u.ts.len() >= 7 {
            let mo = u.ts[..7].to_string();
            match by.iter_mut().find(|(m, _)| *m == mo) {
                Some(e) => e.1.push(p),
                None => by.push((mo, vec![p])),
            }
        }
    }
    all.sort_unstable();
    by.sort_by(|a, b| a.0.cmp(&b.0));
    let months: Vec<(String, usize, u64, u64)> = by
        .into_iter()
        .filter(|(_, v)| v.len() >= 15)
        .map(|(mo, mut v)| {
            v.sort_unstable();
            (mo, v.len(), v[v.len() / 2], v[v.len() * 9 / 10])
        })
        .collect();
    let q = |p: usize| all[all.len() * p / 100];
    if json {
        emit(serde_json::json!({
            "n": all.len(),
            "p25": q(25),
            "p50": q(50),
            "p75": q(75),
            "p90": q(90),
            "months": months
                .iter()
                .map(|(mo, n, med, p90)| {
                    (mo.clone(), serde_json::json!({ "sessions": n, "median": med, "p90": p90 }))
                })
                .collect::<serde_json::Map<_, _>>(),
        }));
        return;
    }
    println!("first-turn billed prefix, n={}", all.len());
    for p in [25, 50, 75, 90] {
        println!("  p{p}: {:>10}", q(p));
    }
    println!(
        "\n{:<10}{:>10}{:>12}{:>12}",
        "month", "sessions", "median", "p90"
    );
    for (mo, n, med, p90) in &months {
        println!("{:<10}{:>10}{:>12}{:>12}", mo, n, med, p90);
    }
    println!("\nthis is re-read every turn and no proxy can compress it.");
    println!("attribute it by changing config and re-running: tool definitions");
    println!("never appear in a transcript.");
}

fn cmd_probes(all: &[Session], family: &[u32], max_df: usize, min_gap: usize, json: bool) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, family, max_df, min_gap);
    let n: usize = probes.iter().map(|v| v.len()).sum();
    let yielding = probes.iter().filter(|v| !v.is_empty()).count();
    let head = format!(
        "sessions scanned {}   yielding probes {}   probes {}",
        sessions.len(),
        yielding,
        n
    );
    if n == 0 {
        let msg = format!(
            "{head}\n\nNO PROBES HARVESTED. Loud failure by design: an empty denominator must\nnever read as a pass. Relax --max-df or lower --min-gap."
        );
        if json {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
        return 1;
    }
    let mut gaps: Vec<usize> = probes
        .iter()
        .flatten()
        .map(|p| p.use_at - p.origin)
        .collect();
    gaps.sort_unstable();

    // How deep into its session each reuse sits, counted in messages, which is where
    // tier two cuts. It is reported because the two halves are not interchangeable:
    // a replay of a shallow turn reproduces the trace far more often than a deep one,
    // so tier two can only speak for the shallow half, and a retention figure that
    // differs across the split is really two figures.
    let mut depths: Vec<u32> = Vec::new();
    for (s, ps) in sessions.iter().zip(&probes) {
        for p in ps {
            depths.push(s.blocks.get(p.use_at).map(|b| b.msg).unwrap_or(0));
        }
    }
    depths.sort_unstable();
    let median = depths[depths.len() / 2];

    let split = |deep: bool| -> Vec<Vec<probes::Probe>> {
        sessions
            .iter()
            .zip(&probes)
            .map(|(s, ps)| {
                ps.iter()
                    .filter(|p| {
                        let d = s.blocks.get(p.use_at).map(|b| b.msg).unwrap_or(0);
                        if deep {
                            d > median
                        } else {
                            d <= median
                        }
                    })
                    .copied()
                    .collect()
            })
            .collect()
    };
    let (shallow, deep) = (split(false), split(true));
    let share = |k: u64, t: u64| (t > 0).then(|| k as f64 / t as f64 * 100.0);
    let rows: Vec<_> = default_policies()
        .into_iter()
        .filter_map(|pol| {
            let (kept, total) = retention(&sessions, &probes, pol);
            if total == 0 {
                return None;
            }
            let (sk, st) = retention(&sessions, &shallow, pol);
            let (dk, dt) = retention(&sessions, &deep, pol);
            Some((pol, kept, total, share(sk, st), share(dk, dt)))
        })
        .collect();

    if json {
        emit(serde_json::json!({
            "sessions_scanned": sessions.len(),
            "sessions_yielding": yielding,
            "probes": n,
            "per_session": n as f64 / sessions.len() as f64,
            "max_df": max_df,
            "min_gap": min_gap,
            "gap": {
                "p50": gaps[gaps.len() / 2],
                "p90": gaps[gaps.len() * 9 / 10],
                "max": gaps[gaps.len() - 1],
            },
            "depth": {
                "p50": median,
                "p90": depths[depths.len() * 9 / 10],
                "max": depths[depths.len() - 1],
            },
            "policies": by_policy(rows.iter().map(|&(pol, kept, total, sh, dp)| {
                (pol, serde_json::json!({
                    "probes": total,
                    "destroyed": total - kept,
                    "retained": kept as f64 / total as f64 * 100.0,
                    "shallow": sh,
                    "deep": dp,
                }))
            })),
        }));
        return 0;
    }

    println!("{head}");
    println!(
        "probes per scanned session {:.2}   (token in <= {} sessions)",
        n as f64 / sessions.len() as f64,
        max_df
    );
    println!(
        "gap from established to needed, in blocks: p50 {}  p90 {}  max {}",
        gaps[gaps.len() / 2],
        gaps[gaps.len() * 9 / 10],
        gaps[gaps.len() - 1]
    );
    println!(
        "depth of the reuse, in messages: p50 {}  p90 {}  max {}\n",
        median,
        depths[depths.len() * 9 / 10],
        depths[depths.len() - 1]
    );
    println!(
        "{:<20}{:>10}{:>12}{:>11}{:>13}{:>10}",
        "policy", "probes", "destroyed", "retained", "shallow", "deep"
    );
    let cell = |x: Option<f64>| x.map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}%"));
    for &(pol, kept, total, sh, dp) in &rows {
        println!(
            "{:<20}{:>10}{:>12}{:>11}{:>13}{:>10}",
            pol.label(),
            total,
            total - kept,
            pct(kept as f64 / total as f64),
            cell(sh),
            cell(dp)
        );
    }
    println!("\nshallow = reuse at message {median} or earlier, deep = later. where the two");
    println!("columns disagree, one retained figure is averaging two populations. tier two");
    println!("replays a turn by cutting at its depth, and a replay agrees with the trace far");
    println!("more often when the prefix is short, so the probes it can speak for come from");
    println!("the shallow end of this distribution rather than from across it.");
    println!("\nretained = the fact was still literally present when the agent needed it.");
    println!("this measures information retention, not task success. losing a fact is not");
    println!("proof of failure: another valid route may exist.");
    0
}

fn cmd_sensitivity(all: &[Session], family: &[u32], json: bool) -> i32 {
    let sessions = eligible(all);
    let pols = default_policies();
    let mut conditions: Vec<(usize, usize, usize, Vec<f64>)> = Vec::new();
    for max_df in [1usize, 3, 10] {
        for min_gap in [5usize, 20] {
            let probes = harvest(&sessions, family, max_df, min_gap);
            let n: usize = probes.iter().map(|v| v.len()).sum();
            if n == 0 {
                continue;
            }
            let rs = pols
                .iter()
                .map(|&pol| {
                    let (kept, total) = retention(&sessions, &probes, pol);
                    if total == 0 {
                        0.0
                    } else {
                        kept as f64 / total as f64
                    }
                })
                .collect();
            conditions.push((max_df, min_gap, n, rs));
        }
    }
    let header = || {
        print!("{:>7}{:>6}{:>10}", "rarity", "gap", "probes");
        for p in &pols {
            print!("{:>18}", p.label());
        }
        println!();
    };
    if conditions.is_empty() {
        if json {
            eprintln!("no conditions produced probes.");
        } else {
            header();
            println!("\nno conditions produced probes.");
        }
        return 1;
    }
    let orders: Vec<Vec<String>> = conditions
        .iter()
        .map(|(_, _, _, rs)| {
            let mut scored: Vec<(String, f64)> = pols
                .iter()
                .map(|p| p.label())
                .zip(rs.iter().copied())
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            scored.into_iter().map(|(k, _)| k).collect()
        })
        .collect();
    let base = &orders[0];
    let moved = orders.iter().filter(|o| *o != base).count();

    if json {
        emit(serde_json::json!({
            "conditions": conditions
                .iter()
                .map(|(max_df, min_gap, n, rs)| {
                    serde_json::json!({
                        "max_df": max_df,
                        "min_gap": min_gap,
                        "probes": n,
                        "retained": pols
                            .iter()
                            .zip(rs)
                            .map(|(p, r)| (p.label(), serde_json::json!(r * 100.0)))
                            .collect::<serde_json::Map<_, _>>(),
                    })
                })
                .collect::<Vec<_>>(),
            "ranking": base,
            "moved": moved,
            "stable": moved == 0,
        }));
        return i32::from(moved > 0);
    }

    header();
    for (max_df, min_gap, n, rs) in &conditions {
        print!("{max_df:>7}{min_gap:>6}{n:>10}");
        for r in rs {
            print!("{:>17.1}%", r * 100.0);
        }
        println!();
    }
    println!("\npolicy ranking: {}", base.join(" > "));
    if moved > 0 {
        println!(
            "RANKING MOVED in {moved} of {} conditions. The result is scope-dependent.",
            orders.len()
        );
        return 1;
    }
    println!(
        "stable across all {} conditions. widening the sample does not move the answer.",
        orders.len()
    );
    0
}

struct CfArgs {
    keep_last: usize,
    model: String,
    sample: usize,
    skip: usize,
    dry_run: bool,
    yes: bool,
    preflight: bool,
    show_raw: bool,
    max_df: usize,
    min_gap: usize,
}

/// What the selection lost, and where each denominator comes from. `other model`
/// is counted over the whole corpus before any turn is visited, so it is not a
/// component of `facts considered` and is not printed as one.
fn print_drops(d: &counterfactual::Dropped, skip: usize, built: usize) {
    println!(
        "facts on another model, corpus-wide: {} (excluded before selection)",
        d.other_model
    );
    println!("turns skipped     : {skip} (built, then passed over; not counted below)");
    println!("turns considered  : {}", d.turns_considered);
    println!("  unrebuildable   : {}", d.unrebuildable);
    println!("facts considered  : {}", d.considered);
    println!("  policy kept it  : {}", d.policy_kept_the_fact);
    println!("  no re-fetch tgt : {}", d.no_refetch_target);
    println!("  fact is origin  : {}", d.fact_is_its_origin);
    println!("turns built       : {built}");
}

fn cmd_counterfactual(all: &[Session], names: &[String], family: &[u32], a: CfArgs) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, family, a.max_df, a.min_gap);
    let paths: Vec<String> = sessions.iter().map(|s| s.path.clone()).collect();
    let pol = Policy::KeepLast(a.keep_last);

    let mut dropped = counterfactual::Dropped::default();
    let corpus = counterfactual::Corpus {
        sessions: &sessions,
        probes: &probes,
        paths: &paths,
        interned: names,
    };
    let cases = counterfactual::build_cases(&corpus, pol, a.sample, a.skip, &a.model, &mut dropped);
    if cases.is_empty() {
        println!("No usable turns.");
        print_drops(&dropped, a.skip, cases.len());
        return 1;
    }

    let mut models: Vec<&str> = cases.iter().map(|c| c.model.as_str()).collect();
    models.sort_unstable();
    models.dedup();
    if models.len() > 1 {
        eprintln!("--model {:?} matched more than one model:", a.model);
        for m in &models {
            eprintln!("  {m}");
        }
        eprintln!("one retention figure over two of them conflates them, and they are");
        eprintln!("priced differently. narrow the filter to one.");
        return 2;
    }

    let cost = counterfactual::estimate_cost(&cases);
    let toks: u64 = cases
        .iter()
        .map(|c| {
            let (a, b) = counterfactual::estimate_tokens(c);
            a + b
        })
        .sum();
    println!("policy under test: {}", pol.label());
    println!("model family    : {}", a.model);
    print_drops(&dropped, a.skip, cases.len());
    println!("input tokens     : {toks} across both arms");
    println!("spend ceiling    : ${cost:.2}  (published prices, checked 2026-09-14)");
    println!("\na ceiling, not an estimate, and it reads high: it bounds output at the");
    println!("ceiling for two arms per turn where a real turn produces a fraction of it,");
    println!("and it buys both arms of every turn where a turn whose control arm fails");
    println!("never buys its second. Against that, it takes input from serialized length");
    println!("over 4, which undercounts. The run prints the measured count beside it.");
    println!("\neach turn is at most two calls: the intact context as control, then the");
    println!("policy applied. a fact only counts if the control arm reproduces it.");

    if a.dry_run {
        let mut models: Vec<(String, usize)> = Vec::new();
        for c in &cases {
            match models.iter_mut().find(|(m, _)| *m == c.model) {
                Some(e) => e.1 += 1,
                None => models.push((c.model.clone(), 1)),
            }
        }
        println!("\nmodels to be called (the one that produced each trace):");
        for (m, n) in models {
            println!("  {m}  {n}");
        }
        let mut bad = 0;
        for (i, c) in cases.iter().enumerate() {
            for (arm, msgs) in [
                ("intact", &c.messages_intact),
                ("masked", &c.messages_masked),
            ] {
                if let Some(why) = counterfactual::invalid(msgs) {
                    if bad < 5 {
                        println!("\ncase {i} {arm} is not a sendable request: {why}");
                    }
                    bad += 1;
                }
            }
        }
        if bad > 0 {
            println!("\n{bad} arm(s) would be rejected by the API. Counting them all rather");
            println!("than stopping at the first: one is a bug, and the rate is the news.");
        } else {
            println!(
                "\nall {} cases satisfy the request invariants.",
                cases.len()
            );
        }
        let mut sess: Vec<usize> = cases.iter().map(|c| c.session).collect();
        sess.sort_unstable();
        sess.dedup();
        println!("{} turns across {} sessions.", cases.len(), sess.len());
        let (a, b) = counterfactual::estimate_tokens(&cases[0]);
        let facts: usize = cases.iter().map(|c| c.facts.len()).sum();
        println!(
            "\n{facts} facts over {} turns, {:.2} per turn",
            cases.len(),
            facts as f64 / cases.len() as f64
        );
        println!(
            "first turn: {} messages intact ({a} tok), masked ({b} tok), {} fact(s)",
            cases[0].messages_intact.len(),
            cases[0].facts.len()
        );
        println!("\ndry run: nothing was sent.");
        return i32::from(bad > 0);
    }

    let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("\nANTHROPIC_API_KEY is not set.");
        eprintln!("This needs a real API key. A Claude subscription credential will not do:");
        eprintln!("Anthropic's terms do not permit using Free, Pro or Max OAuth tokens in");
        eprintln!("another tool, so ctxmeter will not read them.");
        return 2;
    }
    if a.preflight {
        let (sizes, errs) = counterfactual::preflight(&cases, &key);
        let measured: u64 = sizes.iter().map(|(x, y)| x + y).sum();
        println!(
            "\nmeasured {measured} input tokens across both arms of {} turns",
            cases.len()
        );
        if errs.is_empty() {
            println!("every arm was accepted. nothing was spent.");
            return 0;
        }
        eprintln!(
            "\n{} of {} arms would be rejected:",
            errs.len(),
            cases.len() * 2
        );
        for e in &errs {
            eprintln!("  {e}");
        }
        eprintln!("\nnothing was spent. these are bugs in the rebuild, not results.");
        return 3;
    }
    if !a.yes {
        eprintln!("\nRefusing to spend ${cost:.2} without --yes.");
        return 2;
    }

    let v = counterfactual::run(&cases, &key, a.show_raw);
    let attempted = v.turns_attempted;
    if !v.errors.is_empty() {
        eprintln!("\nThe run stopped: the API did not accept a request.");
        for e in &v.errors {
            eprintln!("  {e}");
        }
        eprintln!("\nThis is a bug in the rebuild, not a result about the policy.");
        if attempted == 0 {
            return 3;
        }
        eprintln!("{attempted} turn(s) completed before it, reported below.");
    }
    println!("\n{:<30}{:>8}", "turns replayed", v.turns_attempted);
    println!("{:<30}{:>8}", "turns yielding a fact", v.turns_informative);
    println!("{:<30}{:>8}", "sessions", v.sessions);
    println!("{:<30}{:>8}", "facts discarded (control)", v.discarded);
    println!(
        "{:<30}{:>8}",
        "facts unusable, control arm", v.unusable_control
    );
    println!(
        "{:<30}{:>8}",
        "facts unusable, treatment arm", v.unusable_treatment
    );
    println!("{:<30}{:>8}", "facts informative", v.informative);
    if v.measured_tokens > 0 {
        println!(
            "\ninput tokens: {} measured by count_tokens, {toks} estimated",
            v.measured_tokens
        );
    }
    if v.informative == 0 {
        println!("\nNo informative cases. Nothing can be concluded.");
        return 1;
    }
    if !v.errors.is_empty() {
        println!("\nThe sample below is what completed, not what was asked for.");
    }
    let p = |n: usize| format!("{:.1}%", n as f64 / v.informative as f64 * 100.0);
    println!("\nof the informative cases, with the fact removed the model:");
    println!(
        "  {:<24}{:>8}{:>9}",
        "reproduced it anyway",
        v.reproduced,
        p(v.reproduced)
    );
    println!(
        "  {:<24}{:>8}{:>9}",
        "rewrote it into prose",
        v.regenerated,
        p(v.regenerated)
    );
    println!(
        "  {:<24}{:>8}{:>9}",
        "went to fetch it",
        v.sought,
        p(v.sought)
    );
    println!("  {:<24}{:>8}{:>9}", "did neither", v.silent, p(v.silent));
    println!("\n'did neither' is the irreversible share: the fact was gone and the model");
    println!("did not ask for it back. 'went to fetch it' is the healthy failure.");
    println!(
        "\n{} facts over {} turns in {} sessions. facts sharing a turn share one",
        v.informative, v.turns_informative, v.sessions
    );
    println!("response, and facts sharing an origin share their 'went to fetch it'");
    println!("verdict, so an interval bootstraps over sessions and not over facts.");
    // A run that stopped early still reports what it bought, but it did not succeed.
    if v.errors.is_empty() {
        0
    } else {
        3
    }
}

/// A CLI piped into `head` or `less` must exit quietly, not panic on a closed pipe.
#[cfg(unix)]
fn allow_sigpipe() {
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
}
#[cfg(not(unix))]
fn allow_sigpipe() {}

fn cmd_tradeoff(all: &[Session], family: &[u32], max_df: usize, min_gap: usize, json: bool) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, family, max_df, min_gap);
    let n: usize = probes.iter().map(|v| v.len()).sum();
    if n == 0 {
        if json {
            eprintln!("NO PROBES HARVESTED; cannot report a tradeoff.");
        } else {
            println!("NO PROBES HARVESTED; cannot report a tradeoff.");
        }
        return 1;
    }
    let base = billed_cost(&sessions, None);
    let rows: Vec<(Policy, f64, f64)> = default_policies()
        .into_iter()
        .filter_map(|pol| {
            let c = billed_cost(&sessions, Some(pol));
            let (kept, total) = retention(&sessions, &probes, pol);
            (total > 0).then(|| (pol, (1.0 - c / base) * 100.0, kept as f64 / total as f64))
        })
        .collect();
    if json {
        emit(serde_json::json!({
            "sessions": sessions.len(),
            "probes": n,
            "baseline_billed": base,
            "policies": by_policy(rows.iter().map(|&(pol, saved, r)| {
                (pol, serde_json::json!({
                    "cost_saved": saved,
                    "retained": r * 100.0,
                    "lost": (1.0 - r) * 100.0,
                }))
            })),
        }));
        return 0;
    }
    println!(
        "{} sessions, {} probes, baseline billed cost {:.0} equivalents\n",
        sessions.len(),
        n,
        base
    );
    println!(
        "{:<20}{:>12}{:>14}{:>14}",
        "policy", "cost saved", "info retained", "info lost"
    );
    for &(pol, saved, r) in &rows {
        println!(
            "{:<20}{:>11.1}%{:>13.1}%{:>13.1}%",
            pol.label(),
            saved,
            r * 100.0,
            (1.0 - r) * 100.0
        );
    }
    println!("\nboth columns come from the same policy applied the same way.");
    println!("cost is grounded in real prefix sizes, so the system prompt and tool");
    println!("definitions are carried unchanged: no context policy can touch them.");
    println!("retention measures information survival, not task success.");
    0
}

fn cmd_robustness(
    all: &[Session],
    family: &[u32],
    max_df: usize,
    min_gap: usize,
    json: bool,
) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, family, max_df, min_gap);
    if probes.iter().all(|v| v.is_empty()) {
        if json {
            eprintln!("NO PROBES HARVESTED.");
        } else {
            println!("NO PROBES HARVESTED.");
        }
        return 1;
    }
    let pols = default_policies();
    let saved = |pol: Policy, model: CacheModel, scale: f64| {
        let o = CostOpts { model, scale };
        let base = billed_cost_with(&sessions, None, &o);
        let c = billed_cost_with(&sessions, Some(pol), &o);
        (1.0 - c / base) * 100.0
    };
    let cost_model: Vec<(Policy, f64, f64)> = pols
        .iter()
        .map(|&p| {
            (
                p,
                saved(p, CacheModel::LongestPrefix, 1.0),
                saved(p, CacheModel::AllOrNothing, 1.0),
            )
        })
        .collect();
    let estimator: Vec<(Policy, [f64; 3])> = pols
        .iter()
        .map(|&p| {
            (
                p,
                [1.0, 2.0, 3.0].map(|sc| saved(p, CacheModel::LongestPrefix, sc)),
            )
        })
        .collect();
    let uncertainty: Vec<(Policy, f64, f64, f64, usize)> = pols
        .iter()
        .filter_map(|&p| {
            let per = retention_by_session(&sessions, &probes, p);
            let (kept, total) = retention(&sessions, &probes, p);
            if total == 0 {
                return None;
            }
            let (lo, hi) = bootstrap_ci(&per, 2000);
            Some((
                p,
                kept as f64 / total as f64 * 100.0,
                lo * 100.0,
                hi * 100.0,
                per.len(),
            ))
        })
        .collect();

    if json {
        emit(serde_json::json!({
            "cost_model": by_policy(cost_model.iter().map(|&(p, lp, aon)| {
                (p, serde_json::json!({ "longest_prefix": lp, "all_or_nothing": aon }))
            })),
            "estimator": by_policy(estimator.iter().map(|&(p, [x1, x2, x3])| {
                (p, serde_json::json!({ "x1": x1, "x2": x2, "x3": x3 }))
            })),
            "uncertainty": by_policy(uncertainty.iter().map(|&(p, r, lo, hi, s)| {
                (p, serde_json::json!({ "retained": r, "ci_lo": lo, "ci_hi": hi, "sessions": s }))
            })),
        }));
        return 0;
    }

    println!("1. COST MODEL. The cost column is a simulation. Does its sign survive");
    println!("   replacing generous prefix matching with the all-or-nothing behaviour");
    println!("   measured on real traces?\n");
    println!(
        "{:<20}{:>16}{:>16}",
        "policy", "longest-prefix", "all-or-nothing"
    );
    for &(p, lp, aon) in &cost_model {
        println!("{:<20}{:>15.1}%{:>15.1}%", p.label(), lp, aon);
    }

    println!("\n2. ESTIMATOR. Per-block sizes are character estimates. Does the sign");
    println!("   survive scaling every visible block by 2x and 3x?\n");
    println!("{:<20}{:>10}{:>10}{:>10}", "policy", "1x", "2x", "3x");
    for &(p, [x1, x2, x3]) in &estimator {
        println!("{:<20}{:>9.1}%{:>9.1}%{:>9.1}%", p.label(), x1, x2, x3);
    }

    println!("\n3. UNCERTAINTY. Probes inside one session share a trajectory, so they");
    println!("   are not independent. 95% interval from a cluster bootstrap over");
    println!("   sessions, 2000 resamples.\n");
    println!(
        "{:<20}{:>10}{:>22}{:>10}",
        "policy", "retained", "95% CI (clustered)", "sessions"
    );
    for &(p, r, lo, hi, s) in &uncertainty {
        println!(
            "{:<20}{:>9.1}%{:>14.1}% - {:>4.1}%{:>10}",
            p.label(),
            r,
            lo,
            hi,
            s
        );
    }
    println!("\nnot tested here: whether a lost fact changes the outcome, and whether");
    println!("an implementation that keeps originals retrievable recovers it.");
    println!("both of those make this an upper bound on harm, not a measurement of it.");
    0
}

fn main() {
    allow_sigpipe();
    let cli = Cli::parse();
    let json = cli.json;
    let root = cli.root.unwrap_or_else(default_root);
    let mut it = Interner::default();
    let sessions = transcript::load(&root, &mut it);
    if sessions.is_empty() {
        eprintln!("no transcripts found under {}", root.display());
        std::process::exit(2);
    }
    let code = match cli.cmd {
        Cmd::Summary => {
            cmd_summary(&sessions, json);
            0
        }
        Cmd::Floor => {
            cmd_floor(&sessions, json);
            0
        }
        Cmd::Probes { max_df, min_gap } => cmd_probes(&sessions, &it.family, max_df, min_gap, json),
        Cmd::Sensitivity => cmd_sensitivity(&sessions, &it.family, json),
        Cmd::Tradeoff { max_df, min_gap } => {
            cmd_tradeoff(&sessions, &it.family, max_df, min_gap, json)
        }
        Cmd::Robustness { max_df, min_gap } => {
            cmd_robustness(&sessions, &it.family, max_df, min_gap, json)
        }
        Cmd::Counterfactual {
            keep_last,
            model,
            sample,
            skip,
            dry_run,
            yes,
            preflight,
            show_raw,
            max_df,
            min_gap,
        } => cmd_counterfactual(
            &sessions,
            &it.names,
            &it.family,
            CfArgs {
                keep_last,
                model,
                sample,
                skip,
                dry_run,
                yes,
                preflight,
                show_raw,
                max_df,
                min_gap,
            },
        ),
    };
    std::process::exit(code);
}
