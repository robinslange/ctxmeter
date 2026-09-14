mod probes;
mod transcript;

use clap::{Parser, Subcommand};
use probes::{billed_cost, default_policies, eligible, harvest, retention};
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
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".claude/projects")
}

fn pct(x: f64) -> String {
    format!("{:.2}%", x * 100.0)
}

fn cmd_summary(sessions: &[Session]) {
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
    println!("turns {turns}   sessions {n_sessions}");
    println!("\n{:<10}{:>18}{:>10}{:>18}", "", "tokens", "share", "billed equiv");
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
    models.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let list: Vec<String> = models.iter().take(4).map(|(m, n)| format!("{m} {n}")).collect();
    println!("\nmodels: {}", list.join(", "));
}

fn cmd_floor(sessions: &[Session]) {
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
    println!("first-turn billed prefix, n={}", all.len());
    for q in [25, 50, 75, 90] {
        println!("  p{q}: {:>10}", all[all.len() * q / 100]);
    }
    by.sort_by(|a, b| a.0.cmp(&b.0));
    println!("\n{:<10}{:>10}{:>12}{:>12}", "month", "sessions", "median", "p90");
    for (mo, mut v) in by {
        if v.len() < 15 {
            continue;
        }
        v.sort_unstable();
        println!(
            "{:<10}{:>10}{:>12}{:>12}",
            mo,
            v.len(),
            v[v.len() / 2],
            v[v.len() * 9 / 10]
        );
    }
    println!("\nthis is re-read every turn and no proxy can compress it.");
    println!("attribute it by changing config and re-running: tool definitions");
    println!("never appear in a transcript.");
}

fn cmd_probes(all: &[Session], max_df: usize, min_gap: usize) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, max_df, min_gap);
    let n: usize = probes.iter().map(|v| v.len()).sum();
    let yielding = probes.iter().filter(|v| !v.is_empty()).count();
    println!(
        "sessions scanned {}   yielding probes {}   probes {}",
        sessions.len(),
        yielding,
        n
    );
    if n == 0 {
        println!("\nNO PROBES HARVESTED. Loud failure by design: an empty denominator must");
        println!("never read as a pass. Relax --max-df or lower --min-gap.");
        return 1;
    }
    let mut gaps: Vec<usize> = probes
        .iter()
        .flatten()
        .map(|p| p.use_at - p.origin)
        .collect();
    gaps.sort_unstable();
    println!(
        "probes per scanned session {:.2}   (token in <= {} sessions)",
        n as f64 / sessions.len() as f64,
        max_df
    );
    println!(
        "gap from established to needed, in blocks: p50 {}  p90 {}  max {}\n",
        gaps[gaps.len() / 2],
        gaps[gaps.len() * 9 / 10],
        gaps[gaps.len() - 1]
    );
    println!("{:<20}{:>10}{:>12}{:>11}", "policy", "probes", "destroyed", "retained");
    for pol in default_policies() {
        let (kept, total) = retention(&sessions, &probes, pol);
        if total == 0 {
            continue;
        }
        println!(
            "{:<20}{:>10}{:>12}{:>11}",
            pol.label(),
            total,
            total - kept,
            pct(kept as f64 / total as f64)
        );
    }
    println!("\nretained = the fact was still literally present when the agent needed it.");
    println!("this measures information retention, not task success. losing a fact is not");
    println!("proof of failure: another valid route may exist.");
    0
}

fn cmd_sensitivity(all: &[Session]) -> i32 {
    let sessions = eligible(all);
    let pols = default_policies();
    print!("{:>7}{:>6}{:>10}", "rarity", "gap", "probes");
    for p in &pols {
        print!("{:>18}", p.label());
    }
    println!();
    let mut orders: Vec<Vec<String>> = Vec::new();
    for max_df in [1usize, 3, 10] {
        for min_gap in [5usize, 20] {
            let probes = harvest(&sessions, max_df, min_gap);
            let n: usize = probes.iter().map(|v| v.len()).sum();
            if n == 0 {
                continue;
            }
            print!("{max_df:>7}{min_gap:>6}{n:>10}");
            let mut scored: Vec<(String, f64)> = Vec::new();
            for pol in &pols {
                let (kept, total) = retention(&sessions, &probes, *pol);
                let r = if total == 0 { 0.0 } else { kept as f64 / total as f64 };
                print!("{:>17.1}%", r * 100.0);
                scored.push((pol.label(), r));
            }
            println!();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            orders.push(scored.into_iter().map(|(k, _)| k).collect());
        }
    }
    if orders.is_empty() {
        println!("\nno conditions produced probes.");
        return 1;
    }
    let base = &orders[0];
    let moved = orders.iter().filter(|o| *o != base).count();
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

/// A CLI piped into `head` or `less` must exit quietly, not panic on a closed pipe.
#[cfg(unix)]
fn allow_sigpipe() {
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
}
#[cfg(not(unix))]
fn allow_sigpipe() {}

fn cmd_tradeoff(all: &[Session], max_df: usize, min_gap: usize) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, max_df, min_gap);
    let n: usize = probes.iter().map(|v| v.len()).sum();
    if n == 0 {
        println!("NO PROBES HARVESTED; cannot report a tradeoff.");
        return 1;
    }
    let base = billed_cost(&sessions, None);
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
    for pol in default_policies() {
        let c = billed_cost(&sessions, Some(pol));
        let (kept, total) = retention(&sessions, &probes, pol);
        if total == 0 {
            continue;
        }
        let r = kept as f64 / total as f64;
        println!(
            "{:<20}{:>11.1}%{:>13.1}%{:>13.1}%",
            pol.label(),
            (1.0 - c / base) * 100.0,
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

fn main() {
    allow_sigpipe();
    let cli = Cli::parse();
    let root = cli.root.unwrap_or_else(default_root);
    let mut it = Interner::default();
    let sessions = transcript::load(&root, &mut it);
    if sessions.is_empty() {
        eprintln!("no transcripts found under {}", root.display());
        std::process::exit(2);
    }
    let code = match cli.cmd {
        Cmd::Summary => {
            cmd_summary(&sessions);
            0
        }
        Cmd::Floor => {
            cmd_floor(&sessions);
            0
        }
        Cmd::Probes { max_df, min_gap } => cmd_probes(&sessions, max_df, min_gap),
        Cmd::Sensitivity => cmd_sensitivity(&sessions),
        Cmd::Tradeoff { max_df, min_gap } => cmd_tradeoff(&sessions, max_df, min_gap),
    };
    std::process::exit(code);
}
