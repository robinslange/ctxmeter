mod counterfactual;
mod probes;
mod transcript;

use clap::{Parser, Subcommand};
use probes::{
    billed_cost, billed_cost_with, bootstrap_ci, default_policies, eligible, harvest, retention,
    retention_by_session, CacheModel, CostOpts,
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
        #[arg(long, default_value_t = 40)]
        sample: usize,
        /// Build every request and price it, without calling the API.
        #[arg(long)]
        dry_run: bool,
        /// Required to spend money.
        #[arg(long)]
        yes: bool,
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
    models.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let list: Vec<String> = models
        .iter()
        .take(4)
        .map(|(m, n)| format!("{m} {n}"))
        .collect();
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
    println!(
        "\n{:<10}{:>10}{:>12}{:>12}",
        "month", "sessions", "median", "p90"
    );
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
    println!(
        "{:<20}{:>10}{:>12}{:>11}",
        "policy", "probes", "destroyed", "retained"
    );
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
                let r = if total == 0 {
                    0.0
                } else {
                    kept as f64 / total as f64
                };
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

struct CfArgs {
    keep_last: usize,
    model: String,
    sample: usize,
    dry_run: bool,
    yes: bool,
    show_raw: bool,
    max_df: usize,
    min_gap: usize,
}

fn cmd_counterfactual(all: &[Session], names: &[String], a: CfArgs) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, a.max_df, a.min_gap);
    let paths: Vec<String> = sessions.iter().map(|s| s.path.clone()).collect();
    let pol = probes::Policy::KeepLast(a.keep_last);

    let mut dropped = counterfactual::Dropped::default();
    let corpus = counterfactual::Corpus {
        sessions: &sessions,
        probes: &probes,
        paths: &paths,
        interned: names,
    };
    let cases = counterfactual::build_cases(&corpus, pol, a.sample, &a.model, &mut dropped);
    if cases.is_empty() {
        println!(
            "No usable cases, out of {} probes considered:",
            dropped.considered
        );
        println!("  unrebuildable  : {}", dropped.unrebuildable);
        println!("  other model    : {}", dropped.other_model);
        println!("  policy kept it : {}", dropped.policy_kept_the_fact);
        println!("  no re-fetch tgt: {}", dropped.no_refetch_target);
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
    println!("probes considered: {}", dropped.considered);
    println!("  unrebuildable  : {}", dropped.unrebuildable);
    println!("  other model    : {}", dropped.other_model);
    println!("  policy kept it : {}", dropped.policy_kept_the_fact);
    println!("  no re-fetch tgt: {}", dropped.no_refetch_target);
    println!("cases built      : {}", cases.len());
    println!("input tokens     : {toks} across both arms");
    println!("estimated spend  : ${cost:.2}  (published prices, checked 2026-09-14)");
    println!("\neach case is two calls: the intact context as control, then the");
    println!("policy applied. a case only counts if the control reproduces the fact.");

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
        }
        println!(
            "\nall {} cases satisfy the request invariants.",
            cases.len()
        );
        let mut turns: Vec<(usize, u32)> = cases.iter().map(|c| (c.session, c.cut)).collect();
        turns.sort_unstable();
        turns.dedup();
        let mut sess: Vec<usize> = cases.iter().map(|c| c.session).collect();
        sess.sort_unstable();
        sess.dedup();
        println!(
            "{} cases over {} distinct replayed turns in {} sessions.",
            cases.len(),
            turns.len(),
            sess.len()
        );
        if turns.len() < cases.len() {
            println!(
                "cases sharing a turn share a request: one observation scored more than once."
            );
        }
        let (a, b) = counterfactual::estimate_tokens(&cases[0]);
        println!(
            "\nfirst case: {} messages intact ({a} tok), masked ({b} tok)",
            cases[0].messages_intact.len()
        );
        println!(
            "fact length {} chars, origin tool {:?}, target known {}",
            cases[0].fact.len(),
            cases[0].origin_tool,
            !cases[0].origin_target.is_empty()
        );
        println!("\ndry run: nothing was sent.");
        return 0;
    }

    let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("\nANTHROPIC_API_KEY is not set.");
        eprintln!("This needs a real API key. A Claude subscription credential will not do:");
        eprintln!("Anthropic's terms do not permit using Free, Pro or Max OAuth tokens in");
        eprintln!("another tool, so ctxmeter will not read them.");
        return 2;
    }
    if !a.yes {
        eprintln!("\nRefusing to spend ${cost:.2} without --yes.");
        return 2;
    }

    let v = counterfactual::run(&cases, &key, a.show_raw);
    let attempted = v.informative + v.discarded + v.unusable;
    if !v.errors.is_empty() {
        eprintln!("\nThe run stopped: the API did not accept a request.");
        for e in &v.errors {
            eprintln!("  {e}");
        }
        eprintln!("\nThis is a bug in the rebuild, not a result about the policy.");
        if attempted == 0 {
            return 3;
        }
        eprintln!("{attempted} case(s) completed before it, reported below.");
    }
    println!("\n{:<26}{:>8}", "cases attempted", attempted);
    println!("{:<26}{:>8}", "discarded (control failed)", v.discarded);
    println!("{:<26}{:>8}", "unusable (truncated)", v.unusable);
    println!("{:<26}{:>8}", "informative", v.informative);
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
        "went to fetch it",
        v.sought,
        p(v.sought)
    );
    println!("  {:<24}{:>8}{:>9}", "did neither", v.silent, p(v.silent));
    println!("\n'did neither' is the irreversible share: the fact was gone and the model");
    println!("did not ask for it back. 'went to fetch it' is the healthy failure.");
    println!("this still measures the next action, not task success.");
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

fn cmd_robustness(all: &[Session], max_df: usize, min_gap: usize) -> i32 {
    let sessions = eligible(all);
    let probes = harvest(&sessions, max_df, min_gap);
    if probes.iter().all(|v| v.is_empty()) {
        println!("NO PROBES HARVESTED.");
        return 1;
    }
    let pols = default_policies();

    println!("1. COST MODEL. The cost column is a simulation. Does its sign survive");
    println!("   replacing generous prefix matching with the all-or-nothing behaviour");
    println!("   measured on real traces?\n");
    println!(
        "{:<20}{:>16}{:>16}",
        "policy", "longest-prefix", "all-or-nothing"
    );
    for &m in &[CacheModel::LongestPrefix, CacheModel::AllOrNothing] {
        let _ = m;
    }
    for pol in &pols {
        let mut cells = Vec::new();
        for &m in &[CacheModel::LongestPrefix, CacheModel::AllOrNothing] {
            let o = CostOpts {
                model: m,
                scale: 1.0,
            };
            let base = billed_cost_with(&sessions, None, &o);
            let c = billed_cost_with(&sessions, Some(*pol), &o);
            cells.push((1.0 - c / base) * 100.0);
        }
        println!("{:<20}{:>15.1}%{:>15.1}%", pol.label(), cells[0], cells[1]);
    }

    println!("\n2. ESTIMATOR. Per-block sizes are character estimates. Does the sign");
    println!("   survive scaling every visible block by 2x and 3x?\n");
    println!("{:<20}{:>10}{:>10}{:>10}", "policy", "1x", "2x", "3x");
    for pol in &pols {
        let mut cells = Vec::new();
        for sc in [1.0, 2.0, 3.0] {
            let o = CostOpts {
                model: CacheModel::LongestPrefix,
                scale: sc,
            };
            let base = billed_cost_with(&sessions, None, &o);
            let c = billed_cost_with(&sessions, Some(*pol), &o);
            cells.push((1.0 - c / base) * 100.0);
        }
        println!(
            "{:<20}{:>9.1}%{:>9.1}%{:>9.1}%",
            pol.label(),
            cells[0],
            cells[1],
            cells[2]
        );
    }

    println!("\n3. UNCERTAINTY. Probes inside one session share a trajectory, so they");
    println!("   are not independent. 95% interval from a cluster bootstrap over");
    println!("   sessions, 2000 resamples.\n");
    println!(
        "{:<20}{:>10}{:>22}{:>10}",
        "policy", "retained", "95% CI (clustered)", "sessions"
    );
    for pol in &pols {
        let per = retention_by_session(&sessions, &probes, *pol);
        let (kept, total) = retention(&sessions, &probes, *pol);
        if total == 0 {
            continue;
        }
        let (lo, hi) = bootstrap_ci(&per, 2000);
        println!(
            "{:<20}{:>9.1}%{:>14.1}% - {:>4.1}%{:>10}",
            pol.label(),
            kept as f64 / total as f64 * 100.0,
            lo * 100.0,
            hi * 100.0,
            per.len()
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
        Cmd::Robustness { max_df, min_gap } => cmd_robustness(&sessions, max_df, min_gap),
        Cmd::Counterfactual {
            keep_last,
            model,
            sample,
            dry_run,
            yes,
            show_raw,
            max_df,
            min_gap,
        } => cmd_counterfactual(
            &sessions,
            &it.names,
            CfArgs {
                keep_last,
                model,
                sample,
                dry_run,
                yes,
                show_raw,
                max_df,
                min_gap,
            },
        ),
    };
    std::process::exit(code);
}
