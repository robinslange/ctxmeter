# Per-Turn Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay each turn once and grade that one response against every fact the policy destroyed in it, instead of issuing a separate request per fact.

**Architecture:** `Case` stops being "one fact" and becomes "one replayed turn holding many facts". The two requests per turn stay; the grading loop moves inside. Selection switches from striding over probes to round-robin over sessions' turns, so `--sample` counts observations rather than scorings. The three-way outcome split stays per fact, and the reported n becomes distinct turns and sessions.

**Tech Stack:** Rust 2021 (rust-version 1.85), `clap`, `serde_json`, `walkdir`. No new dependencies. Requests go out through `curl` with headers in a 0600 config file.

**Spec:** `README.md`, section "Tier two: did losing it matter?" — it states what the command claims to measure, and three of its claims are what this plan makes true.

## Global Constraints

- No new dependencies. `Cargo.toml` stays as it is.
- `cargo fmt`, `cargo clippy --release --all-targets` (zero warnings), and `cargo test --release` all pass before each commit. CI runs all three plus a smoke job against `tests/fixture`.
- **No probe token, absolute path, repository name, or username from a real trace may enter this repository.** Tests use synthetic paths (`/home/dev/...`). This is the privacy promise in `README.md`; it has already been violated once and corrected.
- `--dry-run` sends nothing over the network. Ever. The pre-flight (`count_tokens`) is part of a paid run, not the dry run.
- Comments explain why, not what. No `// Note:` or change-log comments.
- Every measured figure quoted below came from this corpus on 2026-09-14 and is reproducible with `--dry-run`. The corpus is live and grows during use, so counts drift between runs; ratios are stable.

## Why this shape

Measured on the full eligible set (`--model sonnet`, `keep_last_3`, `--dry-run --sample 2000`):

| quantity | value |
|---|---|
| probes considered | 3,060 |
| dropped, policy kept the fact | 1,044 |
| dropped, no re-fetch target | 468 |
| cases built | 1,548 |
| distinct replayed turns | 878 |
| sessions | 163 |
| facts per turn | 1.76 |

The cost argument is real but secondary: 1,548 requests collapse to 878, and the
duplicates that go away were already the cheap ones, since a second request on the
same prefix reads from cache at 0.1x (observed: `cache_read_input_tokens` 166,623
on two consecutive cases sharing a turn).

The correctness argument is the reason to do it. Today, N facts reused in one
assistant turn produce N cases that issue N *separate* control requests against a
byte-identical prefix. Each draws an independent sample of a non-deterministic
response, so the same replayed turn can return contradictory verdicts — one
control reproducing, another not, for the same context. Measured control
reproduction rate is 1 in 7, which makes that divergence the common case rather
than a corner. One turn is one event. Grading one response N ways says so; issuing
N requests denies it.

The yield gain follows: one paid control call is tested against 1.76 facts instead
of 1, so informative observations per dollar rise without touching which turns get
selected.

**The trap to avoid:** do not select turns by how many facts they carry. Turns that
reuse many facts are long-form document writes, which are not a random subset of
agent behaviour. The saving comes from not re-requesting a response already in
hand, not from preferring fact-dense turns.

---

### Task 1: Refuse a run that spans model families

`README.md` says "A run covers one model family, because pooling two families into
a single retention figure conflates them." The filter is a substring match, so
`--model sonnet` selected 1,547 `claude-sonnet-5` cases and 1 `claude-sonnet-4-5-20250929`
case, which is the conflation the README forbids. It also misprices: `price()`
charges the sonnet-5 rate to both, and sonnet-4-5 is $3/$15 against sonnet-5's $2/$10.

**Files:**
- Modify: `src/main.rs` — `cmd_counterfactual`, immediately after `build_cases` returns and the drop breakdown is printed

**Interfaces:**
- Consumes: `counterfactual::Case.model` (`String`), already public
- Produces: nothing new. Exit code 2 for a refused run, matching the other refusals in this function

- [ ] **Step 1: Add the guard**

In `src/main.rs`, after the `cases.is_empty()` block and before `let cost = counterfactual::estimate_cost(&cases);`:

```rust
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
```

- [ ] **Step 2: Verify it refuses the pooled filter**

Run: `./target/release/ctxmeter counterfactual --dry-run --sample 2000 --model sonnet; echo "exit=$?"`
Expected: the two model ids listed, `exit=2`.

- [ ] **Step 3: Verify the narrowed filter still runs**

Run: `./target/release/ctxmeter counterfactual --dry-run --sample 40 --model sonnet-5 | tail -3`
Expected: a normal dry run ending in `dry run: nothing was sent.`

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -F - <<'MSG'
counterfactual: refuse a run whose cases span two model families

A substring filter matched claude-sonnet-4-5 alongside claude-sonnet-5, which
is the pooling the README says invalidates a retention figure, and the two are
priced differently.
MSG
```

---

### Task 2: Drop an unanswered tool_use instead of building an unsendable request

3 of 1,548 cases carry a `tool_use` whose `tool_result` was never recorded, in the
middle of the message array rather than at the end where the existing trim would
catch it. The API rejects the whole request, and before the invariant check existed
that rejection would have been counted as a control arm failing to reproduce a fact.

**Files:**
- Modify: `src/counterfactual.rs` — `rebuild`, after `have` is populated and before the tail trim
- Test: `src/counterfactual.rs` — `mod tests`

**Interfaces:**
- Consumes: `have: HashSet<String>` (already built in `rebuild`), `invalid(&[Value]) -> Option<String>` (already public)
- Produces: no signature change. `rebuild` keeps returning `Option<Rebuilt>`

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/counterfactual.rs`:

```rust
/// A call whose result was never recorded makes the whole request invalid, and
/// the API rejects the request rather than the block. The rest of the turn is
/// still what the agent did, so drop the block and keep the case.
#[test]
fn a_call_with_no_recorded_result_is_dropped_not_kept() {
    let p = std::env::temp_dir().join("ctxmeter-orphan-test.jsonl");
    let lines = [
        r#"{"uuid":"a1","message":{"role":"user","content":[{"type":"text","text":"go"}]}}"#,
        r#"{"uuid":"a2","message":{"role":"assistant","content":[{"type":"text","text":"looking"},{"type":"tool_use","id":"lost","name":"Read","input":{"file_path":"/home/dev/x.md"}}]}}"#,
        r#"{"uuid":"a3","message":{"role":"assistant","content":[{"type":"tool_use","id":"kept","name":"Read","input":{"file_path":"/home/dev/y.md"}}]}}"#,
        r#"{"uuid":"a4","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"kept","content":"body"}]}}"#,
    ];
    std::fs::write(&p, lines.join("\n")).unwrap();
    let (msgs, _, _) = rebuild(&p, 9).expect("rebuildable");
    let _ = std::fs::remove_file(&p);
    assert_eq!(invalid(&msgs), None);
    let whole = serde_json::to_string(&msgs).unwrap();
    assert!(!whole.contains("\"lost\""), "the unanswered call survived");
    assert!(whole.contains("\"kept\""), "the answered call was dropped too");
    assert!(whole.contains("looking"), "text in the same message was lost");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release a_call_with_no_recorded_result -- --nocapture`
Expected: FAIL on `assert_eq!(invalid(&msgs), None)` with `tool_use lost has no result`.

- [ ] **Step 3: Write the implementation**

In `rebuild`, directly after the loop that fills `have` and before the comment about the last message being a user turn:

```rust
// A tool_use whose result was never recorded invalidates the whole request, not
// just the block, and the API rejects all of it. Drop the block and keep the
// rest of the turn, which is still what the agent did. A message left with no
// content goes with it.
for m in &mut msgs {
    if let Some(arr) = m["content"].as_array_mut() {
        arr.retain(|b| {
            b["type"] != "tool_use"
                || b["id"].as_str().is_some_and(|i| have.contains(i))
        });
    }
}
msgs.retain(|m| m["content"].as_array().is_some_and(|a| !a.is_empty()));
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release`
Expected: PASS, 13 tests.

- [ ] **Step 5: Verify no arm in the corpus is rejected any more**

Run: `./target/release/ctxmeter counterfactual --dry-run --sample 2000 --model sonnet-5 | grep -E "would be rejected|satisfy the request"`
Expected: `all N cases satisfy the request invariants.` and no rejection line.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --release --all-targets && cargo test --release
git add src/counterfactual.rs
git commit -F - <<'MSG'
counterfactual: drop a call whose result was never recorded

Three cases in the corpus carried a tool_use with no tool_result, mid-array
where the tail trim could not see it. The API rejects the whole request for it.
MSG
```

---

### Task 3: A case becomes a turn holding many facts

**Files:**
- Modify: `src/counterfactual.rs` — `Case`, new `Fact`, `build_cases`, `Dropped`, `estimate_tokens` unchanged
- Modify: `src/main.rs` — the dry-run reporting that reads `cases[0].fact` and `cases[0].origin_tool`
- Test: `src/counterfactual.rs` — `mod tests`

**Interfaces:**
- Consumes: `Policy::survives_from_indexed`, `rebuild`, `apply_mask` — all unchanged
- Produces:
  - `pub struct Fact { pub text: String, pub origin: String }`
  - `Case` fields become `pub facts: Vec<Fact>`, replacing `pub fact: String`, `pub origin_tool: String`, `pub origin_target: String`. `session`, `cut`, `model`, `messages_intact`, `messages_masked`, `tools` are unchanged.
  - `Dropped` gains `pub turns_considered: usize`; `considered` keeps counting candidate facts.
  - `build_cases(&Corpus, Policy, sample: usize, model_filter: &str, &mut Dropped) -> Vec<Case>` keeps its signature; `sample` now counts turns.

- [ ] **Step 1: Write the failing test**

Add to `mod tests`:

```rust
/// Four facts reused in one assistant turn are one replayed request, not four.
/// Issuing four would draw four independent samples of a non-deterministic
/// response and could return contradictory verdicts for the same context.
#[test]
fn facts_reused_in_one_turn_become_one_case() {
    let c = Case {
        session: 3,
        cut: 40,
        model: "claude-sonnet-5".into(),
        facts: vec![
            Fact { text: "a7f3c9e21b84".into(), origin: "/home/dev/build.log".into() },
            Fact { text: "b81d0c4a9f27".into(), origin: "/home/dev/build.log".into() },
        ],
        messages_intact: vec![],
        messages_masked: vec![],
        tools: vec![],
    };
    assert_eq!(c.facts.len(), 2);
    assert_eq!(c.cut, 40);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release facts_reused_in_one_turn`
Expected: FAIL to compile — `Fact` not found, `Case` has no field `facts`.

- [ ] **Step 3: Define `Fact` and reshape `Case`**

Replace the `Case` struct in `src/counterfactual.rs`:

```rust
/// One destroyed fact and the file the call that produced it named. Seeking it
/// means naming that file again, by any tool: `cat` and `Read` fetch the same
/// thing, and the tool name alone is not evidence of anything in a corpus that
/// is mostly `Read`.
pub struct Fact {
    pub text: String,
    pub origin: String,
}

/// One replayed turn: the request as the agent saw it, the same request with the
/// policy applied, and every fact the policy destroyed in it. The turn is the
/// observation. The facts are what that one observation is scored against.
pub struct Case {
    pub session: usize,
    pub cut: u32,
    pub model: String,
    pub facts: Vec<Fact>,
    pub messages_intact: Vec<serde_json::Value>,
    pub messages_masked: Vec<serde_json::Value>,
    pub tools: Vec<serde_json::Value>,
}
```

- [ ] **Step 4: Add the turn counter to `Dropped`**

```rust
#[derive(Default)]
pub struct Dropped {
    pub turns_considered: usize,
    pub considered: usize,
    pub unrebuildable: usize,
    pub policy_kept_the_fact: usize,
    pub other_model: usize,
    pub no_refetch_target: usize,
}
```

- [ ] **Step 5: Group probes by turn in `build_cases`**

Replace the body of `build_cases` from `let mut flat` through the `out.push(...)` loop with:

```rust
    // Probes reused in the same assistant turn replay as the same request, so
    // group them before anything is selected or sent.
    let mut turns: HashMap<(usize, u32), Vec<&Probe>> = HashMap::new();
    for (si, ps) in probes.iter().enumerate() {
        // Pooling two model families into one retention figure conflates them,
        // so a run covers one family. It is also 5x cheaper.
        let model_ok = sessions[si]
            .usage
            .first()
            .map(|u| u.model.contains(model_filter))
            .unwrap_or(false);
        if !model_ok {
            dropped.other_model += ps.len();
            continue;
        }
        for p in ps {
            let cut = sessions[si].blocks.get(p.use_at).map(|b| b.msg).unwrap_or(0);
            turns.entry((si, cut)).or_default().push(p);
        }
    }
    let mut order: Vec<(usize, u32)> = turns.keys().copied().collect();
    order.sort_unstable();

    let mut out: Vec<Case> = Vec::new();
    for (si, cut) in order {
        if out.len() >= sample {
            break;
        }
        let ps = &turns[&(si, cut)];
        dropped.turns_considered += 1;
        dropped.considered += ps.len();
        let path = Path::new(&paths[si]);
        let Some((intact, tools, origin)) = rebuild(path, cut) else {
            dropped.unrebuildable += 1;
            continue;
        };
        let masked = apply_mask(&intact, &pol);
        let mut facts: Vec<Fact> = Vec::new();
        for p in ps {
            let text = interned[p.tok as usize].clone();
            // A fact is only usable if the policy actually removed it.
            if masked.iter().any(|m| m["content"].to_string().contains(&text)) {
                dropped.policy_kept_the_fact += 1;
                continue;
            }
            // Find the tool_result carrying the fact, then the call that produced
            // it. Any other call in the session would mislabel a re-fetch.
            let mut found = String::new();
            'find: for m in &intact {
                for b in m["content"].as_array().into_iter().flatten() {
                    if b["type"] == "tool_result" && b["content"].to_string().contains(&text) {
                        if let Some((_, t)) = b["tool_use_id"].as_str().and_then(|id| origin.get(id))
                        {
                            found = t.clone();
                        }
                        break 'find;
                    }
                }
            }
            if found.is_empty() {
                dropped.no_refetch_target += 1;
                continue;
            }
            facts.push(Fact { text, origin: found });
        }
        if facts.is_empty() {
            continue;
        }
        let model = sessions[si]
            .usage
            .first()
            .map(|u| u.model.clone())
            .unwrap_or_else(|| "claude-sonnet-5".into());
        out.push(Case {
            session: si,
            cut,
            model,
            facts,
            messages_intact: intact,
            messages_masked: masked,
            tools,
        });
    }
    out
```

Delete the trailing `out.sort_by_key(...)` and the `.map(|(_, _, c)| c)` collect: ordering by session and cut is now done up front by `order.sort_unstable()`, which keeps consecutive turns from one session adjacent so their control arms still extend each other's cached prefix.

- [ ] **Step 6: Update the dry-run reporting in `src/main.rs`**

Delete the `turns` / `sess` dedup block added while diagnosing this (the six lines
from `let mut turns: Vec<(usize, u32)>` through the `if turns.len() < cases.len()`
branch, inclusive). With one case per turn, `turns.len() == cases.len()` always and
the warning it prints can no longer fire.

Then replace both of these `println!` calls — the one opening `"\nfirst case:"` and
the `"fact length {} chars, origin tool {:?}, target known {}"` one that follows it,
together with the `let (a, b) = ...` line above them — with:

```rust
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
```

Replace `println!("  policy kept it : {}", dropped.policy_kept_the_fact);` and its
neighbours with a breakdown that names the units, in both the empty-cases branch
and the main one:

```rust
    println!("turns considered  : {}", dropped.turns_considered);
    println!("  unrebuildable   : {}", dropped.unrebuildable);
    println!("facts considered  : {}", dropped.considered);
    println!("  policy kept it  : {}", dropped.policy_kept_the_fact);
    println!("  no re-fetch tgt : {}", dropped.no_refetch_target);
    println!("  other model     : {}", dropped.other_model);
    println!("turns built       : {}", cases.len());
```

- [ ] **Step 7: Update `--sample` help text**

In `src/main.rs`, the `Counterfactual` variant:

```rust
        /// Turns to replay. A turn is one observation; several facts destroyed in
        /// the same turn are scored against its one response, not re-requested.
        #[arg(long, default_value_t = 40)]
        sample: usize,
```

- [ ] **Step 8: Run the tests**

Run: `cargo fmt && cargo clippy --release --all-targets && cargo test --release`
Expected: PASS. `grade`, `run`, and `Verdict` still reference `c.fact` and will not compile — fix them in Task 4, so expect compilation failures confined to those three and complete Task 4 before committing.

- [ ] **Step 9: Commit (after Task 4 compiles)**

This task and Task 4 land together, because `Case` losing `fact` breaks `grade` in
the same change. Commit once at the end of Task 4.

---

### Task 4: Grade one response against every fact of the turn

**Files:**
- Modify: `src/counterfactual.rs` — `grade`, `unusable`, `Verdict`, `show`, `run`, `estimate_cost`
- Modify: `src/main.rs` — the result table
- Test: `src/counterfactual.rs` — `mod tests`, updating the four grading tests to take a `Fact`

**Interfaces:**
- Consumes: `Fact`, `Case` from Task 3; `call`, `count_tokens`, `action_blocks`, `price`, `MAX_TOKENS` unchanged
- Produces:
  - `fn grade(resp: &Value, f: &Fact) -> Outcome` — was `(&Value, &Case)`
  - `fn unusable(resp: &Value, g: Outcome) -> Option<String>` — unchanged
  - `Verdict` fields: `turns_attempted`, `turns_informative`, `sessions`, `informative`, `discarded`, `unusable`, `reproduced`, `sought`, `silent`, `errors`, `measured_tokens` — all `usize` except `errors: Vec<String>` and `measured_tokens: u64`. `informative`, `discarded`, `unusable`, `reproduced`, `sought`, `silent` count **facts**; `turns_*` count turns.
  - `pub fn run(cases: &[Case], api_key: &str, show_raw: bool) -> Verdict` — unchanged signature

- [ ] **Step 1: Update the grading tests to take a Fact**

Replace `fn case() -> Case { ... }` in `mod tests` with:

```rust
fn fact() -> Fact {
    Fact {
        text: "a7f3c9e21b84".into(),
        origin: "/home/dev/build.log".into(),
    }
}
```

Then in `a_control_that_does_not_reproduce_the_fact_is_not_informative`,
`going_back_for_the_fact_means_going_back_to_its_origin`, and
`a_fact_recalled_while_thinking_is_not_a_reuse`, replace `let c = case();` with
`let c = fact();`. The `grade(&got, &c)` calls are unchanged. In
`going_back_for_the_fact_means_going_back_to_its_origin`, the `"/tmp/build.log"`
literals become `"/home/dev/build.log"`.

- [ ] **Step 2: Add the test for one response scored many ways**

```rust
/// The point of the change: one replayed response, several facts. A turn that
/// reproduces one fact and not another is one observation with two outcomes,
/// which is what it always was. Two requests would have made it two events and
/// could have disagreed with itself.
#[test]
fn one_response_is_graded_against_each_fact_of_the_turn() {
    let here = Fact { text: "a7f3c9e21b84".into(), origin: "/home/dev/build.log".into() };
    let gone = Fact { text: "b81d0c4a9f27".into(), origin: "/home/dev/other.log".into() };
    let got = resp(serde_json::json!([
        {"type": "text", "text": "build a7f3c9e21b84 failed"}
    ]));
    assert_eq!(grade(&got, &here), Outcome::Reproduced);
    assert_eq!(grade(&got, &gone), Outcome::Silent);
}
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test --release`
Expected: FAIL to compile — `grade` takes `&Case`.

- [ ] **Step 4: Change `grade` to take a Fact**

```rust
fn grade(resp: &serde_json::Value, f: &Fact) -> Outcome {
    let blocks = action_blocks(resp);
    let whole = serde_json::to_string(&blocks).unwrap_or_default();
    if whole.contains(&f.text) {
        return Outcome::Reproduced;
    }
    for b in blocks {
        // Seeking the fact means going back to what produced it. The tool name
        // alone cannot carry that: a corpus of coding sessions is mostly Read,
        // so matching on it would score almost any next action as a re-fetch and
        // quietly move the result into the healthy bucket. Only the target does
        // it, and any tool that names the target counts, because cat and Read
        // fetch the same file.
        if b["type"] == "tool_use" && b["input"].to_string().contains(&f.origin) {
            return Outcome::Sought;
        }
    }
    Outcome::Silent
}
```

- [ ] **Step 5: Reshape `Verdict`**

```rust
pub struct Verdict {
    pub turns_attempted: usize,
    /// Turns where at least one fact's control arm reproduced it. The unit an
    /// interval may bootstrap over is the session, and this bounds it.
    pub turns_informative: usize,
    pub sessions: usize,
    pub informative: usize,
    pub discarded: usize,
    pub unusable: usize,
    pub reproduced: usize,
    pub sought: usize,
    pub silent: usize,
    /// Requests the API refused to accept, or that never left the machine. Kept
    /// apart from `discarded`: a 400 is a bug in this tool, not a finding about
    /// the policy, and pooling the two makes the discard rate unreadable.
    pub errors: Vec<String>,
    /// Measured by count_tokens, against which the estimate can be checked.
    pub measured_tokens: u64,
}
```

- [ ] **Step 6: Rewrite the paid loop in `run`**

Keep the pre-flight block and the ceiling print exactly as they are. Replace the
`for (i, c) in cases.iter().enumerate()` paid loop body with:

```rust
    let mut sessions: Vec<usize> = cases.iter().map(|c| c.session).collect();
    sessions.sort_unstable();
    sessions.dedup();
    v.sessions = sessions.len();

    for (i, c) in cases.iter().enumerate() {
        if show_raw {
            println!(
                "\nturn {i}: session {}, cut {}, {} / {} tokens, {} fact(s)",
                c.session,
                c.cut,
                sizes[i].0,
                sizes[i].1,
                c.facts.len()
            );
            for f in &c.facts {
                println!("  fact ({} chars): {:?}", f.text.len(), f.text);
                println!("    seeking it means naming {:?}", f.origin);
            }
        }
        // Control first. A fact the intact context does not reproduce cannot tell
        // us anything about the policy, and it is scored per fact because one
        // response can reproduce one and miss another.
        let ctrl = match call(api_key, &c.model, &c.messages_intact, &c.tools) {
            Ok(r) => r,
            Err(e) => {
                v.errors.push(format!("turn {i} control: {e}"));
                return v;
            }
        };
        v.turns_attempted += 1;
        let mut live: Vec<&Fact> = Vec::new();
        for f in &c.facts {
            let g = grade(&ctrl, f);
            if show_raw {
                println!("  control graded {g:?} for {:?}", f.text);
            }
            if let Some(why) = unusable(&ctrl, g) {
                println!("turn {i} fact unusable: {why}");
                v.unusable += 1;
                continue;
            }
            if g != Outcome::Reproduced {
                v.discarded += 1;
                continue;
            }
            live.push(f);
        }
        if show_raw {
            show("control (context intact)", &ctrl);
        }
        if live.is_empty() {
            continue;
        }
        v.turns_informative += 1;
        let treat = match call(api_key, &c.model, &c.messages_masked, &c.tools) {
            Ok(r) => r,
            Err(e) => {
                v.errors.push(format!("turn {i} treatment: {e}"));
                return v;
            }
        };
        if show_raw {
            show("treatment (fact removed)", &treat);
        }
        for f in live {
            let g = grade(&treat, f);
            if show_raw {
                println!("  treatment graded {g:?} for {:?}", f.text);
            }
            if let Some(why) = unusable(&treat, g) {
                println!("turn {i} fact unusable: {why}");
                v.unusable += 1;
                continue;
            }
            v.informative += 1;
            match g {
                Outcome::Reproduced => v.reproduced += 1,
                Outcome::Sought => v.sought += 1,
                Outcome::Silent => v.silent += 1,
            }
        }
    }
    v
```

- [ ] **Step 7: Drop the grade argument from `show`**

`show` printed the grade in its header; grades are now per fact and printed
beside each one, so the header stops claiming a single verdict:

```rust
fn show(label: &str, resp: &serde_json::Value) {
    println!("\n--- {label} ---");
    println!(
        "stop_reason {:?}   usage {}",
        resp.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("?"),
        resp.get("usage").map(|u| u.to_string()).unwrap_or_default()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(resp.get("content").unwrap_or(resp)).unwrap_or_default()
    );
}
```

- [ ] **Step 8: Fix `estimate_cost` for the per-turn shape**

The output bound is two arms per turn rather than two per fact:

```rust
pub fn estimate_cost(cases: &[Case]) -> f64 {
    let mut usd = 0.0;
    for c in cases {
        let (a, b) = estimate_tokens(c);
        let (inp, outp) = price(&c.model);
        usd += (a + b) as f64 / 1e6 * inp;
        usd += 2.0 * MAX_TOKENS as f64 / 1e6 * outp; // both arms, at the ceiling
    }
    usd
}
```

- [ ] **Step 9: Update the result table in `src/main.rs`**

Replace `let attempted = v.informative + v.discarded + v.unusable;` with
`let attempted = v.turns_attempted;`. The error block below it is unchanged: a
pre-flight rejection leaves `turns_attempted` at 0 and still returns 3, and a
rejection part way through reports the turns that completed before it.

Then replace the five-row result table with:

```rust
    println!("\n{:<30}{:>8}", "turns replayed", v.turns_attempted);
    println!("{:<30}{:>8}", "turns with a live control", v.turns_informative);
    println!("{:<30}{:>8}", "sessions", v.sessions);
    println!("{:<30}{:>8}", "facts discarded (control)", v.discarded);
    println!("{:<30}{:>8}", "facts unusable (truncated)", v.unusable);
    println!("{:<30}{:>8}", "facts informative", v.informative);
```

and after the three-way split, replace the closing note with:

```rust
    println!("\n'did neither' is the irreversible share: the fact was gone and the model");
    println!("did not ask for it back. 'went to fetch it' is the healthy failure.");
    println!(
        "\n{} facts over {} turns in {} sessions. facts sharing a turn share one",
        v.informative, v.turns_informative, v.sessions
    );
    println!("response, and facts sharing an origin share their 'went to fetch it'");
    println!("verdict, so an interval bootstraps over sessions and not over facts.");
```

- [ ] **Step 10: Run everything**

Run: `cargo fmt && cargo clippy --release --all-targets && cargo test --release`
Expected: PASS, 14 tests, zero warnings.

- [ ] **Step 11: Verify the dry run against the measured baseline**

Run: `./target/release/ctxmeter counterfactual --dry-run --sample 2000 --model sonnet-5 | grep -E "turns built|facts over|satisfy"`
Expected: about 878 turns and about 1,548 facts, near 1.76 per turn. A large
departure from those figures means the grouping key is wrong, not that the corpus
moved — corpus drift changes counts by single digits over minutes, not by hundreds.

- [ ] **Step 12: Commit Tasks 3 and 4 together**

```bash
git add src/counterfactual.rs src/main.rs
git commit -F - <<'MSG'
counterfactual: replay a turn once and score it against every fact it destroyed

Facts reused in the same assistant turn replay as a byte-identical request. One
per fact drew an independent sample of a non-deterministic response, so the same
context could return contradictory verdicts; measured control reproduction is
1 in 7, which makes that the common case. 1,548 requests become 878, the facts
per turn are 1.76, and the reported n is turns and sessions rather than facts.
MSG
```

---

### Task 5: Select turns round-robin across sessions

`build_cases` carried the comment "Deterministic spread across sessions rather than
the first N of one" while ordering by position within a trace, which interleaves
sessions only when they are of comparable length. They are not: at `--sample 5`,
5 cases came from 3 sessions and 4 of them from one turn. Task 3 replaced that
ordering with `(session, cut)`, which is worse for spread — it takes every turn of
session 0 before touching session 1 — and better for cache reuse. This task gets
both.

**Files:**
- Modify: `src/counterfactual.rs` — `build_cases`, the `order` computation from Task 3
- Test: `src/counterfactual.rs` — `mod tests`

**Interfaces:**
- Consumes: `turns: HashMap<(usize, u32), Vec<&Probe>>` from Task 3
- Produces: `fn round_robin(keys: &[(usize, u32)]) -> Vec<(usize, u32)>`

- [ ] **Step 1: Write the failing test**

```rust
/// One turn from every session before a second from any, so a small sample is
/// not three observations from one long research session wearing five hats.
/// Within a session the original order survives, which keeps consecutive control
/// arms extending each other's cached prefix.
#[test]
fn selection_takes_one_turn_per_session_before_a_second() {
    let keys = [(0, 10), (0, 20), (0, 30), (1, 5), (2, 7), (2, 9)];
    assert_eq!(
        round_robin(&keys),
        vec![(0, 10), (1, 5), (2, 7), (0, 20), (2, 9), (0, 30)]
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release selection_takes_one_turn`
Expected: FAIL to compile — `round_robin` not found.

- [ ] **Step 3: Implement it**

Add above `build_cases`:

```rust
/// Interleave turns so that a sample of n covers as many sessions as it can.
/// Ordering by session takes every turn of the first before any of the second,
/// and ordering by position within a trace spreads nothing, because one long
/// session holds most of the eligible turns at every position.
fn round_robin(keys: &[(usize, u32)]) -> Vec<(usize, u32)> {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    let mut ranked: Vec<(usize, usize, u32)> = Vec::with_capacity(sorted.len());
    let mut rank = 0;
    for (i, &(s, c)) in sorted.iter().enumerate() {
        rank = if i > 0 && sorted[i - 1].0 == s { rank + 1 } else { 0 };
        ranked.push((rank, s, c));
    }
    ranked.sort_unstable();
    ranked.into_iter().map(|(_, s, c)| (s, c)).collect()
}
```

- [ ] **Step 4: Use it in `build_cases`**

Replace:

```rust
    let mut order: Vec<(usize, u32)> = turns.keys().copied().collect();
    order.sort_unstable();
```

with:

```rust
    let keys: Vec<(usize, u32)> = turns.keys().copied().collect();
    let order = round_robin(&keys);
```

- [ ] **Step 5: Run the tests**

Run: `cargo test --release`
Expected: PASS, 15 tests.

- [ ] **Step 6: Verify the spread on real data**

Run: `./target/release/ctxmeter counterfactual --dry-run --sample 5 --model sonnet-5 | grep -E "turns built|distinct replayed|facts over"`
Expected: 5 turns in 5 sessions, where before the change 5 cases came from 3.

- [ ] **Step 7: Commit**

```bash
cargo fmt && cargo clippy --release --all-targets && cargo test --release
git add src/counterfactual.rs
git commit -F - <<'MSG'
counterfactual: take one turn per session before a second from any

The old ordering claimed a spread across sessions and did not deliver one: at
sample 5 it returned three sessions, four of whose cases were a single turn.
MSG
```

---

### Task 6: Say where a reproduction happened

The one control arm that reproduced a fact in eight measured attempts did it
inside an `Edit` regenerating a multi-thousand-word report that happened to
contain the string. That counts — the fact came back — but it is a weaker thing
than using a fact to take an action, and a retention figure built mostly out of
long-form regeneration means something different from one built out of tool calls.
The README's own framing is "the task is the agent's own next action", so the
report should distinguish the two rather than let the reader assume.

**Files:**
- Modify: `src/counterfactual.rs` — `Outcome`, `grade`, `Verdict`, `run`
- Modify: `src/main.rs` — the three-way split output
- Modify: `README.md` — the outcome list under "Tier two"
- Test: `src/counterfactual.rs` — `mod tests`

**Interfaces:**
- Consumes: `grade`, `Fact` from Task 4
- Produces: `Outcome::Reproduced` gains a sibling `Outcome::Regenerated`; `Verdict` gains `pub regenerated: usize`

- [ ] **Step 1: Write the failing test**

```rust
/// A fact inside a tool call is the agent acting on it. A fact inside prose may
/// be a document being rewritten around it. Both are reproductions, and a figure
/// that cannot tell them apart hides which one it is made of.
#[test]
fn a_fact_in_prose_is_told_apart_from_a_fact_in_an_action() {
    let f = fact();
    let acted = resp(serde_json::json!([
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "grep a7f3c9e21b84 /home/dev/other.log"}}
    ]));
    assert_eq!(grade(&acted, &f), Outcome::Reproduced);
    let written = resp(serde_json::json!([
        {"type": "text", "text": "the failing build was a7f3c9e21b84, as reported"}
    ]));
    assert_eq!(grade(&written, &f), Outcome::Regenerated);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release a_fact_in_prose`
Expected: FAIL to compile — no variant `Regenerated`.

- [ ] **Step 3: Add the variant and split the check**

```rust
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Outcome {
    /// The fact came back inside an action the model took.
    Reproduced,
    /// The fact came back in prose. Counted, and counted separately: a document
    /// regenerated around a string is a weaker reuse than acting on it.
    Regenerated,
    /// The model went to fetch it again. Healthy: it noticed something missing.
    Sought,
    /// Neither. The fact is gone and the model did not ask for it.
    Silent,
}
```

In `grade`, replace the single `whole.contains` check:

```rust
    let acted = blocks
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .any(|b| b["input"].to_string().contains(&f.text));
    if acted {
        return Outcome::Reproduced;
    }
    let whole = serde_json::to_string(&blocks).unwrap_or_default();
    if whole.contains(&f.text) {
        return Outcome::Regenerated;
    }
```

- [ ] **Step 4: Treat both as a live control, and count them apart in the treatment**

In `run`, the control filter becomes:

```rust
            if !matches!(g, Outcome::Reproduced | Outcome::Regenerated) {
                v.discarded += 1;
                continue;
            }
```

and the treatment match gains the arm:

```rust
            match g {
                Outcome::Reproduced => v.reproduced += 1,
                Outcome::Regenerated => v.regenerated += 1,
                Outcome::Sought => v.sought += 1,
                Outcome::Silent => v.silent += 1,
            }
```

with `pub regenerated: usize` added to `Verdict` and initialised to 0 in `run`.

- [ ] **Step 5: Report it**

In `src/main.rs`, between the `reproduced it anyway` and `went to fetch it` rows:

```rust
    println!(
        "  {:<24}{:>8}{:>9}",
        "rewrote it into prose",
        v.regenerated,
        p(v.regenerated)
    );
```

- [ ] **Step 6: Update the README outcome list**

Under "Of the cases that survive, the outcome splits three ways", change the
count and add the row:

```markdown
- **Reproduced anyway.** The model did not need the context to get there, and it
  acted on the fact.
- **Rewrote it into prose.** The fact came back inside text rather than inside an
  action. Counted apart, because a document regenerated around a string is a
  weaker reuse than acting on it, and on the first measured corpus this was the
  only way a control arm ever reproduced anything.
```

and change "splits three ways" to "splits four ways".

- [ ] **Step 7: Run everything**

Run: `cargo fmt && cargo clippy --release --all-targets && cargo test --release`
Expected: PASS, 16 tests, zero warnings.

- [ ] **Step 8: Commit**

```bash
git add src/counterfactual.rs src/main.rs README.md
git commit -F - <<'MSG'
counterfactual: separate a fact acted on from a fact rewritten into prose

The only control arm that reproduced anything in eight measured attempts did it
inside an Edit regenerating a report that happened to contain the string. Both
are reproductions; a figure that cannot tell them apart hides which it is made of.
MSG
```

---

## What this plan does not do

- **It does not raise the yield.** The control arm reproduces a given fact about 1
  time in 7, because the next action is not determined by the context and a replay
  draws a different sample. Grouping raises informative facts per paid call, since
  one response is scored against 1.76 facts; it does not make a replay more likely
  to agree with the trace. Sizing a run still starts from informative facts wanted,
  not turns.
- **It does not calibrate the estimate.** `estimate_tokens` divides serialized
  length by 4 and ran 1.51x and 1.59x low against `count_tokens` on two measured
  runs. The pre-flight now measures every arm for free before any spend, so the
  estimate only serves `--dry-run`; the README says it is approximate in both
  directions rather than an upper bound.
- **It does not make a run reproducible over time.** The corpus is the live
  `~/.claude/projects` tree and grows while the tool is used, so two runs minutes
  apart are not the same experiment. Selection is deterministic given a corpus.
- **It does not address the truncation concentration.** A truncated control arm
  now invalidates only the facts it graded silent, but one truncated response takes
  all of that turn's silent facts with it where separate requests might have
  truncated on some and not others.
