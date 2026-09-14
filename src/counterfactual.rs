//! Tier two: does a destroyed fact actually change what the agent does?
//!
//! The design avoids authoring a question and avoids a grader. At the block
//! where the agent reused a fact, it had already produced that fact itself from
//! that context. So replay that exact turn twice, once with the context intact
//! and once with the policy applied, and check whether the literal string comes
//! back. The task is the agent's own next action; the ground truth is what it
//! actually did. No model judges anything.
//!
//! The intact arm is the control. If it fails to reproduce the fact, the probe
//! is uninformative and is discarded, and that rate is reported rather than
//! hidden.

use crate::probes::{Policy, Probe};
use crate::transcript::Session;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::Command;

const MASK: &str = "[tool output removed by the context policy under test]";

/// Published input/output prices per million tokens, checked 2026-09-14.
/// Verify against the current pricing page before trusting an estimate.
fn price(model: &str) -> (f64, f64) {
    if model.contains("opus") {
        (15.0, 75.0)
    } else if model.contains("haiku") {
        (1.0, 5.0)
    } else {
        (3.0, 15.0)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Outcome {
    /// The fact came back in the model's own next action.
    Reproduced,
    /// The model went to fetch it again. Healthy: it noticed something missing.
    Sought,
    /// Neither. The fact is gone and the model did not ask for it.
    Silent,
}

/// Why candidate probes never became cases. Reported, not hidden: a sample whose
/// losses are invisible cannot be audited.
#[derive(Default)]
pub struct Dropped {
    pub considered: usize,
    pub unrebuildable: usize,
    pub policy_kept_the_fact: usize,
    pub other_model: usize,
}

pub struct Case {
    pub fact: String,
    pub model: String,
    pub origin_tool: String,
    pub origin_target: String,
    pub messages_intact: Vec<serde_json::Value>,
    pub messages_masked: Vec<serde_json::Value>,
    pub tools: Vec<serde_json::Value>,
}

fn whitelist(block: &serde_json::Value) -> Option<serde_json::Value> {
    let t = block.get("type")?.as_str()?;
    let mut out = serde_json::Map::new();
    out.insert("type".into(), t.into());
    match t {
        "text" => {
            let s = block.get("text")?.as_str().unwrap_or("");
            if s.trim().is_empty() {
                return None;
            }
            out.insert("text".into(), s.into());
        }
        "tool_use" => {
            out.insert("id".into(), block.get("id")?.clone());
            out.insert("name".into(), block.get("name")?.clone());
            out.insert(
                "input".into(),
                block.get("input").cloned().unwrap_or(serde_json::json!({})),
            );
        }
        "tool_result" => {
            out.insert("tool_use_id".into(), block.get("tool_use_id")?.clone());
            let c = block
                .get("content")
                .cloned()
                .unwrap_or(serde_json::json!(""));
            out.insert("content".into(), c);
        }
        // Thinking blocks carry signatures that will not validate in a fresh
        // request, and images balloon the cost without bearing on the probe.
        _ => return None,
    }
    Some(serde_json::Value::Object(out))
}

/// Messages, synthesised tool definitions, and a map from tool_use id to the
/// call that produced it: (tool name, the path or command it targeted).
type Rebuilt = (
    Vec<serde_json::Value>,
    Vec<serde_json::Value>,
    HashMap<String, (String, String)>,
);

/// Rebuild a valid Messages request from the transcript, cutting before the
/// assistant turn we want the model to produce.
fn rebuild(path: &Path, upto_msg: u32) -> Option<Rebuilt> {
    let f = File::open(path).ok()?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut msgs: Vec<serde_json::Value> = Vec::new();
    let mut tool_names: Vec<String> = Vec::new();
    // tool_use_id -> (tool name, the file path or command it targeted)
    let mut origin: HashMap<String, (String, String)> = HashMap::new();
    let mut idx: u32 = 0;

    for line in BufReader::new(f).split(b'\n') {
        let Ok(raw) = line else { continue };
        let Ok(d) = serde_json::from_slice::<serde_json::Value>(&raw) else {
            continue;
        };
        let Some(m) = d.get("message") else { continue };
        let Some(role) = m.get("role").and_then(|v| v.as_str()) else {
            continue;
        };
        let uuid = d.get("uuid").and_then(|v| v.as_str()).unwrap_or("");
        if uuid.is_empty() || !seen.insert(uuid.to_string()) {
            continue;
        }
        if idx >= upto_msg {
            break;
        }
        idx += 1;

        let owned;
        let items: &Vec<serde_json::Value> = match m.get("content") {
            Some(serde_json::Value::Array(a)) => a,
            Some(serde_json::Value::String(s)) => {
                owned = vec![serde_json::json!({"type":"text","text":s})];
                &owned
            }
            _ => continue,
        };

        let mut kept: Vec<serde_json::Value> = Vec::new();
        for b in items {
            if b.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                if let (Some(id), Some(name)) = (
                    b.get("id").and_then(|v| v.as_str()),
                    b.get("name").and_then(|v| v.as_str()),
                ) {
                    let target = b
                        .get("input")
                        .and_then(|i| {
                            i.get("file_path")
                                .or_else(|| i.get("command"))
                                .or_else(|| i.get("pattern"))
                        })
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    origin.insert(id.to_string(), (name.to_string(), target));
                    if !tool_names.iter().any(|n| n == name) {
                        tool_names.push(name.to_string());
                    }
                }
            }
            if let Some(w) = whitelist(b) {
                kept.push(w);
            }
        }
        if !kept.is_empty() {
            msgs.push(serde_json::json!({"role": role, "content": kept}));
        }
    }

    // Every tool_use needs its result in the same request, and the array must
    // open on a user turn. Trim from both ends until both hold.
    let mut have: HashSet<String> = HashSet::new();
    for m in &msgs {
        for b in m["content"].as_array()? {
            if b["type"] == "tool_result" {
                if let Some(id) = b["tool_use_id"].as_str() {
                    have.insert(id.to_string());
                }
            }
        }
    }
    while let Some(last) = msgs.last() {
        let dangling = last["content"]
            .as_array()
            .map(|a| {
                a.iter().any(|b| {
                    b["type"] == "tool_use"
                        && b["id"].as_str().map(|i| !have.contains(i)).unwrap_or(false)
                })
            })
            .unwrap_or(false);
        if dangling {
            msgs.pop();
        } else {
            break;
        }
    }
    while msgs.first().map(|m| m["role"] != "user").unwrap_or(false) {
        msgs.remove(0);
    }
    if msgs.len() < 2 {
        return None;
    }

    let tools = tool_names
        .into_iter()
        .map(|n| {
            serde_json::json!({
                "name": n,
                "description": "Tool observed in this session. Schema is permissive because \
                                the original definitions are not recorded in a transcript.",
                "input_schema": {"type": "object", "additionalProperties": true}
            })
        })
        .collect();

    Some((msgs, tools, origin))
}

fn apply_mask(msgs: &[serde_json::Value], keep: &Policy) -> Vec<serde_json::Value> {
    // Locate tool_result blocks in order, decide which survive, then rewrite.
    let mut positions: Vec<(usize, usize, u32)> = Vec::new();
    for (mi, m) in msgs.iter().enumerate() {
        if let Some(arr) = m["content"].as_array() {
            for (bi, b) in arr.iter().enumerate() {
                if b["type"] == "tool_result" {
                    let n = b["content"].to_string().len() as u32 / 4;
                    positions.push((mi, bi, n));
                }
            }
        }
    }
    let live: Vec<usize> = (0..positions.len()).collect();
    let sizes: HashMap<usize, u32> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.2))
        .collect();
    let from = keep.survives_from_indexed(&live, &sizes);

    let mut out = msgs.to_vec();
    for (i, (mi, bi, _)) in positions.iter().enumerate() {
        if i < from {
            out[*mi]["content"][*bi]["content"] = serde_json::json!(MASK);
        }
    }
    out
}

/// The corpus a run draws from, kept together so the selection knobs stay
/// legible in the signature.
pub struct Corpus<'a> {
    pub sessions: &'a [&'a Session],
    pub probes: &'a [Vec<Probe>],
    pub paths: &'a [String],
    pub interned: &'a [String],
}

pub fn build_cases(
    c: &Corpus,
    pol: Policy,
    sample: usize,
    model_filter: &str,
    dropped: &mut Dropped,
) -> Vec<Case> {
    let (sessions, probes, paths, interned) = (c.sessions, c.probes, c.paths, c.interned);
    let mut flat: Vec<(usize, &Probe)> = Vec::new();
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
            flat.push((si, p));
        }
    }
    // Deterministic spread across sessions rather than the first N of one.
    flat.sort_by_key(|(si, p)| (p.use_at, *si));
    let step = (flat.len() / sample.max(1)).max(1);

    let mut out: Vec<(usize, usize, Case)> = Vec::new();
    for (si, p) in flat.into_iter().step_by(step) {
        if out.len() >= sample {
            break;
        }
        let path = Path::new(&paths[si]);
        let msg_of_use = sessions[si]
            .blocks
            .get(p.use_at)
            .map(|b| b.msg)
            .unwrap_or(0);
        dropped.considered += 1;
        let Some((intact, tools, origin)) = rebuild(path, msg_of_use) else {
            dropped.unrebuildable += 1;
            continue;
        };
        let fact = interned[p.tok as usize].clone();
        // Find the tool_result that actually carries the fact, then the call that
        // produced it, so a re-fetch attempt is recognisable. Picking any tool
        // from the session would mislabel the outcome.
        let mut otool = String::new();
        let mut otarget = String::new();
        'find: for m in &intact {
            for b in m["content"].as_array().into_iter().flatten() {
                if b["type"] == "tool_result" && b["content"].to_string().contains(&fact) {
                    if let Some((n, t)) = b["tool_use_id"].as_str().and_then(|id| origin.get(id)) {
                        otool = n.clone();
                        otarget = t.clone();
                    }
                    break 'find;
                }
            }
        }
        let masked = apply_mask(&intact, &pol);
        // A probe is only usable if the policy actually removed the fact.
        let still_there = masked
            .iter()
            .any(|m| m["content"].to_string().contains(&fact));
        if still_there {
            dropped.policy_kept_the_fact += 1;
            continue;
        }
        let model = sessions[si]
            .usage
            .first()
            .map(|u| u.model.clone())
            .unwrap_or_else(|| "claude-sonnet-5".into());
        out.push((
            si,
            intact.len(),
            Case {
                fact,
                model,
                origin_tool: otool,
                origin_target: otarget,
                messages_intact: intact,
                messages_masked: masked,
                tools,
            },
        ));
    }
    // Group by session and order by growing prefix, so each control arm extends
    // the one before it and reads most of its context from cache.
    out.sort_by_key(|(si, len, _)| (*si, *len));
    out.into_iter().map(|(_, _, c)| c).collect()
}

pub fn estimate_tokens(c: &Case) -> (u64, u64) {
    let a = serde_json::to_string(&c.messages_intact)
        .unwrap_or_default()
        .len() as u64
        / 4;
    let b = serde_json::to_string(&c.messages_masked)
        .unwrap_or_default()
        .len() as u64
        / 4;
    let t = serde_json::to_string(&c.tools).unwrap_or_default().len() as u64 / 4;
    (a + t, b + t)
}

pub fn estimate_cost(cases: &[Case]) -> f64 {
    let mut usd = 0.0;
    for c in cases {
        let (a, b) = estimate_tokens(c);
        let (inp, outp) = price(&c.model);
        usd += (a + b) as f64 / 1e6 * inp;
        usd += 2.0 * 600.0 / 1e6 * outp; // a short next action, both arms
    }
    usd
}

/// POST one request through curl, keeping the key out of the process list by
/// passing headers in a 0600 config file rather than on the command line.
fn call(
    api_key: &str,
    model: &str,
    msgs: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> Option<serde_json::Value> {
    // The next action is short. 1024 bought nothing and output is billed at 5x.
    let mut msgs = msgs.to_vec();
    // Mark the prefix cacheable. Cases are ordered so that consecutive control
    // arms from one session extend the previous prefix, which then reads at 0.1x
    // instead of being re-sent at full price.
    if let Some(last) = msgs.last_mut() {
        if let Some(arr) = last["content"].as_array_mut() {
            if let Some(b) = arr.last_mut() {
                b["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
        }
    }
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 512,
        "temperature": 0,
        "system": "You are continuing an agent session. Produce only the next action you \
                   would take, in the same style as the transcript so far.",
        "tools": tools,
        "messages": msgs,
    });
    let dir = std::env::temp_dir();
    let bp = dir.join(format!("ctxmeter-body-{}.json", std::process::id()));
    let cp = dir.join(format!("ctxmeter-cfg-{}.txt", std::process::id()));
    std::fs::write(&bp, serde_json::to_vec(&body).ok()?).ok()?;
    {
        let mut f = File::create(&cp).ok()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = f.metadata().ok()?.permissions();
            perm.set_mode(0o600);
            let _ = f.set_permissions(perm);
        }
        writeln!(f, "header = \"x-api-key: {api_key}\"").ok()?;
        writeln!(f, "header = \"anthropic-version: 2023-06-01\"").ok()?;
        writeln!(f, "header = \"content-type: application/json\"").ok()?;
    }
    let out = Command::new("curl")
        .args([
            "-sS",
            "--config",
            cp.to_str()?,
            "-X",
            "POST",
            "https://api.anthropic.com/v1/messages",
            "--data-binary",
            &format!("@{}", bp.to_str()?),
        ])
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&cp);
    let _ = std::fs::remove_file(&bp);
    serde_json::from_slice(&out.stdout).ok()
}

fn grade(resp: &serde_json::Value, c: &Case) -> Outcome {
    let blocks = resp.get("content").and_then(|v| v.as_array());
    let Some(blocks) = blocks else {
        return Outcome::Silent;
    };
    let whole = serde_json::to_string(blocks).unwrap_or_default();
    if whole.contains(&c.fact) {
        return Outcome::Reproduced;
    }
    for b in blocks {
        if b["type"] == "tool_use" {
            let name = b["name"].as_str().unwrap_or("");
            let input = b["input"].to_string();
            let same_tool = !c.origin_tool.is_empty() && name == c.origin_tool;
            let same_target = !c.origin_target.is_empty() && input.contains(&c.origin_target);
            if same_tool || same_target {
                return Outcome::Sought;
            }
        }
    }
    Outcome::Silent
}

pub struct Verdict {
    pub informative: usize,
    pub discarded: usize,
    pub reproduced: usize,
    pub sought: usize,
    pub silent: usize,
}

pub fn run(cases: &[Case], api_key: &str) -> Verdict {
    let mut v = Verdict {
        informative: 0,
        discarded: 0,
        reproduced: 0,
        sought: 0,
        silent: 0,
    };
    for c in cases {
        // Control first. If the intact context does not reproduce the fact, the
        // probe cannot tell us anything about the policy.
        let Some(ctrl) = call(api_key, &c.model, &c.messages_intact, &c.tools) else {
            v.discarded += 1;
            continue;
        };
        if grade(&ctrl, c) != Outcome::Reproduced {
            v.discarded += 1;
            continue;
        }
        let Some(treat) = call(api_key, &c.model, &c.messages_masked, &c.tools) else {
            v.discarded += 1;
            continue;
        };
        v.informative += 1;
        match grade(&treat, c) {
            Outcome::Reproduced => v.reproduced += 1,
            Outcome::Sought => v.sought += 1,
            Outcome::Silent => v.silent += 1,
        }
    }
    v
}
