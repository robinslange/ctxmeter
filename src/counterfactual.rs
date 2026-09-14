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
    if model.contains("fable") || model.contains("mythos") {
        (10.0, 50.0)
    } else if model.contains("opus") {
        (5.0, 25.0)
    } else if model.contains("haiku") {
        (1.0, 5.0)
    } else if model.contains("sonnet-4-6") {
        (3.0, 15.0)
    } else {
        (2.0, 10.0)
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
    pub turns_considered: usize,
    pub considered: usize,
    pub unrebuildable: usize,
    pub policy_kept_the_fact: usize,
    pub other_model: usize,
    /// The call that produced the fact could not be identified, so a re-fetch
    /// would be indistinguishable from silence. Dropped rather than graded, since
    /// the two outcomes it cannot separate are the two that carry the finding.
    pub no_refetch_target: usize,
}

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
            let c = match block.get("content") {
                Some(v) if !v.is_null() => v.clone(),
                _ => serde_json::json!(""),
            };
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
                    let input = b.get("input");
                    let target = input
                        .and_then(|i| i.get("file_path").or_else(|| i.get("pattern")))
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or_else(|| {
                            input
                                .and_then(|i| i.get("command"))
                                .and_then(|v| v.as_str())
                                .and_then(file_in)
                        })
                        .unwrap_or_default();
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
    // A tool_use whose result was never recorded invalidates the whole request, not
    // just the block, and the API rejects all of it. Drop the block and keep the
    // rest of the turn, which is still what the agent did. A message left with no
    // content goes with it.
    for m in &mut msgs {
        if let Some(arr) = m["content"].as_array_mut() {
            arr.retain(|b| {
                b["type"] != "tool_use" || b["id"].as_str().is_some_and(|i| have.contains(i))
            });
        }
    }
    msgs.retain(|m| m["content"].as_array().is_some_and(|a| !a.is_empty()));
    // The last message must be a user turn. A trailing assistant message is a
    // prefill, which these models reject outright, and it is also where a
    // tool_use whose result we cut lands. One condition covers both.
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
        if dangling || last["role"] != "user" {
            msgs.pop();
        } else {
            break;
        }
    }
    // And open on one. A tool_result can only answer a tool_use that came before
    // it, so any tool_result in the first message is answering one we just cut.
    while let Some(first) = msgs.first() {
        let orphan = first["content"]
            .as_array()
            .map(|a| a.iter().any(|b| b["type"] == "tool_result"))
            .unwrap_or(false);
        if first["role"] != "user" || orphan {
            msgs.remove(0);
        } else {
            break;
        }
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

/// The file a command reaches for, if it names one.
///
/// Matching a re-fetch against the whole command makes seeking unreachable for
/// every fact that came out of a shell, since a second attempt almost never
/// reproduces the line byte for byte. Matching any path in it goes too far the
/// other way: the longest path is often the directory the work happens in, and in
/// a session about one app every command names that, which is the tool-name
/// mistake again with a smaller radius.
///
/// A file is an artifact and a directory is a scope. Naming the same file later is
/// evidence of going back to it; being in the same directory is not. So only a
/// path whose last segment looks like a filename counts, and a command that names
/// no file yields no target, which drops the case rather than guessing at it.
fn file_in(cmd: &str) -> Option<String> {
    // The sinks every command redirects into: present in a re-fetch and in
    // anything else, so their appearance is not evidence of one.
    const NOISE: [&str; 3] = ["/dev/null", "/dev/stdout", "/dev/stderr"];
    cmd.split(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | ';' | '|' | '>' | '<' | '(' | ')' | ',')
    })
    .filter(|t| {
        t.contains('/')
            && t.len() > 8
            && !NOISE.contains(t)
            && t.rsplit('/').next().is_some_and(|f| {
                // A filename, not a directory: a dot with something either side.
                f.rfind('.').is_some_and(|i| i > 0 && i + 1 < f.len())
            })
    })
    .max_by_key(|t| t.len())
    .map(str::to_string)
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
            let cut = sessions[si]
                .blocks
                .get(p.use_at)
                .map(|b| b.msg)
                .unwrap_or(0);
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
            if masked
                .iter()
                .any(|m| m["content"].to_string().contains(&text))
            {
                dropped.policy_kept_the_fact += 1;
                continue;
            }
            // Find the tool_result carrying the fact, then the call that produced
            // it. Any other call in the session would mislabel a re-fetch.
            let mut found = String::new();
            'find: for m in &intact {
                for b in m["content"].as_array().into_iter().flatten() {
                    if b["type"] == "tool_result" && b["content"].to_string().contains(&text) {
                        if let Some((_, t)) =
                            b["tool_use_id"].as_str().and_then(|id| origin.get(id))
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
            facts.push(Fact {
                text,
                origin: found,
            });
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
}

/// The structural rules a Messages request has to satisfy. Checked without
/// spending anything, so a malformed rebuild is caught by the dry run rather
/// than by a 400 that later reads as a control arm failing to reproduce a fact.
pub fn invalid(msgs: &[serde_json::Value]) -> Option<String> {
    let first = msgs.first()?;
    if first["role"] != "user" {
        return Some(format!("opens on a {} turn", first["role"]));
    }
    if msgs.last()?["role"] != "user" {
        return Some("ends on an assistant turn, which is a prefill".into());
    }
    let mut uses: HashSet<&str> = HashSet::new();
    let mut results: HashSet<&str> = HashSet::new();
    for m in msgs {
        for b in m["content"].as_array().into_iter().flatten() {
            match b["type"].as_str() {
                Some("tool_use") => uses.insert(b["id"].as_str().unwrap_or("")),
                Some("tool_result") => results.insert(b["tool_use_id"].as_str().unwrap_or("")),
                _ => continue,
            };
        }
    }
    if let Some(id) = uses.difference(&results).next() {
        return Some(format!("tool_use {id} has no result"));
    }
    if let Some(id) = results.difference(&uses).next() {
        return Some(format!(
            "tool_result answers {id}, which is not in the request"
        ));
    }
    None
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
        usd += 2.0 * MAX_TOKENS as f64 / 1e6 * outp; // both arms, at the ceiling
    }
    usd
}

/// Agent turns in a real corpus write documents, not one-liners: at 2048, two of
/// the first five measured cases were cut off mid-action. Output is the cheap half
/// of this measurement and the rebuilt context is the expensive half, so a ceiling
/// that discards a case wastes far more than it saves.
const MAX_TOKENS: u64 = 8192;

const SYSTEM: &str = "You are continuing an agent session. Produce only the next action you \
                      would take, in the same style as the transcript so far.";

/// Create at 0600 rather than create-then-chmod: the key must never exist in a
/// file another user could open, not even for an instant.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(bytes)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// POST through curl, keeping the key out of the process list by passing headers
/// in a 0600 config file rather than on the command line.
///
/// Anything that is not a 200 carrying JSON is an error and says so. A rejected
/// request must never reach the caller looking like an answer, because the
/// caller's next move is to read a missing fact as a failure to reproduce one.
fn post(api_key: &str, url: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let bp = dir.join(format!("ctxmeter-body-{pid}.json"));
    let cp = dir.join(format!("ctxmeter-cfg-{pid}.txt"));
    let rp = dir.join(format!("ctxmeter-resp-{pid}.json"));
    let clean = || {
        let _ = std::fs::remove_file(&bp);
        let _ = std::fs::remove_file(&cp);
        let _ = std::fs::remove_file(&rp);
    };
    // The request carries real tool output, so it is 0600 too, and deleted.
    let r = (|| -> Result<serde_json::Value, String> {
        write_private(&bp, &serde_json::to_vec(body).map_err(|e| e.to_string())?)?;
        write_private(
            &cp,
            format!(
                "header = \"x-api-key: {api_key}\"\n\
                 header = \"anthropic-version: 2023-06-01\"\n\
                 header = \"content-type: application/json\"\n"
            )
            .as_bytes(),
        )?;
        let (Some(cps), Some(bps), Some(rps)) = (cp.to_str(), bp.to_str(), rp.to_str()) else {
            return Err("temp path is not utf-8".into());
        };
        let out = Command::new("curl")
            .args([
                "-sS",
                "--config",
                cps,
                "-X",
                "POST",
                url,
                "--data-binary",
                &format!("@{bps}"),
                "-o",
                rps,
                "-w",
                "%{http_code}",
            ])
            .output()
            .map_err(|e| format!("curl did not run: {e}"))?;
        let code = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let curl_err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let raw = std::fs::read(&rp).unwrap_or_default();
        let parsed: Option<serde_json::Value> = serde_json::from_slice(&raw).ok();
        if code != "200" {
            let detail = parsed
                .as_ref()
                .and_then(|v| v.get("error"))
                .map(|e| e.to_string())
                .unwrap_or_else(|| {
                    String::from_utf8_lossy(&raw)
                        .trim()
                        .chars()
                        .take(400)
                        .collect()
                });
            return Err(format!("HTTP {code} {curl_err} {detail}")
                .trim()
                .to_string());
        }
        let v = parsed.ok_or_else(|| format!("HTTP 200, body is not JSON: {curl_err}"))?;
        if v.get("type").and_then(|t| t.as_str()) == Some("error") {
            return Err(format!("api error: {}", v.get("error").unwrap_or(&v)));
        }
        Ok(v)
    })();
    clean();
    r
}

fn request_body(
    model: &str,
    msgs: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> serde_json::Value {
    // Mark the prefix cacheable. Cases are ordered so that consecutive control
    // arms from one session extend the previous prefix, which then reads at 0.1x
    // instead of being re-sent at full price.
    let mut msgs = msgs.to_vec();
    if let Some(last) = msgs.last_mut() {
        if let Some(arr) = last["content"].as_array_mut() {
            if let Some(b) = arr.last_mut() {
                b["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
        }
    }
    serde_json::json!({
        "model": model,
        "system": SYSTEM,
        "tools": tools,
        "messages": msgs,
    })
}

/// Free, and the same curl path and request shape the paid call uses. It answers
/// the only question worth answering before spending: does the API accept what
/// we rebuilt? It also returns a measured token count instead of an estimate.
fn count_tokens(
    api_key: &str,
    model: &str,
    msgs: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> Result<u64, String> {
    let v = post(
        api_key,
        "https://api.anthropic.com/v1/messages/count_tokens",
        &request_body(model, msgs, tools),
    )?;
    v.get("input_tokens")
        .and_then(|t| t.as_u64())
        .ok_or_else(|| format!("no input_tokens in {v}"))
}

fn call(
    api_key: &str,
    model: &str,
    msgs: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> Result<serde_json::Value, String> {
    let mut body = request_body(model, msgs, tools);
    // No temperature: these models reject sampling parameters outright. Thinking
    // is left at the model default, because that is how the trace was produced.
    body["max_tokens"] = serde_json::json!(MAX_TOKENS);
    post(api_key, "https://api.anthropic.com/v1/messages", &body)
}

/// What the model actually did, reasoning excluded. Ground truth is the action:
/// a fact recalled inside a thinking block is not a reuse of it.
fn action_blocks(resp: &serde_json::Value) -> Vec<&serde_json::Value> {
    resp.get("content")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter(|b| b["type"] != "thinking" && b["type"] != "redacted_thinking")
                .collect()
        })
        .unwrap_or_default()
}

/// Truncation and refusal invalidate silence and nothing else. A fact that has
/// already appeared, or an action that has already named the origin, is evidence
/// whether or not the turn had room to finish. An absence is not, because the
/// ceiling could be the whole reason for it.
fn unusable(resp: &serde_json::Value, g: Outcome) -> Option<String> {
    if g != Outcome::Silent {
        return None;
    }
    match resp.get("stop_reason").and_then(|v| v.as_str()) {
        Some("max_tokens") => Some("silent, but it hit max_tokens first".into()),
        Some("refusal") => Some("silent, but the model declined the turn".into()),
        _ => None,
    }
}

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

fn show(label: &str, resp: &serde_json::Value) {
    println!("\n--- {label} ---");
    println!(
        "stop_reason {:?}   usage {}",
        resp.get("stop_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("?"),
        resp.get("usage").map(|u| u.to_string()).unwrap_or_default()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(resp.get("content").unwrap_or(resp)).unwrap_or_default()
    );
}

pub fn run(cases: &[Case], api_key: &str, show_raw: bool) -> Verdict {
    let mut v = Verdict {
        turns_attempted: 0,
        turns_informative: 0,
        sessions: 0,
        informative: 0,
        discarded: 0,
        unusable: 0,
        reproduced: 0,
        sought: 0,
        silent: 0,
        errors: Vec::new(),
        measured_tokens: 0,
    };
    // Every arm of every case first, for nothing. It rejects a malformed rebuild
    // before the money rather than after it, and it prices the run from the
    // tokenizer instead of from a guess about it.
    let mut sizes: Vec<(u64, u64)> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let ctrl = match count_tokens(api_key, &c.model, &c.messages_intact, &c.tools) {
            Ok(n) => n,
            Err(e) => {
                v.errors.push(format!("case {i} control rejected: {e}"));
                return v;
            }
        };
        let treat = match count_tokens(api_key, &c.model, &c.messages_masked, &c.tools) {
            Ok(n) => n,
            Err(e) => {
                v.errors.push(format!("case {i} treatment rejected: {e}"));
                return v;
            }
        };
        v.measured_tokens += ctrl + treat;
        sizes.push((ctrl, treat));
    }
    let ceiling: f64 = cases
        .iter()
        .zip(&sizes)
        .map(|(c, (a, b))| {
            let (inp, outp) = price(&c.model);
            // A cache write bills at 1.25x, and a treatment arm only runs when its
            // control reproduced, so this is a ceiling and not the bill.
            (a + b) as f64 / 1e6 * inp * 1.25 + 2.0 * MAX_TOKENS as f64 / 1e6 * outp
        })
        .sum();
    println!(
        "\nmeasured {} input tokens across both arms of {} cases: at most ${ceiling:.2},",
        v.measured_tokens,
        cases.len()
    );
    println!("and less for every case whose control arm fails and never buys a second.");

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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every cut point has to yield a sendable request, not just the tidy ones.
    /// A cut landing inside an assistant turn leaves a prefill, and these models
    /// reject that outright.
    #[test]
    fn a_rebuilt_request_is_valid_at_every_cut_point() {
        let p = Path::new("tests/fixture/session.jsonl");
        let mut built = 0;
        for upto in 1..=9 {
            if let Some((msgs, _, _)) = rebuild(p, upto) {
                assert_eq!(invalid(&msgs), None, "cut at {upto}");
                built += 1;
            }
        }
        assert!(built >= 3, "only {built} cut points rebuilt at all");
    }

    /// A call whose result was never recorded makes the whole request invalid, and
    /// the API rejects the request rather than the block. The rest of the turn is
    /// still what the agent did, so drop the block and keep the case.
    #[test]
    fn a_call_with_no_recorded_result_is_dropped_not_kept() {
        let p = Path::new("tests/fixture/orphan-call.jsonl");
        let (msgs, _, _) = rebuild(p, 9).expect("rebuildable");
        assert_eq!(invalid(&msgs), None);
        let whole = serde_json::to_string(&msgs).unwrap();
        assert!(!whole.contains("\"lost\""), "the unanswered call survived");
        assert!(
            whole.contains("\"kept\""),
            "the answered call was dropped too"
        );
        assert!(
            whole.contains("looking"),
            "text in the same message was lost"
        );
    }

    fn resp(blocks: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"stop_reason": "end_turn", "content": blocks})
    }

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
                Fact {
                    text: "a7f3c9e21b84".into(),
                    origin: "/home/dev/build.log".into(),
                },
                Fact {
                    text: "b81d0c4a9f27".into(),
                    origin: "/home/dev/build.log".into(),
                },
            ],
            messages_intact: vec![],
            messages_masked: vec![],
            tools: vec![],
        };
        assert_eq!(c.facts.len(), 2);
        assert_eq!(c.cut, 40);
    }

    fn fact() -> Fact {
        Fact {
            text: "a7f3c9e21b84".into(),
            origin: "/home/dev/build.log".into(),
        }
    }

    /// The control arm is the whole basis for discarding a probe. If a response
    /// without the fact could still grade as Reproduced, nothing would ever be
    /// discarded and every later number would be built on the wrong cases.
    #[test]
    fn a_control_that_does_not_reproduce_the_fact_is_not_informative() {
        let c = fact();
        let got = resp(serde_json::json!([{"type": "text", "text": "let me check the config"}]));
        assert_ne!(grade(&got, &c), Outcome::Reproduced);
        let got = resp(serde_json::json!([{"type": "text", "text": "build a7f3c9e21b84 failed"}]));
        assert_eq!(grade(&got, &c), Outcome::Reproduced);
    }

    /// Any tool that names the origin counts, because `cat` and `Read` fetch the
    /// same file. Reaching for the same tool on something else does not: it is
    /// the ordinary next action of a session, and counting it as a re-fetch moves
    /// the finding into the healthy bucket for free.
    #[test]
    fn going_back_for_the_fact_means_going_back_to_its_origin() {
        let c = fact();
        let same_file = resp(serde_json::json!([
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/home/dev/build.log"}}
        ]));
        assert_eq!(grade(&same_file, &c), Outcome::Sought);
        let other_tool_same_file = resp(serde_json::json!([
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat /home/dev/build.log"}}
        ]));
        assert_eq!(grade(&other_tool_same_file, &c), Outcome::Sought);
        let same_tool_other_file = resp(serde_json::json!([
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/etc/hosts"}}
        ]));
        assert_eq!(grade(&same_tool_other_file, &c), Outcome::Silent);
    }

    /// Reasoning is not an action. Counting a fact recalled inside a thinking
    /// block as a reuse would let the control arm pass on a turn that never
    /// actually used it.
    #[test]
    fn a_fact_recalled_while_thinking_is_not_a_reuse() {
        let c = fact();
        let got = resp(serde_json::json!([
            {"type": "thinking", "thinking": "the id was a7f3c9e21b84", "signature": "x"},
            {"type": "text", "text": "I will look it up again"}
        ]));
        assert_ne!(grade(&got, &c), Outcome::Reproduced);
    }

    /// The point of the change: one replayed response, several facts. A turn that
    /// reproduces one fact and not another is one observation with two outcomes,
    /// which is what it always was. Two requests would have made it two events and
    /// could have disagreed with itself.
    #[test]
    fn one_response_is_graded_against_each_fact_of_the_turn() {
        let here = Fact {
            text: "a7f3c9e21b84".into(),
            origin: "/home/dev/build.log".into(),
        };
        let gone = Fact {
            text: "b81d0c4a9f27".into(),
            origin: "/home/dev/other.log".into(),
        };
        let got = resp(serde_json::json!([
            {"type": "text", "text": "build a7f3c9e21b84 failed"}
        ]));
        assert_eq!(grade(&got, &here), Outcome::Reproduced);
        assert_eq!(grade(&got, &gone), Outcome::Silent);
    }

    /// A turn cut off by the output ceiling says nothing about the policy. Graded
    /// as silence it would inflate the one number the tool exists to report.
    /// The ceiling explains an absence, so silence under it is not evidence. It
    /// explains nothing about a fact that already came back, and discarding that
    /// case throws away the context it was already paid for.
    #[test]
    fn the_output_ceiling_only_invalidates_silence() {
        let mut cut = resp(serde_json::json!([{"type": "text", "text": "I"}]));
        cut["stop_reason"] = serde_json::json!("max_tokens");
        assert!(unusable(&cut, Outcome::Silent).is_some());
        assert!(unusable(&cut, Outcome::Reproduced).is_none());
        assert!(unusable(&cut, Outcome::Sought).is_none());
        let whole = resp(serde_json::json!([]));
        assert!(unusable(&whole, Outcome::Silent).is_none());
    }

    /// A command is not a target. Every Bash-derived case would be ungradeable
    /// for seeking if it were, because the bucket could never be reached.
    /// Shaped after a real measured case, with the paths replaced: a probe token
    /// or a path out of someone's trace does not belong in this repository. The
    /// directory is the longer path, and taking it would score every command in
    /// that project as a re-fetch. The test file is the artifact.
    #[test]
    fn a_directory_is_a_scope_and_not_a_target() {
        assert_eq!(
            file_in(
                "cd /home/dev/monorepo/apps/checkout && pnpm test \
                 worker/routes/plans.tenant.test.ts 2>&1 | tail -40"
            ),
            Some("worker/routes/plans.tenant.test.ts".into())
        );
        // Names no file, so it yields no target and the case is dropped.
        assert_eq!(file_in("cd /home/dev/monorepo/notes && ls"), None);
        assert_eq!(file_in("pnpm test"), None);
    }

    /// A redirect sink is in every second command, so matching one would score
    /// unrelated work as a re-fetch, exactly as matching the tool name did.
    #[test]
    fn a_redirect_sink_is_not_a_target() {
        assert_eq!(file_in("ls foo 2>/dev/null"), None);
        assert_eq!(
            file_in("grep -m1 status /home/dev/notes/index.md 2>/dev/null"),
            Some("/home/dev/notes/index.md".into())
        );
    }
}
