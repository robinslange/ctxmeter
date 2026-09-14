use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// base64 length overstates image tokens ~2.85x; calibrated against usage records.
const IMG_SCALE: f64 = 0.35;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    ToolResult,
    Text,
    Thinking,
    ToolUse,
    Media,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    User,
    Assistant,
    Other,
}

pub struct Block {
    /// Index of the message this block belongs to. Tier two cuts at message
    /// boundaries, because a request ends before the assistant turn it asks for.
    pub msg: u32,
    pub kind: Kind,
    pub role: Role,
    pub tokens: u32,
    /// Interned non-guessable identifiers found in this block, sorted and deduped.
    pub toks: Vec<u32>,
}

pub struct Usage {
    pub ts: String,
    pub model: String,
    pub read: u64,
    pub write: u64,
    pub fresh: u64,
    pub out: u64,
}

pub struct Session {
    /// Transcript file this came from. Tier two re-reads it to rebuild requests.
    pub path: String,
    /// Content messages seen, used to exclude sessions too short to yield probes.
    pub msgs: usize,
    pub blocks: Vec<Block>,
    /// One per billed assistant turn: (blocks preceding the request, real prefix tokens).
    /// The real prefix includes the system prompt and tool definitions, which never
    /// appear in a transcript, so cost must be grounded in it rather than estimated.
    pub turns: Vec<(usize, u64)>,
    pub usage: Vec<Usage>,
}

#[derive(Default)]
pub struct Interner {
    map: HashMap<String, u32>,
    pub names: Vec<String>,
}

impl Interner {
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.map.insert(s.to_string(), i);
        i
    }
}

fn ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b':' | b'-')
}

/// Non-guessable identifiers: >=10 chars, contains a digit, not a bare number.
pub fn candidates(text: &str, mut push: impl FnMut(&str)) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !ident_char(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && ident_char(b[i]) {
            i += 1;
        }
        let run = &b[start..i];
        if run.len() < 10 {
            continue;
        }
        let mut has_digit = false;
        let mut has_alpha = false;
        for &c in run {
            if c.is_ascii_digit() {
                has_digit = true;
            } else if c.is_ascii_alphabetic() {
                has_alpha = true;
            }
        }
        if has_digit && has_alpha {
            if let Ok(s) = std::str::from_utf8(run) {
                push(s);
            }
        }
    }
}

fn est_tokens(n: usize, media: bool) -> u32 {
    let t = n / 4;
    if media {
        (t as f64 * IMG_SCALE) as u32
    } else {
        t as u32
    }
}

fn text_of(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn read_session(
    path: &Path,
    it: &mut Interner,
    seen_req: &mut std::collections::HashSet<String>,
) -> Option<Session> {
    let f = File::open(path).ok()?;
    let mut seen_uuid: HashMap<String, ()> = HashMap::new();
    let mut blocks = Vec::new();
    let mut usage = Vec::new();
    let mut turns: Vec<(usize, u64)> = Vec::new();
    let mut fresh_usage: Option<u64> = None;
    let mut msgs_seen = 0usize;
    let mut msg_idx: u32 = 0;
    let mut tool_media: HashMap<String, bool> = HashMap::new();
    let mut buf = Vec::new();

    for line in BufReader::new(f).split(b'\n') {
        let Ok(raw) = line else { continue };
        let Ok(d) = serde_json::from_slice::<serde_json::Value>(&raw) else {
            continue;
        };
        let Some(m) = d.get("message") else { continue };

        if let Some(u) = m.get("usage") {
            let key = format!(
                "{}|{}",
                d.get("requestId").and_then(|v| v.as_str()).unwrap_or(""),
                m.get("id").and_then(|v| v.as_str()).unwrap_or("")
            );
            if key != "|" && seen_req.insert(key) {
                let g0 = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
                fresh_usage = Some(
                    g0("cache_read_input_tokens")
                        + g0("cache_creation_input_tokens")
                        + g0("input_tokens"),
                );
                let g = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
                usage.push(Usage {
                    ts: d.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    model: m.get("model").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                    read: g("cache_read_input_tokens"),
                    write: g("cache_creation_input_tokens"),
                    fresh: g("input_tokens"),
                    out: g("output_tokens"),
                });
            }
        }

        let Some(role_s) = m.get("role").and_then(|v| v.as_str()) else {
            continue;
        };
        let uuid = d.get("uuid").and_then(|v| v.as_str()).unwrap_or("");
        if uuid.is_empty() || seen_uuid.contains_key(uuid) {
            continue;
        }
        seen_uuid.insert(uuid.to_string(), ());
        msgs_seen += 1;
        let role = match role_s {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => Role::Other,
        };

        let content = m.get("content");
        let owned;
        let items: &Vec<serde_json::Value> = match content {
            Some(serde_json::Value::Array(a)) => a,
            Some(serde_json::Value::String(s)) => {
                owned = vec![serde_json::json!({"type": "text", "text": s})];
                &owned
            }
            _ => {
                msg_idx += 1;
                continue;
            }
        };

        if role == Role::Assistant {
            if let Some(real) = fresh_usage.take() {
                if real > 0 {
                    turns.push((blocks.len(), real));
                }
            }
        }
        fresh_usage = None;

        for b in items {
            let kind_s = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let (kind, body, media) = match kind_s {
                "tool_result" => {
                    let raw = b.get("content").map(text_of).unwrap_or_default();
                    let media = raw.contains("\"base64\"") || raw.contains("\"image\"");
                    (Kind::ToolResult, raw, media)
                }
                "text" => (Kind::Text, b.get("text").map(text_of).unwrap_or_default(), false),
                "thinking" => (
                    Kind::Thinking,
                    b.get("thinking").map(text_of).unwrap_or_default(),
                    false,
                ),
                "tool_use" => {
                    if let (Some(id), Some(name)) = (
                        b.get("id").and_then(|v| v.as_str()),
                        b.get("name").and_then(|v| v.as_str()),
                    ) {
                        let fp = b
                            .get("input")
                            .and_then(|i| i.get("file_path"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        let is_media = name == "Read"
                            && [".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"]
                                .iter()
                                .any(|e| fp.ends_with(e));
                        tool_media.insert(id.to_string(), is_media);
                    }
                    (
                        Kind::ToolUse,
                        b.get("input").map(|v| v.to_string()).unwrap_or_default(),
                        false,
                    )
                }
                "image" | "document" => (Kind::Media, b.to_string(), true),
                _ => (Kind::Other, b.to_string(), false),
            };

            let media_read = kind == Kind::ToolResult
                && b.get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .and_then(|id| tool_media.get(id))
                    .copied()
                    .unwrap_or(false);

            let want_toks = matches!(kind, Kind::ToolResult)
                || (role == Role::Assistant && matches!(kind, Kind::Text | Kind::ToolUse))
                || (role == Role::User && kind == Kind::Text);
            buf.clear();
            if want_toks {
                candidates(&body, |s| buf.push(s.to_string()));
            }
            let mut toks: Vec<u32> = buf.iter().map(|s| it.intern(s)).collect();
            toks.sort_unstable();
            toks.dedup();

            blocks.push(Block {
                msg: msg_idx,
                kind,
                role,
                tokens: est_tokens(body.len(), media || media_read),
                toks,
            });
        }
        msg_idx += 1;
    }

    if msgs_seen < 6 && usage.is_empty() {
        return None;
    }
    usage.sort_by(|a, b| a.ts.cmp(&b.ts));
    Some(Session {
        path: path.display().to_string(),
        msgs: msgs_seen,
        blocks,
        turns,
        usage,
    })
}

pub fn load(root: &Path, it: &mut Interner) -> Vec<Session> {
    let mut out = Vec::new();
    let mut seen_req = std::collections::HashSet::new();
    for e in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if e.file_type().is_file() && e.path().extension().map(|x| x == "jsonl").unwrap_or(false) {
            if let Some(s) = read_session(e.path(), it, &mut seen_req) {
                out.push(s);
            }
        }
    }
    out
}
