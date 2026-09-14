#!/usr/bin/env python3
"""Measure what Claude Code actually bills you, from your own transcripts."""
import argparse, collections, glob, json, os, statistics, sys

TRANSCRIPTS = os.path.expanduser("~/.claude/projects")
IMG_SCALE = 0.35   # base64 length overstates image tokens ~2.85x; calibrated against usage records
W, R = 1.25, 0.10  # cache write (5m TTL) and cache read, as multiples of base input

def usage_rows(root):
    """One row per billed request, deduplicated by request+message id."""
    seen = set()
    for f in glob.glob(root + "/**/*.jsonl", recursive=True):
        rows = []
        try: fh = open(f, errors="ignore")
        except OSError: continue
        for line in fh:
            try: d = json.loads(line)
            except ValueError: continue
            m = d.get("message") or {}
            u = m.get("usage")
            if not u: continue
            k = (d.get("requestId") or "") + "|" + (m.get("id") or "")
            if not k.strip("|") or k in seen: continue
            seen.add(k)
            rows.append({
                "session": os.path.basename(f)[:-6],
                "ts": d.get("timestamp", ""),
                "model": m.get("model") or "?",
                "read": u.get("cache_read_input_tokens", 0) or 0,
                "write": u.get("cache_creation_input_tokens", 0) or 0,
                "fresh": u.get("input_tokens", 0) or 0,
                "out": u.get("output_tokens", 0) or 0,
            })
        rows.sort(key=lambda r: r["ts"])
        if rows: yield f, rows

def blocks(root):
    """One entry per session: ordered content blocks with API-shaped token estimates."""
    for f in glob.glob(root + "/**/*.jsonl", recursive=True):
        seen, msgs = set(), []
        try: fh = open(f, errors="ignore")
        except OSError: continue
        for line in fh:
            try: d = json.loads(line)
            except ValueError: continue
            m = d.get("message") or {}
            if not m.get("role"): continue
            u = d.get("uuid") or ""
            if u in seen: continue
            seen.add(u); msgs.append(m)
        if len(msgs) >= 6: yield f, msgs

def block_tokens(b):
    t = b.get("type")
    if t == "thinking": return len(b.get("thinking") or "") // 4   # signature is metadata, not billed prose
    if t == "text": return len(b.get("text") or "") // 4
    if t == "tool_use": return len(json.dumps(b.get("input") or {})) // 4
    if t in ("image", "document"): return int(len(json.dumps(b)) // 4 * IMG_SCALE)
    if t == "tool_result":
        c = b.get("content")
        s = c if isinstance(c, str) else json.dumps(c)
        return int(len(s) // 4 * IMG_SCALE) if ('"base64"' in s or '"image"' in s) else len(s) // 4
    return len(json.dumps(b)) // 4

def pct(x): return f"{x:.2%}"

def cmd_summary(root):
    agg = collections.Counter(); models = collections.Counter(); sessions = set()
    for _, rows in usage_rows(root):
        for r in rows:
            agg["turns"] += 1
            for k in ("read", "write", "fresh", "out"): agg[k] += r[k]
            models[r["model"]] += 1; sessions.add(r["session"])
    tot = agg["read"] + agg["write"] + agg["fresh"]
    cost = agg["fresh"] + W * agg["write"] + R * agg["read"]
    print(f"turns {agg['turns']:,}   sessions {len(sessions):,}")
    print(f"\n{'':16}{'tokens':>16}{'share':>9}{'billed equiv':>16}")
    for k, mult in (("fresh", 1.0), ("write", W), ("read", R)):
        print(f"{k:16}{agg[k]:>16,}{pct(agg[k]/tot):>9}{mult*agg[k]:>16,.0f}")
    print(f"{'output':16}{agg['out']:>16,}")
    print(f"\ncache hit rate      {pct(agg['read']/tot)}")
    print(f"billed equivalents  {cost:,.0f}  (no-cache counterfactual {tot:,.0f})")
    print(f"caching already saves {pct(1 - cost/tot)}")
    print(f"\nmodels: {', '.join(f'{m} {n:,}' for m, n in models.most_common(4))}")

def cmd_floor(root):
    """System prompt + tool definitions, approximated by each session's first billed prefix."""
    by = collections.defaultdict(list); allf = []
    for _, rows in usage_rows(root):
        r = rows[0]
        p = r["read"] + r["write"] + r["fresh"]
        if p <= 0: continue
        allf.append(p)
        if len(r["ts"]) >= 7: by[r["ts"][:7]].append(p)
    q = statistics.quantiles(allf, n=100)
    print(f"first-turn billed prefix, n={len(allf):,}")
    for p in (25, 50, 75, 90): print(f"  p{p}: {q[p-1]:>10,.0f}")
    print(f"\n{'month':10}{'sessions':>10}{'median':>11}{'p90':>11}")
    for mo in sorted(by):
        v = sorted(by[mo])
        if len(v) < 15: continue
        print(f"{mo:10}{len(v):>10,}{v[len(v)//2]:>11,}{v[int(len(v)*0.9)]:>11,}")
    print("\nthis is re-read every turn and cannot be compressed by a proxy.")
    print("attribute it by changing config and re-running, not by reading logs:")
    print("tool definitions never appear in a transcript.")

def cmd_composition(root):
    """Carry cost by content type. Carry weights a block by the turns that re-read it."""
    carry = collections.Counter(); tot = 0; ratios = []
    for _, msgs in blocks(root):
        tn = {}; T = len(msgs); run = 0
        for i, m in enumerate(msgs):
            c = m.get("content")
            if isinstance(c, str): c = [{"type": "text", "text": c}]
            if not isinstance(c, list): continue
            for b in c:
                if not isinstance(b, dict): continue
                if b.get("type") == "tool_use":
                    tn[b.get("id")] = (b.get("name"), (b.get("input") or {}).get("file_path") or "")
                n = block_tokens(b); run += n
                t = b.get("type")
                if t == "tool_result":
                    nm, fp = tn.get(b.get("tool_use_id"), ("", ""))
                    ext = (os.path.splitext(fp)[1] or "").lower()
                    if nm == "Read" and ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"):
                        k = "image/PDF via Read"
                    elif nm == "Read": k = "Read: text & code"
                    elif nm == "Bash": k = "Bash output"
                    elif "search" in (nm or "").lower(): k = "web search"
                    else: k = "other tool results"
                elif t in ("image", "document"): k = "pasted image/doc"
                elif t == "thinking": k = "thinking"
                elif t == "text": k = "conversation text"
                elif t == "tool_use": k = "tool call inputs"
                else: k = "other"
                carry[k] += n * max(0, T - i - 1); tot += n * max(0, T - i - 1)
            u = m.get("usage")
            if u:
                real = (u.get("cache_read_input_tokens", 0) or 0) + (u.get("cache_creation_input_tokens", 0) or 0) + (u.get("input_tokens", 0) or 0)
                if real > 20000 and run > 5000: ratios.append(run / real)
    print(f"calibration: median estimate/real prefix = {statistics.median(ratios):.2f} (n={len(ratios):,})")
    print("below 1.0 because system prompt and tool definitions are in the real number, not this one.\n")
    print(f"{'category':26}{'carry share':>13}")
    for k, v in carry.most_common(12): print(f"{k:26}{pct(v/tot):>13}")

def cmd_invalidation(root):
    """Is invalidation partial or all-or-nothing? Determines whether mid-history edits can pay."""
    writes = []; pairs = []
    for _, rows in usage_rows(root):
        for i, r in enumerate(rows):
            writes.append(r["write"])
            if i: pairs.append((rows[i-1]["read"] + rows[i-1]["write"], r["read"]))
    q = statistics.quantiles(writes, n=100)
    print("incremental cache write per turn")
    for p in (50, 90, 99): print(f"  p{p}: {q[p-1]:>10,.0f}")
    clean = sum(1 for prev, r in pairs if r >= prev * 0.98)
    lost = [prev - r for prev, r in pairs if r < prev * 0.98]
    print(f"\nclean incremental hits: {clean:,} of {len(pairs):,} ({pct(clean/len(pairs))})")
    if lost:
        lq = statistics.quantiles(lost, n=100)
        print(f"when it misses, tokens lost from the cached prefix: p50 {lq[49]:,.0f}  p90 {lq[89]:,.0f}")
    print("\na median loss near a median prefix means invalidation is all-or-nothing,")
    print("which is what breakpoint-anchored matching with a 20-block lookback predicts.")

CMDS = {"summary": cmd_summary, "floor": cmd_floor,
        "composition": cmd_composition, "invalidation": cmd_invalidation}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=sorted(CMDS))
    ap.add_argument("--root", default=TRANSCRIPTS)
    a = ap.parse_args()
    CMDS[a.command](a.root)
