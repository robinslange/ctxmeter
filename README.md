# ctxmeter

Measures what Claude Code actually bills you, from your own transcripts in `~/.claude/projects`.

No API calls, no proxy, no config, and nothing leaves your machine. Reads the `usage`
records the CLI already writes.

```bash
cargo build --release
./target/release/ctxmeter summary      # cache hit rate, token classes, billed equivalents
./target/release/ctxmeter floor        # system prompt + tool definitions, by month
./target/release/ctxmeter probes       # how much later-needed information each policy destroys
./target/release/ctxmeter sensitivity  # does the policy ranking survive widening the sample
./target/release/ctxmeter tradeoff     # what each policy saves against what it destroys
```

`ctxmeter.py` is kept as a reference implementation. The Rust binary is what you
distribute; the Python is what you check it against.

```bash
python3 ctxmeter.py summary        # cache hit rate, token classes, billed equivalents
python3 ctxmeter.py floor          # system prompt + tool definitions, by month
python3 ctxmeter.py composition    # carry cost by content type
python3 ctxmeter.py invalidation   # is cache invalidation partial or all-or-nothing
python3 ctxmeter.py probes         # how much later-needed information each policy destroys
python3 ctxmeter.py sensitivity    # does the policy ranking survive widening the sample
```

## The probe benchmark

Every context-compaction tool ships a token-savings number. None ships an
information-loss number. `probes` measures the second one.

Ground truth is not authored and no model is asked to judge. A probe is a
distinctive identifier that a tool result established at block *i* and that the
agent demonstrably reused at block *j*, far later. The fact that the agent used
it is a mechanical property of the trace; the answer is the literal string. A
model labelling its own recall would only measure imitation of the labeller.

Rules the harvester enforces:

- **Derived, never authored.** Probes come from the corpus. The command prints
  sessions scanned against probes yielded, and exits non-zero on an empty
  harvest, so a denominator that stopped growing cannot read as a pass.
- **Non-guessable.** A probe token must be at least 10 characters, contain a
  digit, and appear in at most `max_df` sessions corpus-wide.
- **Not user-supplied.** Any token the user typed is excluded; recalling it is
  not a memory test.
- **A real gap.** The reuse must be at least `min_gap` blocks after the origin.

It measures **information retention, not task success.** Losing a fact is not
proof of failure, because another valid route may exist. Tier two, which asks
whether a model can still answer once a fact was summarised rather than deleted,
is where judging gets hard and is not built.

### First result, 1,952 sessions, 7,530 probes

| policy | facts retained when needed |
|---|---|
| keep last 3 tool results | 28.1% |
| keep last 10 | 57.3% |
| keep last 25 | 80.2% |
| tail budget 40k tokens | 87.4% |
| tail budget 100k tokens | 97.8% |

Keeping the last three tool results is a shipped default. It destroys roughly
72% of the information the agent went on to use.

**Budget-based retention beats count-based, robustly.** A count policy cannot
tell whether the fourth-from-last tool result is 50 tokens or 50,000. At
`min_gap` 20 the count policies fall to 8-13% while the budget policies stay
above 96%. `sensitivity` sweeps rarity and gap thresholds and exits non-zero if
the ranking moves; across six conditions it does not.

## Why billed equivalents, not tokens

Input cost is `fresh + 1.25 x cache_write + 0.10 x cache_read`, in multiples of base input
price. Expressing cost this way is model-independent on the input side and makes the trap
visible: on a cached workload, tokens sent and dollars billed move in **opposite**
directions. A compressor that removes 40% of tokens while breaking the prefix converts
reads at 0.10x into fresh input at 1.00x, and reports a win.

Never trust a tool's own "tokens saved" dashboard. Read `cache_read_input_tokens` and
`cache_creation_input_tokens` back from the response.

## Measurement traps this tool avoids

Each of these produced a wrong answer before it was caught.

- **Image payloads.** Transcripts store images as base64, but the API prices them by pixel
  area. Counting base64 length overstates image-bearing content by ~2.85x, calibrated
  against usage records. `IMG_SCALE` corrects it.
- **Thinking signatures.** Thinking blocks carry a long opaque `signature` field. Counting
  it produced a spurious 24.7% cost line for content that is metadata. Only `thinking` text
  is counted.
- **Visible content is not the bill.** Estimated conversation content is about 0.32 of the
  real billed prefix. Any saving expressed as a fraction of visible content overstates by
  roughly 3x. `composition` prints the calibration ratio so the gap stays in view.
- **Two dedup keys.** Usage rows dedupe on `requestId` + `message.id`; content blocks dedupe
  on `uuid`. They yield different turn counts. Do not mix the two in one ratio.
- **The floor is invisible.** Tool definitions never appear in a transcript, so the floor
  cannot be attributed from logs. Change one config item, start a fresh session, and read
  the first-turn prefix back.

## What it found here

- Prompt caching already saves 86.9% against a no-cache counterfactual, at a 97.3% hit rate.
- Cache reads are 74% of the remaining input bill.
- The fixed floor is 50,171 tokens median, about 38% of a median prefix, re-read every turn,
  and up 73% in two months as plugins accumulated.
- Invalidation is all-or-nothing: 98.7% of turn pairs are clean incremental hits, and the
  rest lose a median 148,152 tokens, roughly a whole prefix. That is what breakpoint-anchored
  matching with a 20-block lookback predicts, and it means mid-history edits rarely pay.


## Parity

The Rust binary and the Python reference are checked against each other on the
same corpus. `summary` agrees exactly. `probes` agrees within one point on every
policy and gives the identical ranking; the small gap is deliberate, because the
Rust build tests exact identifier membership per block where the Python tests
substring containment.

| policy | Rust | Python |
|---|---|---|
| keep last 3 | 27.6% | 28.1% |
| keep last 10 | 56.2% | 57.4% |
| keep last 25 | 79.2% | 80.2% |
| tail budget 40k | 87.1% | 87.4% |
| tail budget 100k | 97.9% | 97.8% |

Full corpus, 2,485 sessions: 3.9s in Rust against 15.9s in Python. The reason to
ship the Rust build is not speed, it is that replicating a finding should cost a
stranger one command and no language runtime.

## The tradeoff

`tradeoff` reports both numbers from the same policy applied the same way. Cost is
grounded in the real prefix sizes from the usage records, so the system prompt and
tool definitions are carried unchanged: no context policy can touch them.

| policy | cost saved | info retained |
|---|---:|---:|
| keep last 1 | +18.6% | 5.1% |
| keep last 3 | +11.0% | 27.6% |
| keep last 5 | +4.1% | 38.8% |
| keep last 10 | -10.9% | 56.2% |
| keep last 25 | -39.2% | 79.2% |
| keep last 50 | -59.7% | 91.8% |
| tail budget 10k | -9.0% | 53.0% |
| tail budget 40k | -30.3% | 87.1% |
| tail budget 100k | -8.8% | 97.9% |
| tail budget 200k | +0.7% | 99.9% |

Every policy that retains a meaningful amount of information costs **more** than
doing nothing. The mechanism: once the mask boundary advances, the whole kept
region is re-written at 1.25x instead of read at 0.10x, so the penalty is roughly
how often it fires multiplied by how much it keeps. That product peaks in the
middle, which is why 40k is worse than both 10k and 100k.

Masking pays only at the extremes, keeping almost nothing or almost everything.
A shipped default of keeping the last 3 tool results sits inside the only band
where it saves money at all, and retains 27.6%.

This is the mechanised form of Anthropic's own guidance for the context-editing
beta: clear enough tokens to make the cache invalidation worthwhile.

## Privacy

Probe tokens are literal strings lifted from tool output, so they can contain
credentials, absolute paths, and client data. They are interned in memory and
never printed. Every command emits aggregate numbers only. Nothing is written
anywhere except stdout.
