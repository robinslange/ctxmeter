# ctxmeter

Measures what Claude Code actually bills you, from your own transcripts in `~/.claude/projects`.

No API calls, no proxy, no config. Reads the `usage` records the CLI already writes.

```bash
python3 ctxmeter.py summary        # cache hit rate, token classes, billed equivalents
python3 ctxmeter.py floor          # system prompt + tool definitions, by month
python3 ctxmeter.py composition    # carry cost by content type
python3 ctxmeter.py invalidation   # is cache invalidation partial or all-or-nothing
```

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
