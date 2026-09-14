# ctxmeter

Your coding agent writes a billing record for every request it makes. ctxmeter
reads it, and tells you two things nobody else will: what you are actually paying
for, and what a context-compaction tool would destroy to save you money.

Offline by default. Nothing leaves your machine.

## Why you might want this

There is a growing class of tools that shrink an agent's context to cut your
token bill. They advertise a savings number: 40 to 70 percent fewer tokens, 20
percent for coding agents, same answers.

None of them advertise an information-loss number. Compaction is lossy by
construction, so there is one, and it is measurable.

There is also a good chance you are optimising the wrong thing. If prompt
caching is already working for you, a compression proxy is fighting over a small
remainder, and it can lose: editing context mid-conversation turns cheap cache
reads into expensive writes, and a tool that counts tokens sent rather than
dollars billed will report that as a win.

ctxmeter measures all of it against your own sessions.

## Install

Prebuilt binaries for macOS and Linux are attached to each
[release](https://github.com/robinslange/ctxmeter/releases):

```bash
# pick your platform, verify, run
tar xzf ctxmeter-aarch64-apple-darwin.tar.gz
shasum -a 256 -c ctxmeter-aarch64-apple-darwin.tar.gz.sha256
./ctxmeter-aarch64-apple-darwin/ctxmeter summary
```

Or build it:

```bash
cargo install --git https://github.com/robinslange/ctxmeter
```

It reads `~/.claude/projects` by default. Point it elsewhere with `--root`.

## Start here

```bash
ctxmeter summary
```

You get your cache hit rate, where your input tokens go, and how much prompt
caching is already saving you. If that last number is high, read it as a warning:
the savings a compaction tool is selling you have largely been collected.

## The commands

| Command | What it answers |
|---|---|
| `summary` | What am I paying for, and how much has caching already saved? |
| `floor` | How big is my system prompt plus tool definitions, and is it growing? |
| `probes` | How much later-needed information does each retention policy destroy? |
| `tradeoff` | What does each policy save, set against what it destroys? |
| `sensitivity` | Does the policy ranking survive widening the sample? |
| `robustness` | Does any of this survive its own assumptions? |
| `counterfactual` | When a fact is destroyed, does the agent actually change course? |

Everything except `counterfactual` is offline and free.

**If you are reading someone else's measurements, run `robustness` first.** It
attacks the tool's own output three ways: swap the cache model, scale the token
estimator, and replace point estimates with confidence intervals clustered by
session. It is the command most likely to tell you a headline is overstated,
including the ones in this README.

## How to read the numbers

**Billed equivalents.** Input cost is `fresh + 1.25 x cache_write + 0.10 x
cache_read`, expressed in multiples of the base input price. Stating cost this
way is model-independent, and it makes the central trap visible: on a cached
workload, tokens sent and dollars billed move in opposite directions. Removing
40 percent of your tokens while breaking the cache prefix converts reads at 0.10x
into fresh input at 1.00x.

Never trust a tool's own "tokens saved" dashboard. Read
`cache_read_input_tokens` and `cache_creation_input_tokens` back out of the API
response, which is what ctxmeter does.

**The floor.** Your system prompt and tool definitions are sent ahead of the
messages on every single request, and no context policy can touch them without
breaking the agent. `floor` approximates them from the first billed prefix of
each session. It also tracks them by month, because this number grows quietly as
plugins and tool servers accumulate.

Tool definitions never appear in a transcript, so the floor cannot be attributed
from logs. Change one config item, start a fresh session, and re-run.

**Retention.** `probes` reports the share of later-needed facts still present
when the agent reached for them. A probe is a distinctive identifier that a tool
result established, and that the agent demonstrably reused much later. Retention
measures information survival, not task success.

## Tier two: did losing it matter?

`counterfactual` measures whether a destroyed fact changes what the agent does,
without asking a model to grade itself.

At the block where the agent reused a fact, it had already produced that fact
from that context. So replay that exact turn twice, once with the context intact
and once with the policy applied, and check whether the literal string comes
back. The task is the agent's own next action. The ground truth is what it
actually did.

The intact arm is the control. If it fails to reproduce the fact, that probe
cannot say anything about the policy, so it is discarded and the discard rate is
printed. Of the cases that survive, the outcome splits three ways:

- **Reproduced anyway.** The model did not need the context to get there.
- **Went to fetch it.** It noticed something was missing. The healthy failure.
- **Neither.** The fact was gone and the model did not ask for it back.

```bash
ctxmeter counterfactual --dry-run --sample 40   # builds and prices every request, sends nothing
ANTHROPIC_API_KEY=... ctxmeter counterfactual --sample 40 --yes
```

Two calls per case at full session length, so this is not cheap. The dry run
prints the estimate first, and spending requires `--yes`.

It needs a real API key. It will not read a Claude subscription credential,
because Anthropic's terms do not permit using Free, Pro or Max OAuth tokens in
another tool. If you keep keys in a password manager, pass it without writing it
to disk:

```bash
ANTHROPIC_API_KEY="$(op read 'op://YourVault/Anthropic/credential')" \
  ctxmeter counterfactual --sample 40 --yes
```

## What this does not establish

Read this before quoting any number it gives you.

**That a lost fact is a failed task.** Retention is not success. The agent may
have had another valid route to the answer. Establishing harm needs a restore
counterfactual, and `counterfactual` is only an approximation of one.

**That real tools destroy as much as a bare policy does.** Most keep originals
retrievable, so treating a mask as a deletion is an upper bound on harm rather
than a measurement of it. The other half of that: a model which has lost a fact
does not know to ask for it back.

**That the cost column is a measurement.** Only the baseline uses measured
prefix sizes. Every policy figure is simulated, and its sign can depend on cache
behaviour no external party can observe. `robustness` runs it under two models
that bracket the real thing, and where they disagree, the honest answer is that
the question is open.

**That anyone's numbers transfer to you.** Workload shape decides almost
everything here. That is the whole reason this is a binary you can run rather
than a blog post you have to believe.

Known limits of the tier-two rebuild: the system prompt and real tool schemas are
not recorded in a transcript, so schemas are synthesised from the calls a session
actually made and are permissive. Thinking blocks are dropped, because their
signatures will not validate in a fresh request.

## How it avoids the usual measurement mistakes

Each of these produced a wrong answer during development before it was caught.
They are documented because they are easy to repeat.

- **Image payloads.** Transcripts store images as base64, but the API prices them
  by pixel area. Counting base64 length overstated image-bearing content by about
  2.85x against usage records.
- **Thinking signatures.** Thinking blocks carry a long opaque signature.
  Counting it invented a 24.7 percent cost line for pure metadata.
- **Visible content is not the bill.** Estimated conversation content came to
  roughly a third of the real billed prefix. Any saving expressed as a share of
  visible content overstates by about 3x, so cost is grounded in real prefix
  sizes instead.
- **Mismatched denominators.** Usage rows deduplicate on request and message id;
  content blocks deduplicate on uuid. They give different turn counts, and mixing
  them in one ratio is wrong.
- **Probes must be derived, never authored.** They come out of your corpus.
  `probes` prints sessions scanned against probes yielded and exits non-zero on
  an empty harvest, because a denominator that has stopped growing must not read
  as a pass.
- **Probes are clustered, not independent.** Probes inside one session share a
  trajectory and a mask boundary. Intervals come from a bootstrap over sessions,
  not over probes.
- **A grader would measure the grader.** Ground truth never comes from a model's
  output. It is a literal string from your trace and an observed reuse.

## Privacy

Probe tokens are literal strings lifted from tool output, which means they can
carry credentials, absolute paths and client data. They are interned in memory
and never printed. Every command emits aggregate numbers only, to stdout, and
writes nothing.

`counterfactual` is the single exception, and it is opt-in: it sends
reconstructed context to the Anthropic API, using your key. The dry run sends
nothing at all.

## Results from one workload

These come from the author's own 2,485 sessions on Opus-class and Sonnet-class
models. They are an example of what the tool reports, not a claim about yours.

Prompt caching was already saving 86.9 percent against a no-cache counterfactual
at a 97.3 percent hit rate, and cache reads were 74 percent of the remaining
input bill. The fixed prefix was 50,171 tokens at the median, roughly 38 percent
of every request, and it had grown 73 percent in two months.

Against 7,212 probes across 939 sessions, retention by policy:

| policy | information retained | cost saved (simulated) |
|---|---:|---:|
| keep last 1 tool result | 5.1% | +18.7% |
| keep last 3 | 27.6% | +11.1% |
| keep last 10 | 56.2% | -10.9% |
| keep last 50 | 91.8% | -60.3% |
| tail budget 40k tokens | 87.1% | -30.1% |
| tail budget 100k tokens | 97.8% | -8.8% |

Two things fall out. Every policy that retained a meaningful share of information
cost more than doing nothing. And budget-based retention beat count-based
retention everywhere, because a count cannot tell whether the fourth-from-last
tool result is fifty tokens or fifty thousand.

The positive rows in that table do not survive `robustness`. Under the
all-or-nothing cache invalidation observed on the same traces, keeping the last
tool result moves from +18.7 percent to -117 percent. Which is the point of
having the command.

## Contributing

Bug reports about the measurement method are more welcome than feature requests.
If ctxmeter tells you something that looks wrong, that is worth an issue.

`ctxmeter.py` is a reference implementation kept for cross-checking the Rust.
`cargo test` covers the identifier rules and the token accounting; CI runs the
binary against a synthetic transcript in `tests/fixture`.

MIT licensed.
