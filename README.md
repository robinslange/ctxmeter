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
| `invalidation` | When the cache misses, does it lose part of the prefix or all of it? |
| `probes` | How much later-needed information does each retention policy destroy? |
| `tradeoff` | What does each policy save, set against what it destroys and what fetching it back costs? |
| `sensitivity` | Does the policy ranking survive widening the sample? |
| `robustness` | Does any of this survive its own assumptions? |
| `counterfactual` | When a fact is destroyed, does the agent actually change course? |

Everything except `counterfactual` is offline and free.

Every command takes `--json` and prints one JSON document instead of the report.
Percentages are percent numbers at full precision, and policies are keyed by name
in the order the report lists them. `counterfactual` sends its narrative and
`--show-raw` output to stderr, so the document can be redirected on its own.

**If you are reading someone else's measurements, run `robustness` first.** It
attacks the tool's own output three ways: swap the cache model, scale the token
estimator, and replace point estimates with confidence intervals clustered by
session. It is the command most likely to tell you a headline is overstated,
including the ones in [docs/method.md](https://github.com/robinslange/ctxmeter/blob/main/docs/method.md#results-from-one-workload).

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

**Invalidation.** A turn can re-read at most what the turn before it cached.
`invalidation` counts a turn as a clean hit when it re-reads at least 98 percent
of that, and for each miss reports how many tokens dropped out of the cache and
what share of the prefix that was. A share near 100 percent means a miss throws
away the whole prefix, which is the all-or-nothing behaviour `robustness` tests
its cost figures against.

**Retention.** `probes` reports the share of later-needed facts still present
when the agent reached for them. A probe is a distinctive identifier that a tool
result established, and that the agent demonstrably reused much later. Retention
measures information survival, not task success.

**Re-fetch.** `tradeoff` also prices the healthy failure. Each fact a policy
destroyed and the agent later reused is charged one extra request: the masked
prefix read back from cache, and the result that held it written again, once per
result per session. It only sees facts a probe can see, and it assumes every one
is fetched back; `counterfactual` measures how often one actually is.

## Tier two: did losing it matter?

`counterfactual` replays the turn where the agent reused a fact, once with the
context intact and once with the policy applied, and checks whether the literal
string comes back. No model grades itself: the ground truth is what the agent
actually did.

```bash
ctxmeter counterfactual --dry-run --sample 40   # builds and prices every request, sends nothing
ANTHROPIC_API_KEY=... ctxmeter counterfactual --sample 40 --yes
```

This costs real money, up to two calls per turn at full session length. The dry
run prints the ceiling first, and spending requires `--yes`. It needs a real API
key; it will not read a Claude subscription credential, because Anthropic's terms
do not permit using Free, Pro or Max OAuth tokens in another tool. If you keep
keys in a password manager, pass it without writing it to disk:

```bash
ANTHROPIC_API_KEY="$(op read 'op://YourVault/Anthropic/credential')" \
  ctxmeter counterfactual --sample 40 --yes
```

## Method and limits

Before quoting any number ctxmeter gives you, read [docs/method.md](https://github.com/robinslange/ctxmeter/blob/main/docs/method.md):
how tier two scores and prices a run, what the numbers do not establish, the
measurement mistakes it was built to avoid, and results from one workload.

## Privacy

Probe tokens are literal strings lifted from tool output, which means they can
carry credentials, absolute paths and client data. They are interned in memory
and never printed. Every command emits aggregate numbers only, to stdout, and
writes nothing.

`counterfactual` is the single exception, and it is opt-in: it sends
reconstructed context to the Anthropic API, using your key, and writes each
request body to a 0600 file in the temp directory for the length of one call
because that is how the key is kept out of the process list. The dry run sends
nothing at all. `--show-raw` prints the probe token and the model's replies, and
exists so a verdict can be checked by eye rather than trusted; nothing else in the
tool prints probe text.

## Contributing

Bug reports about the measurement method are more welcome than feature requests.
If ctxmeter tells you something that looks wrong, that is worth an issue.

`cargo test` covers the identifier rules and the token accounting, and pins every
command's output over the frozen transcripts in `tests/corpus`.

## About

ctxmeter is an [omit](https://omit.nz) product, built and maintained by omit.
MIT licensed.
