# Method

How ctxmeter arrives at its numbers, what they do not establish, and the
mistakes it was built to avoid. The [README](../README.md) covers install and
the commands.

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
printed. Of the cases that survive, the outcome splits four ways:

- **Reproduced anyway.** The model did not need the context to get there, and it
  acted on the fact.
- **Rewrote it into prose.** The fact came back inside text rather than inside an
  action. Counted apart, because a document regenerated around a string is a
  weaker reuse than acting on it, and on the first measured corpus this was the
  only way a control arm ever reproduced anything.
- **Went to fetch it.** It noticed something was missing. The healthy failure.
  Counted only when the next action names what produced the fact. Matching on the
  tool name instead would score most of a coding corpus as a re-fetch, since most
  of it is `Read`, and that error lands entirely in this bucket.
- **Neither.** The fact was gone and the model did not ask for it back.

### Cost

At most two calls per turn at full session length: the control arm, then the
treatment arm, which is only bought if the control reproduced the fact. The unit
priced is the turn, not the fact: one replayed turn carries every fact the
policy destroyed in it, and the dry run prints that ratio.

Selection is deterministic over a given corpus, so a smaller sample is the first
turns of a larger one, and `--skip` passes over turns an earlier run already
bought (see `ctxmeter counterfactual --help`). On one corpus the order moved two
positions in a day as transcripts were added. The selection counts in the header
describe only the turns after the skip.

A run covers one model family, because pooling two families into a single
retention figure conflates them. That also happens to be where most of the cost
lives. Opus input costs two and a half times Sonnet-5 input per token, but the
gap between two runs is wider than that, because the turns are not the same
turns: on one corpus the same 133-turn sample priced at a $29.97 ceiling under
`--model sonnet-5` and $99.29 under `--model opus-5`, about 3.3x, and the Opus
sample carried 2.36 facts per turn against Sonnet's 1.90. Both figures are
ceilings the tool printed, not bills. Price your own corpus; that is what the
dry run is for.

The ceiling is wrong in known directions, and the run prints the measured count
beside it. Turns are selected to cover as many distinct sessions as the sample
allows, because a retention figure bootstrapped from a sample concentrated in a
handful of sessions cannot stand in for the corpus. When a sample is large
enough to revisit a session, its turns are sent adjacently so a later control
arm can extend an earlier one's cached prefix — but the two arms of one turn
never share a prefix with each other, because the policy rewrites the earliest
tool results and the two rebuilds diverge near the start. Below the corpus's
session count, a run gets no cache reuse at all. The ceiling prices every input
token as a cache write, at 1.25x, and never at the 0.1x a genuine cache read
would bill. It bounds output at the ceiling for two arms of every turn, where a
real turn produces a fraction of that, and it buys both arms of every turn where
a turn whose control arm fails never buys its second. Against all of that, token
counts come from serialized length rather than the tokenizer, which undercounts.
The magnitudes are corpus-specific and the first measured run put the whole
product near 3x high, from five replies, eight control arms and two token counts
on one corpus — too thin to quote as a correction factor.

Before either paid call, both arms go to `/v1/messages/count_tokens`, which is
free. It answers the only question worth answering first, which is whether the
API accepts what was rebuilt, and it returns the measured count.

### Known limits of the rebuild

The system prompt and real tool schemas are not recorded in a transcript, so
schemas are synthesised from the calls a session actually made and are
permissive. Thinking blocks are dropped, because their signatures will not
validate in a fresh request. Tier two sizes a tool result as serialized length
over 4 with no image scaling, while the retention tables size the same block
with the scaled estimator, so a `tail_budget_*` policy masks a different set of
results than the table reports, which moves the budget boundary on an
image-bearing result (see image payloads below). `keep_last_*` counts results
rather than tokens and is unaffected. And turns are spread across distinct
sessions, but session order comes from an unsorted filesystem walk over one
directory per project, so a sample smaller than the corpus's session count is
the first N sessions that walk returned: many sessions, possibly drawn from few
projects.

## What this does not establish

Read this before quoting any number ctxmeter gives you.

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

## Results from one workload

These come from one developer's 2,485 sessions on Opus-class and Sonnet-class
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

Two things fall out. Every policy that retained a meaningful share of
information cost more than doing nothing. And budget-based retention beat
count-based retention everywhere, because a count cannot tell whether the
fourth-from-last tool result is fifty tokens or fifty thousand.

The positive rows in that table do not survive `robustness`. Under the
all-or-nothing cache invalidation observed on the same traces, keeping the last
tool result moves from +18.7 percent to -117 percent. Which is the point of
having the command.
