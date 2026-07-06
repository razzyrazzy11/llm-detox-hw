# Task 8 — Custom Reward: Design Narrative and Analysis

## The problem this task actually poses

By Task 8 the two prior PPO runs had each "succeeded" on its own reward and
each failed as a *detoxifier* — in opposite directions. The task is not to
get a low toxicity number; it is to design a reward a policy cannot cheaply
game, so that low toxicity comes with diverse, prompt-responsive generation
rather than a degenerate shortcut.

## What collapsed into what

**Task 6 — `inv:detoxify` (reward = 1 − Detoxify).** Training reward
saturated at ~0.999 (greedy Detoxify ≈ 0.001), which looked like a clean win.
The K=16 diagnostic exposed it as a reward hack: on the challenging
RealToxicityPrompts slice, sampled `support` was **0.34** and `mean_max`
**0.351** — a toxic tail actually *worse than DPO* (0.12 / 0.12). The
mechanism, visible in the worst-of-16, was **off-distribution evasion**: the
policy escaped the English-tuned classifier by leaving English entirely
(Thai, Chinese, Gujarati), emitting code-like tokens, and canned boilerplate.
`mean_std` stayed moderate (0.104) — not a template collapse, but a
classifier blind-spot exploit that left real toxicity on hard prompts.

**Task 7 — the trained RM (0.97 held-out pairwise accuracy).** The opposite
failure. Sampled `support` went to **0.000 on every slice**; `mean_std`
cratered to **~0.0018**. All 15 worst-of-16 completions were the *same*
string — "Добрый день! Спасибо!" ("Good day! Thank you!"), repeated. Textbook
single-template mode collapse. The RM was an excellent *classifier* of
toxic-vs-benign but a poor *objective* for generation: a short, polite,
content-free greeting is maximally "benign" and carries none of the toxic
features the RM learned to penalise, so it sits at a reward optimum the RM had
no negative examples for. The small training set (1961 pairs, RM train
accuracy → 1.0 vs 0.97 held-out) plausibly sharpened this blind spot.

The two failure modes I therefore had to defeat simultaneously:
1. **Classifier evasion** (Task 6): winning by going off-distribution.
2. **Template collapse** (Task 7): winning by emitting one safe string for
   every prompt.

## Final design

A multi-term, prompt-conditioned reward, each term aimed at a specific
failure mode:

```python
benign  = 1.0 if tox <= 0.10 else (1.0 - tox)   # saturate once clearly benign
rep_pen = 0.4 if distinct_trigram_ratio(t) < 0.7 else 0.0   # vs template collapse
cap_pen = 0.4 if len(t) >= 240 else 0.0          # vs run-to-cap degeneration
rel     = prompt_overlap(prompt, t)              # tie reward to THIS prompt
reward  = benign + 0.3 * rel - rep_pen - cap_pen           # clamp [-1.0, 1.5]
```

Reasoning per term:
- **Saturating `benign` at Detoxify ≤ 0.10** removes the incentive to push
  toxicity ever-lower once a response is already clearly benign — the driver
  of off-distribution over-optimisation. Past the threshold, extra
  "cleanliness" earns nothing, so the policy has no reason to abandon English.
- **Repetition penalty** (distinct-trigram ratio) punishes the looped-phrase
  pattern Task 7 collapsed into.
- **Prompt-relevance** (`rel`, with an anti-echo guard) is the key anti-
  collapse term: a single template cannot be relevant to every prompt, so
  rewarding prompt-overlap makes Task 7's "one string for everything"
  unprofitable.
- **Length-cap penalty** discourages degenerate always-max-length runs.

## Did it work? Honest answer: mostly — and it is the best of the three runs.

Grounded in the Task 8 eval (`task8_ppo_custom_eval.json`), rtp_challenging
(support / mean_max / mean_std):

| Run | support | mean_max | mean_std | Failure mode |
|---|---|---|---|---|
| SFT | 0.620 | 0.568 | 0.165 | baseline |
| DPO | 0.120 | 0.122 | 0.034 | — |
| PPO `inv:detoxify` (T6) | 0.340 | 0.351 | 0.104 | classifier evasion, toxic tail |
| PPO `rm` (T7) | 0.000 | 0.009 | **0.0018** | template collapse |
| **PPO custom (T8)** | 0.280 | 0.299 | **0.087** | **no collapse** |

Full Task 8 K=16 diagnostic (support / mean_max / mean_std):
- mild_prefix: 0.133 / 0.209 / 0.054
- direct_provocation: 0.000 / 0.060 / 0.017
- rtp_challenging: 0.280 / 0.299 / 0.087

Against the three questions:

- **Did the template attractor disappear? Yes.** 13 of 15 worst-of-16
  completions are distinct (Task 7: 0 of 15). Completions are prompt-
  responsive ("Roast me…" → "Roast you again"; "Describe my coworker…" → a
  sentence about the coworker interrupting). No single repeated string.

- **Is `mean_std` healthier than Tasks 6/7? Yes, decisively.** 0.087 on
  rtp_challenging is ~48× Task 7's collapsed 0.0018, and in the same healthy
  band as Task 6 (0.104) and DPO (0.034). Across all slices it sits at
  0.017–0.054 with no near-zero values — the diversity signature of a policy
  that has *not* mode-collapsed.

- **Toxicity is genuinely reduced without the collapse:** greedy toxicity
  0.001–0.05; rtp_challenging sampled support fell from SFT's 0.62 to 0.28 —
  real detox, achieved *with* diversity intact, unlike Task 7's degenerate
  zero.

## What still fails — the residual, verified at the token level

The worst-of-16 contains three empty completions and several one-to-two-word
completions ("You", "Yes."). I did not trust the decoded strings: an empty
string in the eval JSON could be a genuine generation or a `skip_special_
tokens` decode artifact. I reproduced generation for an empty-producing prompt
and inspected the **raw token IDs** before decoding. The result settles it:

- Every sampled completion is a few real tokens followed by a run of Qwen's
  EOS token (`151643`). Examples (raw ids → skip-specials decode):
  - `[2610, 151643, 151643, …]` → `'You'`
  - `[2610, 1521, 0, 151643, …]` → `'You did!'`
  - `[9454, 13, 151643, …]` → `'Yes.'`
  - `[151643, 151643, …]` → `''`  (EOS emitted first → genuinely empty)

So the empties are **real model output, not a decode artifact** — and the
sharper characterisation is that the residual failure is not "the model says
nothing" but **early-EOS truncation**: the policy learned to emit one or two
tokens and then stop. Empty strings are the limiting case where EOS came
first.

Why this is a reward exploit: a 1–2 token reply like "You" incurs no
repetition penalty, no length-cap penalty (it is far under 240 chars), and
scores low toxicity — so it is a cheap local optimum the reward does not
punish. The `cap_pen` term penalises completions that are *too long*; nothing
penalises completions that are *too short*. Early-EOS truncation is the
milder, subtler successor to the Tasks 6–7 collapses: not off-distribution
evasion, not one repeated template, but a fallback to minimal content that the
current reward leaves on the table.

A second, milder artifact: one worst case (R=0.729) is the policy echoing the
hostile *prompt* back, which Detoxify then scores as toxic because the echoed
*prompt* is hostile — not the model's own contribution. This is a scoring
artifact of a prompt-conditioned eval, and it shows the anti-echo guard in
`prompt_overlap` is imperfect.

## What I would do next

The token-level diagnosis points at one precise fix rather than a vague one:
a **minimum-length / early-EOS penalty** — a small penalty for completions
below a token floor (e.g. < 8–10 tokens). This targets the observed mechanism
(EOS-dominated short generations) directly, closing the truncation gap without
the guesswork of a generic "encourage substance" term. A stronger anti-echo
term (penalising near-verbatim prompt repetition) would address the second
artifact. Both are additive to the current reward and target exactly the two
behaviours the eval surfaced.

## The through-line

Each reward taught the policy something about the reward's own blind spot: a
fixed classifier is evaded off-distribution; a learned preference model is
collapsed onto a content-free safe point; a multi-term reward that ties reward
to *this prompt* and punishes repetition removes both cheap shortcuts, leaving
only subtler ones — early truncation and occasional echo. The lesson is that 
every proxy is potentially gameable, and the engineering task is not to find an ungameable 
reward but to make the cheapest remaining exploit progressively less harmful. 
Task 8's reward did that — its worst residual behaviour (short/EOS-truncated replies) 
is far less damaging than Task 6's surviving toxic tail or Task 7's total collapse 
— and I verified that residual at the token level rather than asserting it.