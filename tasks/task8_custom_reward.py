"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

from typing import Sequence

_DX = None

def _detox(texts):
  global _DX
  if _DX is None:
      from detoxify import Detoxify
      import torch
      _DX = Detoxify("original", device="cuda" if torch.cuda.is_available() else "cpu")
  raw = _DX.predict(list(texts))["toxicity"]
  return [float(x) for x in raw] if isinstance(raw, (list, tuple)) else [float(raw)]

def _distinct_trigram_ratio(text):
  w = text.split()
  if len(w) < 5: return 1.0
  tri = [" ".join(w[i:i+3]) for i in range(len(w)-2)]
  return len(set(tri)) / max(1, len(tri))

def _prompt_overlap(prompt, resp):
  p, r = set(prompt.lower().split()), set(resp.lower().split())
  if not p or not r: return 0.0
  inter = len(p & r)
  # relevance, but penalise trivial echo (near-total overlap of a short reply)
  jacc = inter / len(p | r)
  echo = inter / max(1, len(r))
  return jacc * (1.0 - max(0.0, echo - 0.8) * 5.0)   # kill reward when reply ≈ echo

def reward_score(
  texts: Sequence[str],
  prompts: Sequence[str] | None = None,
) -> list[float]:
  """Return one reward per completion. Higher = better.

  The verl reward worker calls this once per training step with the
  flattened list of K-rollouts across the prompt batch.

  Args:
    texts: completions to score, one entry per completion.
    prompts: same-length list of the originating prompts (the verl
          dispatcher uses ``reward_score.prompt_conditioned`` below
          to decide whether to pass these). Set the attribute to
          ``False`` if your design is purely response-side.

  Returns:
      ``list[float]`` of the same length as ``texts``. Higher = better.

  See the top-of-file docstring for design hints (saturating Detoxify,
  repetition penalty, length-cap penalty, prompt-relevance, blending
  with your RM).
  """
  prompts = prompts or [""] * len(texts)
  tox = _detox(texts)
  out = []
  for t, p, tx in zip(texts, prompts, tox):
    benign = 1.0 if tx <= 0.10 else (1.0 - tx)     # SATURATE once clearly benign
    rep_pen = 0.5 if _distinct_trigram_ratio(t) < 0.7 else 0.0
    cap_pen = 0.4 if len(t) >= 240 else 0.0        # ran to the 64-token cap
    rel     = _prompt_overlap(p, t)                # ties reward to THIS prompt
    r = benign + 0.3 * rel - rep_pen - cap_pen
    out.append(max(-1.0, min(1.5, r)))
  return out


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
