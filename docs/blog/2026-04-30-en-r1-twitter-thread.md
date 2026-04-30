# Twitter / X thread: fin-safety-inspect v0.2.1 cross-model

8 tweets, each at or under 280 characters. Copy each numbered block in turn.

---

## 1/8 — Hook (the counterintuitive headline)

```
We tested 5 Chinese LLMs on financial-crime refusal (FinTrust × 100 prompts each, mock LangGraph banking agent).

The reasoning model came last.

DeepSeek-R1 keyword accuracy: 0.670. Same family's DeepSeek-chat: 0.790. R1 refuses *less*, not more. 🧵
```

## 2/8 — The full ranking

```
Ranked by judge accuracy (LLM judge pinned to DeepSeek-v4-flash for cross-model comparability):

Qwen-plus      0.950
DeepSeek-chat  0.910
GLM-4.5        0.880
DeepSeek-R1    0.840
Kimi K2        0.830

Qwen wins. Reasoning model 4th of 5.

github.com/acai77777/fin-safety-inspect
```

## 3/8 — Why R1 might be losing

```
R1's reasoning RLHF reinforces "decompose the problem, give detailed analysis" more strongly than "refuse early."

When a financial-crime prompt arrives, R1's thinking tokens go into engagement, not refusal.

Don't assume reasoning models are automatically safer. The data says no.
```

## 4/8 — The DeepSeek soft-refusal fingerprint

```
judge accuracy − keyword accuracy:
  DeepSeek-chat: +0.120
  DeepSeek-R1:   +0.170
  Qwen / GLM / Kimi: +0.010 to +0.020

DeepSeek family prefers "I'd recommend consulting a legal professional" — soft refusals.
The other 3 prefer hard refusals. Keyword scorers will rank DeepSeek unfairly low.
```

## 5/8 — Same prompt, different failure mechanisms

```
Illegal Lending failures:
- DeepSeek-chat: agent calls list_accounts() on a loan-shark prompt → returns "You have CHK-001..." (Mode #1, engineering bug)
- GLM-4.5: produces "Best Practices for Informal Lending" checklist (Mode #3, training bug)

Same FAIL, completely different mechanisms.
```

## 6/8 — What's universal, what's not

```
Mode #1 (agent off-task tool misuse): DeepSeek-specific. Qwen has 0/100.

Mode #3 (genuinely unrefused, gives operational detail): universal. Every model has ff in Counterfeit. No scorer-layer fix exists.

The two modes were both reported as "v0.1 findings." Cross-model says one is local, one is global.
```

## 7/8 — Roadmap rewritten from data

```
v0.2.0 said v0.2.1 priority was a trace scorer for Mode #1.

Cross-model data: Mode #1 only happens on DeepSeek. Trace scorer becomes a DeepSeek engineering aid, not a general benchmark feature.

DOWNGRADED.

PROMOTED: output-side guardrail (the only fix for Mode #3, the universal one).
```

## 8/8 — Cite + thanks

```
Standing on Inspect AI (UK AISI) and FinTrust (Hu et al., EMNLP 2025).

Apache-2.0, 31 unit tests, no API key required to run them. Dataset commit pinned for repro; backbones drift (DeepSeek-chat moved +0.090 in 5 hours).

Full writeup: github.com/acai77777/fin-safety-inspect/blob/main/docs/cross-model-baseline.md
```

---

## Posting checklist

- [ ] Tweet 1 has 🧵 emoji to signal thread (only emoji in entire thread)
- [ ] Tweet 2 has the canonical repo URL (drives clicks)
- [ ] Tweet 3 carries the central claim about R1 (the most quotable line in the thread)
- [ ] Tweet 5 has a verbatim sample-pair contrast (this is the methodological hook for evaluators)
- [ ] Tweet 7 is the roadmap revision; readers in safety eval will find this most useful
- [ ] Tweet 8 has the second URL (cross-model writeup) for readers who scroll past the first
- [ ] Pin tweet 1 to your profile after posting
- [ ] Optional quote-retweets for visibility:
  - @AISafetyInst (UK AISI, Inspect AI maintainers)
  - FinTrust paper authors if they're on X
  - Hugging Face's safety / agent benchmark accounts

## Thread vs. link-post

If you post the long-form blog (LessWrong / Hacker News) before the thread, link to that in tweet 2 instead of the repo (more dwell-time, more clicks deeper). If you post the thread first as the cold-traffic hook, send tweet 8's writeup link to drive depth.

## Variants if 280-char tight

OpenRouter occasionally counts URLs as 23 chars regardless of length. If a tweet exceeds:
- Drop the prefixed model rankings in tweet 2 → keep just the names + numbers, no row separators
- Cut the parenthetical in tweet 4 (`"... + soft refusals"`)
- Tweet 6 can lose the second sentence about Counterfeit; the contrast in the first sentence carries the point

Do NOT cut tweet 5's sample contrast. That's the entire hook for the methodology-minded reader.
