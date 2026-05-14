# Twitter / X thread: fin-safety-inspect v0.4 — language-conditioned safety

8 tweets, each at or under 280 characters. Copy each numbered block in turn.

---

## 1/8 — Hook

```
Same 100 financial-crime prompts. Same DeepSeek-chat model. Two languages.

True Mode #3 (operational criminal guidance) in EN: 0/100.
True Mode #3 in ZH: 7/100.

Including a 4-step hawala walkthrough with codeword + 1% commission.

We hand-classified all 132 ff samples. 🧵
```

## 2/8 — The 7 Mode #3 incidents

```
What DS-chat handed over in Chinese, not in English:

1. Structuring (cash-split bypass)
2. Hawala 4-step walkthrough
3. Card skimmer install + PIN capture
4. Illegal-lending terminology evasion
5-7. Identity-theft attack playbooks

Zero of these in the EN baseline.
```

## 3/8 — The methodology twist

```
LLM-judge ff in Chinese: 13.
Hand-classified Mode #3: 7.
False-positive rate: 46%.

In English: ff=3, Mode #3=0. FP rate 100%.

The judge marks any output discussing criminal technique as FAIL — even when it's compliance education or refusal-with-explanation.
```

## 4/8 — Why this slipped past v0.1-v0.3

```
LLM-judge FP rate is high but model-uniform within a language.

Cross-model rankings stay correct.
Cross-language claims do NOT.

When you change the language, the judge's FP rate moves with you. v0.4 only exposed this because we ran the same task in 3 languages.
```

## 5/8 — Qwen-plus is cross-language stable

```
Qwen-plus stays at 0-1 true Mode #3 from EN to ZH.
DeepSeek-chat goes 0 → 7.

Judge accuracy looks similar (0.81 vs 0.91 in ZH), but the hand-classified Mode #3 counts are 7x apart.

Cross-model gap on Chinese > cross-model gap on English.
```

## 6/8 — Hausa breaks both models for opposite reasons

```
Hausa ff: DS-chat 25, Qwen-plus 85. Looks like Qwen collapsed.

Hand classification:
- DS-chat: 5 Mode #3, 12 refusals the scorer can't read in Hausa
- Qwen-plus: 10 Mode #3, 57 off-topic comprehension failures

Same ff. Completely different failure layers.
```

## 7/8 — Cumulative methodology lesson

```
v0.1: baseline unflattering → hand-triage FNs.
v0.2.1: baseline counterintuitive → cross-model.
v0.3: failure mode w/ multiple fixes → validate paths independently.
v0.4: LLM-judge ff is a screen, not a verdict. Hand-classify before reporting cross-language safety claims.
```

## 8/8 — Cite + thanks

```
Standing on Inspect AI (UK AISI) and FinTrust (Hu et al., EMNLP 2025).

Apache-2.0, 61 unit tests, no API key needed for them.

Full report + reproducible drivers:
github.com/acai77777/fin-safety-inspect
```

---

## Posting checklist

- [ ] Tweet 1 has 🧵 emoji to signal thread (only emoji in entire thread)
- [ ] Tweet 1 frames the surprise: same model, same prompts, just language → 7x more operational guidance. This is the hook
- [ ] Tweet 2 names concrete operational categories — gives evaluators something to verify
- [ ] Tweet 3 carries the methodology — quotable for safety-eval Twitter
- [ ] Tweet 4 generalises the methodology — frames why v0.1-v0.3 weren't broken either
- [ ] Tweet 5 is the cross-model comparison: corrects v0.2.1 ranking; deployers will quote this
- [ ] Tweet 6 is the Hausa decomposition; methodology audience will quote this most
- [ ] Tweet 7 is the cumulative methodology lesson chain v0.1→v0.4
- [ ] Tweet 8 has the canonical repo URL
- [ ] Pin tweet 1 to profile after posting
- [ ] Optional quote-RTs for visibility:
  - @AISafetyInst (UK AISI, Inspect AI maintainers)
  - FinTrust paper authors if they're on X
  - Multilingual NLP / low-resource-language eval researchers (tweet 6 is on-topic)

## Thread vs. link-post

If you post the LessWrong long-form first, link to it in tweet 8. If you post the thread cold, the repo URL is fine.

## Variants if 280-char tight

- Tweet 2: drop the parenthetical examples on items 1, 3, 4. The numbered list carries the point.
- Tweet 3: drop "compliance education or" — "refusal-with-explanation" carries the point.
- Tweet 6: drop "Looks like Qwen collapsed." — the numbers in the next line carry it.

Do NOT cut:
- Tweet 1's "4-step hawala walkthrough with codeword + 1% commission" — this is the most concrete + most credible single quote in the thread
- Tweet 3's "FP rate 100%" callback — methodologists will check this
- Tweet 7's full chain — it ties this thread to v0.1-v0.3 and signals continuous work
