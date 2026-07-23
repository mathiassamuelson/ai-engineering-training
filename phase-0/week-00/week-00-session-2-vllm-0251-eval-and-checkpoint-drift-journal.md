# Week 00 · Session 2 — vLLM v0.25.1 evaluation (null result) and checkpoint drift

**Date:** 2026-07-23
**Host:** inference, RTX PRO 6000 Blackwell Max-Q, driver 580.159.03
**Surface:** Claude chat
**Scope:** Single-variable engine evaluation (v0.23.0 → v0.25.1) on the 31B-QAT config, motivated by upstream Gemma 4 sliding-window/KV fixes; then checkpoint-refresh diligence for both QAT models.

---

## Context

Session 1's KV "clean miss" prompted a hypothesis (Mathias, from memory) that vLLM releases after our v0.23.0 pin fixed Gemma 4 KV handling. Upstream evidence located: a v0.24 change unifying FlashAttention (FA4) across Gemma 4's layers, further Gemma 4 sliding-window/FA4 fixes in v0.25.0, and an upstream issue matching our symptom (small KV pools for Gemma 4 at INT4). Experiment: hold model snapshot + all serving flags constant, vary only the engine image.

Reboot noted at session start (monitoring containers freshly up); session-1 startup logs in /tmp lost, partially recovered later from terminal scrollback. Persistence mode re-enabled manually (reverts on reboot — durable fix still queued).

## KV arithmetic from config (prediction basis)

From the 31B config (`text_config`): 60 layers, 5:1 pattern → **50 sliding-window (window 1024) + 10 full-attention**. Heterogeneous geometry in both dimensions: sliding = 16 KV heads × 256 head_dim; full = 4 KV heads × 512 global_head_dim. `attention_k_eq_v: true` (K and V identical — storable once if exploited).

Per-token bounds at MML 131072, BF16 KV:

| policy | KiB/token | pool tokens at 66.84 GiB |
|---|---:|---:|
| naive (full context everywhere, K+V) | ~880 | ~80K |
| SWA-aware, K+V separate | ~86 | ~810K |
| SWA-aware + k_eq_v exploited | ~43 | ~1.63M |

**Measured v0.23.0: 111.4 KiB/token** — matches no clean policy. Working model (Mathias's recollection + arithmetic fit): SWA-aware allocation ✓, K/V stored separately ✓, residual **~1.29× overhead attributed to hybrid-KV-manager grouping/padding** [assumed — residual attribution, not directly observed; consistent with upstream design doc's acknowledged grouping inefficiency at our 50:10 split].

This supersedes session 1's miss analysis: the eyeball error wasn't just missing layer count — it was not modeling window amortization or storage policy at all.

## Predictions (committed before boot)

1. **KV pool, bimodal:** (A) k_eq_v exploitation landed → ~1.15–1.35M tokens; (B) fixes were kernel-level only → pool moves <10% from 629,403. In-between falsifies the working model.
2. **Decode:** 63–69 tok/s, unchanged within noise (weight-read bound; weights identical).
3. **Boot:** passes; Marlin WNA16 reselected. Flagged risks: Model Runner V2 default, device-selection changes in v0.25.
4. **Prefill (tightened after backend confirmation, below):** within ±10% of ~3,120 tok/s at 2048.

## Execution notes

- `start-vllm.sh` turned out to already have an `--image` override (line 132) — undocumented in the help header. My prior claim "no override, bypass the launcher" was wrong; concluded from help text without checking the flag loop. Used the launcher with `--image vllm/vllm-openai:v0.25.1` — guarantees flag identity with session 1's boot. Help-text drift logged for T.
- Image provenance: `v0.25.1` = `sha256:e4f88a835143…4766089`.
- Bringup checks: **PASS** on all gates; log-scan signatures survived the version jump.

## Results

**KV: Outcome B, cleanly.** 68.65 GiB / **646,462 tokens** (+2.7%) / 4.93×. Per-token cost **111.4 KiB/token — identical to v0.23.0 to three significant figures.** Allocation policy unchanged; the small pool gain is ~1.8 GiB more free memory, not a policy change.

**Attention backend finding:** v0.25.1 log states the mechanism outright — *"Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). FA4 not available, forcing TRITON_ATTN backend."* The geometry that explains the KV layout **also excludes this model from FA4.** Scrollback recovery then showed **v0.23.0 also ran TRITON_ATTN** — same backend both engines, so the sweep comparison is backend-controlled. (Scope caveat: "no allocation change" is a claim about *this model at this config*, not about the upstream fix in general — it may target variants we don't run.)

**Sweep, side by side (c=1, same ladder, same tool):**

| prompt | prefill v0.23.0 | prefill v0.25.1 | decode v0.23.0 | decode v0.25.1 |
|-------:|----------------:|----------------:|---------------:|---------------:|
| 128    | ~2624           | ~2354           | 66.0           | 66.0           |
| 512    | ~3341           | ~3341           | 64.5           | 64.6           |
| 2048   | ~3118           | ~3121           | 62.6           | 62.8           |

**Scores:** decode PASS (statistically identical). Prefill-2048 PASS (3121 vs 3120). **Unpredicted observation:** prefill-128 down ~10% — arithmetic implies a fixed **~+5 ms per-request overhead** (~50→~55 ms wall at that size), amortized invisible at 512+. Candidate sources: Model Runner V2 request path, FlashInfer sampler setup, first-boot autotune-cache miss ("Falling back to default tactics" in log) [mechanism inferred from timing; source unattributed]. Not scored — no prediction existed at that size.

## Conclusion (engine experiment)

**v0.25.1 is neutral for this deployment.** No KV capacity change (allocator untouched for this config), no FA4 (geometry-excluded on both engines), throughput unchanged except a ~5 ms short-prompt overhead. Null result, kept as a finding: the successor-pin decision rests on forward-looking grounds (currency for checkpoint work, future model support, fix runway), not on measured benefit today.

## Checkpoint refresh — diligence findings

Intended as "step 2: update both QAT checkpoints." Diligence inverted the premise:

1. **The refresh already happened, silently, during session 1.** Cache holds HEAD snapshots for both QAT models with creation timestamps *mid-boot on Jul 22* (31B `52f3f65b` at 03:26, root-owned — created by the container). The launcher passes no `--revision`; vLLM resolves `main` at boot; HEAD had moved Jul 20. **Both session-1 boots and today's boot served HEAD `52f3f65b`, not the June snapshot** — my repeated "pinned at ca9d5250" claims were an unverified assumption from `find | head -1` ordering. Saving grace: all three boots served the *same* snapshot, so the engine comparison's single-variable integrity holds — confirmed by timestamp, not assumed.
2. **The drift is materially near-empty.** Blob-level comparison, June → HEAD (31B): `config.json`, `model.safetensors`, `generation_config.json`, `tokenizer.json` are the **same blobs**. Weights provably unchanged → QAT≈BF16 quality baselines undisturbed by checkpoint drift. Changed: `chat_template.jinja` + `tokenizer_config.json` (canonical template, Jul 15; response_template, Jul 20). **The template change is a behavior change for all chat-endpoint output** — logged as a mandatory variable in any future quality comparison against concluded-program baselines. Throughput sweeps (raw /v1/completions) unaffected.
3. **12B MML recollection: Mathias correct; my "falsified" was the error.** I tested the recollection against the two QAT repos' commit histories only and declared it falsified. The fix lives on the **BF16 parent** `gemma-4-12B-it` (in our roster as quality reference): commit `5926caa`, Jun 4 — `max_position_embeddings` 131072 → 262144, one line. Both QAT exports were cut Jun 5, *after* the fix, and carry 262144 (verified directly in both configs). Scored: recollection correct; the falsification claim was scoped narrower than the evidence it pronounced on.

**Consequence:** both production models support 262144 context. On this card the KV budget makes MML 262144 mechanically trivial (~2.4× concurrency at full length at measured per-token cost) — but the 131K–262K range remains **quality-unvalidated** per the launcher's standing rationale; raising MML is a designed long-context experiment, not a flag flip.

## Session lesson (three reversals, one species)

1. KV miss reattributed: model-intrinsic cost → engine allocation policy.
2. FA4 as "where the fixes went" → FA4 never applicable to this model's geometry.
3. "MML recollection falsified" → correct, on the repo I didn't check.

All three: **claims pronounced beyond the sample that supported them, corrected by widening the sample.** The predict-and-score protocol caught each one because the numbers refused to fit.

## Decisions

- Successor checkpoint pins: **HEAD** (`52f3f65b` 31B / `1d2c2d7f` 12B) — weights are provably the June weights; canonical template is where upstream converged. Formalize when `--revision` lands in the launcher.
- Sweep JSONs renamed to encode engine version (`vllm-0230` / `vllm-0251` segments); pre-commit rename, append-only thereafter.

## Open items carried forward

`--revision` flag in start-vllm.sh (closes engine-pinned/model-floating asymmetry) · `--image` help-text drift fix · `--engine-label` for throughput_sweep.py · long-context (262144) experiment, quality-gated · quality re-validation covering engine + template in one design · NVFP4 roster question (current path confirmed INT4 Marlin, not NVFP4) · host housekeeping from session 1 (apt pin, ECC, persistence unit, CUDA toolkit) · permanent repo home for `~/blackwell-bringup-results/`
