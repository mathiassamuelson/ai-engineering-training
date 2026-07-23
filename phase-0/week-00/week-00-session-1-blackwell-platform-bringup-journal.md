# Week 00 · Session 1 — Blackwell platform bringup, model smokes, first sweep

**Date:** 2026-07-22
**Host:** inference (B650 Eagle AX, Ryzen 9600X), new enclosure, single RTX PRO 6000 Blackwell Max-Q (96 GB GDDR7, 300 W)
**Surface:** Claude chat (judgment/planning); commands pasted from terminal
**Scope:** Bring the new GPU to a serving-capable state, smoke-test both production QAT models on the frozen v0.23.0 stack, capture a first c=1 throughput anchor. Gate for selling the 3090s.

---

## Method note

Ordered gates, one at a time, prediction committed before each check. This entry records predictions and scores inline, including misses. Corrections discovered later in the session are appended where they occurred, not rewritten.

## Gate 1 — PCIe detection

**Prediction:** one NVIDIA VGA device, GB202-class, CPU-attached x16 slot; link trains Gen4 x16 (card is Gen5, board ceiling is Gen4).

**Result: PASS.** `lspci` shows `10de:2bb4` at 01:00.0 (name unresolved locally — stale pci.ids, cosmetic). `LnkCap` 32GT/s x16, `LnkSta` 16GT/s x16 "(downgraded)" — board ceiling, expected, irrelevant for single-GPU steady-state inference.

## Gate 2 — Driver

**Prediction (committed before checking):** resident driver is Ampere-era; `nvidia-smi` fails to see the card until a 575+ open-kernel-module driver is installed. Confidence: high on "driver work needed," specific mechanism asserted.

**Result: prediction partially right, mechanism wrong — twice.**

- Actual state: **no kernel module loaded at all** (`/proc/driver/nvidia/version` absent, zero NVRM dmesg lines). Not an old driver rejecting the card.
- Installed userspace was already `nvidia-driver-580-open` (580.126.09) — newer than my predicted floor, correct open flavor for GB20x.
- Root cause: **kernel/module version skew.** Running kernel 6.8.0-136; prebuilt `linux-modules-nvidia-580-open` packages stopped at 6.8.0-110. No `nvidia.ko` existed for the running kernel, so nothing loaded, silently.

**Fix path had a second layer:** installing the -136 module package hit a dependency wall — the NVIDIA CUDA apt repo (priority 600) offers 580.173.02 userspace, beating Ubuntu archive's 580.159.03 (priority 500), but ships no Ubuntu-style prebuilt signed modules; the Ubuntu module package requires `<= 580.159.03-1`. Two repos fighting; apt prefers a set no one publishes a closure for.

**Resolution:** explicit version pins to the Ubuntu archive's `580.159.03-0ubuntu0.24.04.1` across the full 580 package set + the -136 module package, one transaction. `modprobe nvidia` succeeded live (no stale module to displace).

**Open item (recurrence risk):** next `apt upgrade` re-triggers the collision. Durable fix: apt preferences file dropping the NVIDIA repo below 500 for `nvidia-driver-*` / `libnvidia-*` / `nvidia-kernel-*` patterns while leaving it authoritative for CUDA toolkit + container toolkit. Not done this session.

## Gate 3 — nvidia-smi ground truth

**Prediction:** one RTX PRO 6000 Blackwell Max-Q, ~97.8 GiB VRAM, 300 W cap, **ECC on by default** (workstation-part assumption).

**Result: 3 of 4.** Name ✓, 97,887 MiB ✓, 300 W ✓ (idle P8 at 17 W). **ECC: OFF — prediction wrong.** Unlike HBM datacenter parts, this GDDR7 workstation card ships ECC-disabled; enabling is opt-in (`nvidia-smi -e 1`, reboot, capacity/bandwidth cost). ECC on/off is now a deliberate open decision with default "off," inverted from my assumption.

Driver reports CUDA 13.0. Persistence mode enabled (`nvidia-smi -pm 1`) — later found not to survive reboot; boot-persistent fix (nvidia-persistenced) queued.

## Gate 4 — Container passthrough + image provenance

**Prediction:** container toolkit (1.18.2, predates the swap) works unmodified.

**Result: PASS.** Same driver stack visible inside the pinned vLLM image.

**Provenance wrinkle:** `docker images --digests` shows `<none>` for the pinned image — local RepoDigests metadata lost. Verified the strong way: `docker pull` reported the exact pinned digest `sha256:6d8429e38e37…22ed8f` with "Image is up to date," proving local content is byte-identical to the registry tag. Digest column quirk noted for future gate checks: the pull output is the authoritative path when `--digests` is empty.

## Gate 5 — vLLM smokes on the frozen stack (v0.23.0)

### 12B-QAT (worker preset, GPU/name overridden to single-die reality)

**Prediction:** boots clean on sm_120; W4A16 compressed-tensors path flagged at moderate confidence (quantized-kernel arch coverage historically lags core support). Fallback plan: BF16 parent.

**Result: PASS, clean sweep** via `vllm-bringup-checks.sh`. `MarlinLinearKernel for CompressedTensorsWNA16` selected explicitly — quant active on sm_120. KV pool 72.77 GiB / **2,673,610 tokens** / 20.40× concurrency at MML 131072. First taste of the new memory regime. Init 37.55 s (22.72 s compilation, first-boot). Chat round-trip 0.92 s. Placement verified empirically GPU 0 (UUID→PID join).

### 31B-QAT (orchestrator preset, --size 1 --gpus 0)

**Predictions:**
1. Boots clean, same Marlin line (high confidence — 12B proved the kernel path). ✓
2. KV bytes 68–72 GiB (moderate). **Near-miss:** actual 66.84 GiB — underweighted the CUDA-graph reservation at util 0.95.
3. KV tokens 1.3–1.9M (moderate; mechanism "per-token KV modestly larger than 12B"). **Clean miss:** actual **629,403** tokens, 4.80× concurrency.

**Miss analysis (as understood this session):** dividing the lines gives ~111 KiB/token for the 31B vs ~28 KiB/token for the 12B — I assumed 1.5–2× per-token cost, reality ~4×. Lesson recorded at the time: compute per-token KV from config (layers × KV heads × head dim), don't eyeball.

> **Appended correction (session 2, 2026-07-23):** that lesson was itself incomplete. The correct computation also requires per-layer-type window amortization and storage policy. The 111 KiB/token figure is the *engine's effective* cost (SWA-aware allocation, K/V stored separately, ~29% grouping overhead), not the model's intrinsic cost. See session 2 entry for the full derivation.

**Capacity fact (capacity, not performance — legitimate cross-platform statement):** this model serves MML 131072 at 4.80× concurrency on one die; the 3090 NVLink pair served the same MML at 1.48×. The weight-dominated/KV-starved regime is gone.

Boot log also recorded: TRITON_ATTN attention backend (significance recognized only in session 2), model weights 18.7 GiB in GPU memory, CUDA graph pool 0.66 GiB actual vs 0.75 estimated.

## First throughput anchor — c=1 sweep, v0.23.0

`throughput_sweep.py`, prompt ladder 128/512/2048, max_tokens 256, 3 iters + 1 warmup, `--parallelism tp1`, `--host-label blackwell-rtxpro6000-maxq`. **Not comparable to 3090 results** (different die, memory system, no all-reduce) — this is the Week-0 characterization anchor, nothing more.

**Decode prediction (committed from spec):** 55–75 tok/s at c=1, roughly flat across the ladder. Mechanism: bandwidth-bound decode, ~18.5 GiB weight read per token against 1.79 TB/s nominal GDDR7 at an assumed 60–75% achieved fraction. **Prefill:** "several thousand tok/s at 2048," low confidence.

**Results:**

| prompt | prefill tok/s | decode tok/s |
|-------:|--------------:|-------------:|
| 128    | ~2624         | 66.0         |
| 512    | ~3341         | 64.5         |
| 2048   | ~3118         | 62.6         |

**Score: decode PASS** (inside range; ~5% downslope from KV reads is "roughly flat"). Back-computed achieved bandwidth ~68–72% of nominal (later refined to ~71–75% using the measured 18.7 GiB weight footprint) — inside the stated assumption. **Prefill PASS** at its stated low confidence.

Result JSON: `throughput_sweep_vllm-0230_gemma-4-31B-it-qat-w4a16-ct_c1_tp1_20260722T033100Z.json` (renamed in session 2 to encode engine version; original default filename lacked it).

## Decisions

- **3090 sale gate: satisfied** (both model scales smoke-tested green). Listing prepared separately.
- ECC left off for all session measurements; on/off is a deliberate open decision.
- v0.23.0 remains the concluded program's frozen reference; the successor program's pin is a separate open decision.

## Open items carried forward

apt preferences pin (recurrence risk) · ECC decision · persistence-mode boot persistence · host CUDA 12.6 toolkit (predates sm_120; irrelevant to containerized serving) · evaluate v0.25.x as successor pin · journal home for `~/blackwell-bringup-results/`
