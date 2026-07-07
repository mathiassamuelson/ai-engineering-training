# Training Plan: AI Engineering — Building an Operator Copilot for RCA

*Version 1.0 draft — July 2026. Successor to the 16-week inference-engineering program (`inference-engineering-training`), which concluded with a validated two-tier serving architecture on self-hosted hardware. This program builds the application layer: an AI system that investigates incidents in a distributed system.*

---

## North Star

The program builds toward an **operator copilot for root-cause analysis over a distributed system**: an agentic system that reasons over architecture and per-component knowledge, then investigates incidents by reading logs and telemetry, running commands against live service components, and issuing SQL queries — concluding with a diagnosis and, where appropriate, a proposed remediation for human approval.

The copilot is the vehicle, not the destination. Every phase produces generalizable AI-engineering capability, demonstrated on the copilot. Where the AI-engineer role requires a skill the copilot does not exercise, the program covers that skill anyway, in a dedicated breadth week (see Principles).

**Self-containedness test:** every phase's acceptance criteria reference only the lab environment and public artifacts. The program takes no dependency on anything that does not exist publicly; anyone can execute it start to finish with no context beyond this repository set. 

## Program Theses & Principles

**Substrate neutrality is the program's thesis.** Frontier API models and self-hosted open-weight models are peers, composed per deployment: *use what you can, what you're allowed to use, and what's available.* The first half of 2026 demonstrated that frontier-model access can be interrupted by forces outside any vendor's or customer's control; continuity is an architectural requirement, not a preference.

- **Per-role ordered model chains.** Every role in the copilot is assigned an ordered chain, not a single model — e.g. `orchestrator: [frontier-large → local-31B]`, `workers: [frontier-small → local-12B]`. Chain-walking (health check, reroute, degraded-mode flag in output metadata) is implemented once, generically; which chain a deployment runs is configuration. Chains are invertible (local-first with frontier escalation) without code changes.
- **Three-corner evaluation.** The chain combination space is not evaluated exhaustively. Three corners carry the findings: **all-frontier** (quality ceiling), **all-local** (continuity floor / sovereign mode), **mixed default**. The frontier-vs-local delta, per capability, is a standing eval output — it is the empirical input to model-rightsizing decisions.

  The combination grid, with the three evaluated corners:

  ```
                             worker model
                         frontier        local
                      +--------------+--------------+
            frontier  | ALL-FRONTIER | MIXED        |
                      | quality      | DEFAULT      |
     orchestrator     | ceiling      | frontier     |
        model         |              | reasoning,   |
                      |              | local bulk   |
                      +--------------+--------------+
            local     | (not         | ALL-LOCAL    |
                      |  evaluated)  | continuity   |
                      |              | floor,       |
                      |              | sovereign    |
                      +--------------+--------------+
  ```

**Data-provenance stake.** Frontier models may **evaluate** finished systems (LLM-as-judge). They may not appear anywhere in the **training-data pipeline** — not as generators, not as filters, not as rankers. This is a deliberate stake against distillation-based training: frontier providers' commercial terms generally restrict using model outputs to train other models, and a public program should not depend on a favorable reading of those terms — the rule removes the question rather than answering it. The stake also excludes public datasets that are themselves frontier-model distillates (see Phase 5), and it keeps the Phase 5 experiment uncontaminated: if a fine-tune closes a gap, it did so without borrowing the frontier model's capability.

**Depth vs. breadth.** The North Star drives depth (build what the copilot needs); the role claim — a self-trained AI engineer — drives breadth (cover the map even where the copilot doesn't need it). Where they diverge, a dedicated breadth week is added rather than inflating a North-Star-scoped phase. Phase 4's "Retrieval, complete" week is the first instance.

**Repo partition principle.** Each repository holds artifacts with a single mutability profile, so each cleanliness gate protects exactly one kind of tree. The program's repo set:

| Repo | Contents | Mutability profile |
|---|---|---|
| `ai-engineering-training` (this repo) | Plan, journals, results, captures | Continuous churn on training cadence |
| `ai-training-tools` (shared with predecessor) | Benchmark/eval/judge tooling, eval rubrics & probes | Tool-release cadence |
| `rca-lab` | Lab substrate configs, **fault definitions**, **authored knowledge base**, **golden incident set** | Build-then-freeze; versioned deliberately |
| `inference-reference-stack` | Serving stack (predecessor deliverable) | Near-static; config-drift fixes only |

- **Tool/definition split:** fault-injection and eval *tooling* lives in `ai-training-tools` (multi-model, CLI-argument house style); fault *definitions*, runbooks, and incident data live in `rca-lab`, with the environment they describe.
- **Dual-SHA provenance:** results record both the tools repo commit and the `rca-lab` commit (incident-set version). The pre-session clean-tree gate covers both trees during the phases that grow `rca-lab`.

**Implementation conventions.** Tool contracts and structured outputs are implemented as Pydantic models throughout, with JSON Schema as the interchange form (MCP speaks schema) — a dedicated Pydantic study block was considered and rejected: it is the implementation vocabulary for work the program already contains, acquired in use. Structured-output enforcement is substrate-neutral: guided decoding on the local endpoints, native structured-output modes on frontier API models. Framework references in this plan mean *the dominant orchestration framework at execution time* — currently LangGraph, with LangChain 1.0 as its convenience layer (a separate LangChain track was rejected — the 1.0 consolidation made LangGraph the runtime; the convenience layer gets a session fragment in Phase 3, not a week).

**Methodology carry-over from the predecessor program:** predictions committed before measurement; one experiment at a time with explicit go/no-go gates; honest null results over forced demonstrations; daily journals never rewritten; weekly cadence with scoped sessions; confidence-provenance drafting for all derived documents (summaries trace to dailies; results-bearing artifacts use the evidential tag taxonomy).

**Program shape:** Week 0 prologue plus six phases. 18 logical weeks allocated against a 17 ±1 target — the per-phase allocation is a planning estimate, and adjustments land in the Key Changes log; logical weeks ≠ calendar weeks (the predecessor's 16 logical weeks took ~5 calendar months; a similar ratio is expected and fine).

---

## Week 0 — Prologue: Platform Revalidation & Lab Bring-Up

*Outside the six phases conceptually, documented as a first-class week (`week-00`, session journals, weekly summary — the summary is the platform-revalidation record later regression claims will cite). Stated generically: **before starting the program, bring your serving platform to a validated state and stand up the lab environment.** The hardware specifics below are this instance's, not the program's requirements — a follower runs the same prologue against whatever they have.*

**Platform revalidation**

- One convergence event, not incremental steps: hardware migration (4× RTX 3090 → 1× RTX PRO 6000 Blackwell Max-Q, 96 GB) and engine upgrade (vLLM v0.23.0 → v0.24.x) land together.
- **Platform characterization is predict-from-spec.** The predecessor's frozen baselines describe retired silicon and a retired parallelism configuration, so regression-against-history is replaced by first-principles prediction: expected numbers are committed from spec arithmetic (memory-bandwidth ratios, published references for this GPU class) before first measurement, then measured, and every miss is explained. A large unexplained miss is the misconfiguration detector the regression used to be. The measured numbers become this program's own baselines.
- `inference-reference-stack` changes are config-drift correction only (image digest, device-count and topology assumptions, dashboard panels, and retirement of the worker-pool front door — see Phase 3) — not refactoring.
- **Deployment shape:** single die, no GPU-layout question. The orchestrator (31B, FP8) and a single 12B FP8 worker run as two services on one device with pooled memory; request concurrency is the engine's job (continuous batching), not a process-count decision.
- **Model roster frozen for the program:** Gemma 4 31B/12B as the local models in every chain — the roster underpins judge calibration and every cross-phase comparison, so it holds on eval-continuity grounds regardless of what the hardware could serve. **Production precision: FP8**, quantized from the BF16 parents. (The QAT w4a16 checkpoints were reconsidered and dropped as the default: QAT artifacts cannot be fine-tuned — forcing a two-variable precision swap at the Phase 5 redeploy — while FP8 keeps the deployed stack single-variable across the program, runs native on this hardware, and forgoes only a decode-byte advantage this bandwidth doesn't need.) Week 0 items that follow: produce the FP8 quants of both parents (the same llm-compressor path Phase 5 will apply to the fine-tune — a deliberate rehearsal), and verify FP8 ≈ BF16-parent quality-equivalence with the judge harness as a committed prediction. The cached QAT checkpoints demote to named fallback if that equivalence surprises.
- **The inference substrate is otherwise a black box to this program.** The application layer consumes it as an OpenAI-compatible service; hardware-level questions — same-die co-residency interference, MIG partitioning tradeoffs (isolation vs. pooled memory), larger-orchestrator deployments up to 100B-class at 4-bit float — are recorded for a potential follow-on inference-engineering module, not explored here.
- Sequencing: the new platform passes its smoke tests and characterization before the outgoing GPUs are sold — the known-good stack stays physically assembled until its replacement is proven.

**Lab bring-up**

- Deploy the **OpenTelemetry Demo ("Astronomy Shop")** via Docker Compose as the lab substrate: ~15 microservices written in a deliberate mix of languages (Go, Java, .NET, Python, and others) with Postgres/Kafka/Redis persistence, full logs/metrics/traces wiring, bundled Prometheus/Grafana/Jaeger/OpenSearch, and a load generator.
- Acceptance test: all services green, telemetry flowing, load generator producing baseline traffic, and a couple of **built-in feature-flag faults** toggled → observed → reverted. *Built-in flags validate the lab; custom faults (Phase 1) will serve the evals.*
- **Resource coexistence is measured, not assumed:** the lab plus the full serving stack share one host; the deliverable is a headroom number under concurrent load. Fallback if tight: lab and heavy serving experiments time-share the box, procedurally.
- **SQL surface verified** (the Postgres-backed paths give the copilot something real to query); extended later only if a phase needs more.
- `rca-lab` repo populated with the substrate configs.

**Deliverable:** revalidated platform with regression results vs. frozen baselines; running lab with acceptance smokes; coexistence headroom number; `week-00` summary journal as the revalidation record.

---

## Phase 1 — Evaluation Foundations (Weeks 1–3)

*Evals precede the application. The copilot will be built against an evaluation harness that already exists — the reverse order is how AI systems accumulate unmeasured regressions.*

- **Golden incident set**: incidents produced by injecting faults into the lab, so every incident has **ground truth by construction** — the property that later makes Phase 5's verifiable reward possible. Incident schema: injected fault (cause), observable symptoms, investigation-relevant surfaces, labeled root cause.
- **Custom fault library**: authored faults beyond the demo's built-in flags — via flagd's raw JSON configuration and beyond it (container-level resource pressure, network degradation between specific services, poisoned data rows). *Contamination rationale, recorded:* every current model has the OTel Demo's documented failure scenarios in its training data; a copilot "diagnosing" a famous built-in flag may be recalling a blog post. Custom faults restore ground-truth integrity for evaluation. Definitions land in `rca-lab`; injection tooling in `ai-training-tools`.
- **Judge infrastructure, generalized**: the predecessor's position-bias-controlled pairwise/pointwise judge harness extends from quality-equivalence duty to general investigation-quality scoring: rubrics, regression gates, dataset management. The frontier-vs-local delta is produced as a standing output from the first eval run (three-corner discipline).
- **Prompt & context engineering**: system-prompt design, structured outputs (Pydantic → JSON Schema; guided decoding local / native modes frontier), tool-call contracts, context budgeting, multi-turn structure. Evaluated, not vibed: prompt variants go through the harness. One named method deliverable: **prompt-sensitivity evaluation** — a principled prompt-tier construction procedure (full production prompt stripped in defined steps down to a bare task statement) and a sensitivity metric over the resulting score curve, with prompt identity recorded in result metadata the way model identity is: a prompt is an experimental variable, not ambient configuration. Built here, reused by Phase 5's internalization measurement — and valuable on its own, since prompt-dependence is a per-model, per-capability property worth knowing for any seat in any chain.

**Deliverable:** eval harness + golden incident set v1 + fault library v1, with baseline scores at the three corners for a naive (single-prompt) investigator — the number every later phase must beat.

## Phase 2 — Knowledge Grounding (Weeks 4–5)

- **The retrieval decision, measured first** (expanded on review from a two-way to a three-way comparison — agentic navigation is a first-class contender now, not a footnote): long-context loading vs. embedding retrieval vs. **agentic navigation** (the agent reads a curated index and drills down the document hierarchy with read tools — retrieval as navigation rather than lookup), compared on cost, latency, and answer quality over the lab knowledge base. Each earns its place; the deliverable is a decision framework with numbers, not a default architecture. The three have inverted cost profiles (navigation: no retrieval infrastructure, expensive per query; embeddings: the reverse), and navigation quality is itself a reasoning capability — whether the local models navigate as reliably as frontier ones is a standing three-corner question. *(The field's context windows have made "RAG by reflex" obsolete; the copilot's heterogeneous operational-knowledge case is precisely the carve-out where retrieval in some form is expected to justify itself — expected, then verified.)*
- **Authored knowledge base**: runbooks, architecture notes, and per-component operational docs for the lab system — *authored, not pointed at public docs*, so Phase 2 can distinguish retrieval working from model recall working. Authored **for navigability** as well: a curated root index (in the `llms.txt` spirit), consistent hierarchy, one runbook per component — which costs nothing during authoring and gives the agentic-navigation contender a fair corpus to compete on. Lands in `rca-lab`.
- **Core RAG, copilot-scoped**: embeddings, chunking, indexing, retrieval evals (faithfulness, context relevance) — the subset the copilot needs. The full canon comes in Phase 4's breadth week.
- **Text-to-SQL with schema grounding**: the copilot issues real queries against the lab's databases; schema provision, query validation, and result integration into investigations.
- **Optional stretch experiment — LLM-organized corpora** *(unscheduled; same status tier as the electives)*: agentic navigation presupposes an organized corpus, and real operational document sets rarely arrive that way. The experiment: strip the authored structure from a copy of the knowledge base (fragment it, remove the index), have a model reorganize it — root index, per-document summaries, hierarchy proposal — then run the same three-way evals over the hand-authored vs. LLM-organized corpus. Measures whether navigation's precondition can itself be manufactured, and at what cost relative to human authoring — the build-time counterpart of query-time navigation, and kin to Phase 3's agent-memory theme (curation under change, on the knowledge side).

**Deliverable:** grounded investigator beating the Phase 1 baseline on the incident set, with the retrieval-vs-context framework documented.

## Phase 3 — Agents & Orchestration (Weeks 6–9)

*The predecessor built the serving architecture; this phase builds its application layer.*

- **Week 6 — The tool surface (MCP)**: the copilot consumes the lab exclusively through MCP, on the existing public `cli-mcp-server` enforcement substrate (Apache 2.0; deny-first, allow-list, no shell, sanitized env). The work is **catalog design as a product decision**: tool definitions and return shapes for the lab's CLIs and query paths, read-only tiering, default exposure policy. Program-adjacent item on that repo: audit logging of agent interactions as a contribution candidate.
- **Week 7 — Agent loops & the orchestrator**: investigation loops, the orchestrator's fan-out (the two-tier pattern finally running its intended workload), per-role model chains implemented in the first skeleton — substrate-neutral from day one. The worker tier is a **single endpoint**: fan-out means N subtask requests in flight against one continuously-batching worker service, not multiple worker instances — which also gives all three eval corners the same shape (one orchestrator endpoint, one worker endpoint). **Agent memory / investigation state** (added on capstone-internal grounds; a role-coverage review confirmed the gap): a real investigation outlives any context window — working-memory management, context compaction, what persists across turns.
- **Week 8 — Pipelined vs. agentic RCA**: the phase's flagship experiment. A hard-coded investigation pipeline and a self-planning agent, same incident set, predictions committed first — quality, cost, tool-call count, latency, variance. Tests the standing prediction that agentic degrades more than pipelined on open-weight models (three-corner comparison). Framework contact is folded in here rather than given a survey week — one deliverable, two purposes: the pipelined arm is built in **LangGraph**; a session fragment runs the LangChain `create_agent` layer against the same tools to locate what the abstraction hides.
- **Week 9 — Security of the AI layer**: a copilot reading logs is reading attacker-influenceable input while holding command and SQL access. Measured, not asserted: adversarial instructions planted in lab logs vs. the tool-surface allow-list (the enforcement should hold — demonstrate it); data minimization in context; tool-boundary enforcement under manipulation; audit logging; output filtering; data-leakage prevention. **Human-in-the-loop** (the approval gate is itself a guardrail pattern — the answer to a wrong or manipulated agent, for anything mutating): remediation is proposed as an **approval artifact** — action, reasoning, expected effect, rollback path — never executed autonomously.
- **Agent tracing** (cross-week; OTel-native chosen over a dedicated LLM-tracing product — the lab is already OTel end-to-end and the semantic conventions are the standards-durable skill; a session-level look at a Langfuse-class product covers conversancy): OpenTelemetry GenAI semantic conventions into the existing lab observability stack.

**Deliverable:** the copilot investigating end-to-end through the MCP surface, the pipelined-vs-agentic report, the security-week measurements, HITL artifacts in the output contract.

## Phase 4 — System-Level Evaluation & Hardening (Weeks 10–12)

- **Week 10 — Agent evals at system level**: failure taxonomy for investigations, incident-replay eval sets, regression discipline for agent behavior; eval-vs-cost dashboards.
- **Week 11 — Reliability & the chaos drill**: stated as an explicit checklist rather than left implied: retries, idempotency, timeouts, circuit breakers, resumable investigation state — plus a folded-in awareness session on durable-execution engines (Temporal-class), one session, not a week. **Chaos drill:** kill the frontier endpoint mid-investigation; predict, then measure, recovery behavior and the per-tier eval delta between frontier and failover modes. *The measured cost of degraded mode — "lose the API, lose X quality at Y% longer time-to-hypothesis" — is the continuity thesis compressed into one number.* Cost work: token budgets; prompt caching measured on both substrates (vLLM prefix caching vs. frontier API caching).
- **Week 12 — Retrieval, complete**: the breadth week. Hybrid sparse+dense, reranking, query rewriting, multi-hop, chunking-strategy tradeoffs, embedding-model selection, RAG eval suites — run against the lab knowledge base so it's measured, not survey-read.

**Deliverable:** hardened copilot with failover demonstrated and priced; the reliability pattern set implemented; role-complete retrieval coverage.

## Phase 5 — Post-Training (Weeks 13–16)

*Positioned late deliberately: fine-tuning without evals is unmeasurable, and "when not to fine-tune" is itself a deliverable — with the null result pre-defined as a finding, not a failure.*

**Honest framing, stated up front:** post-training does not make an open-weight model generally equal to a frontier model. What it demonstrably can do is close the gap **on a narrow, well-specified task distribution**. The program's question is therefore not "can local match frontier?" but "how much of the *worker-task* delta does each post-training stage recover, at what cost?"

**Capability track (the real experiment)**

- Target: the **12B worker tier only**. The reason is methodological, not just pragmatic: post-training needs a reward signal, and the worker tier is where one exists for free. Every incident in the golden set exists because a known fault was injected, so a worker subtask's outcome is machine-checkable — did this investigation trace identify the planted fault, in how many tool calls, with contract-conforming output? (The Phase 1 contract checker is, in effect, a reward function that already exists.) No judge model, no human preference labels. Orchestration quality has no equivalent signal — whether a decomposition was *good*, or a synthesis faithful to worker findings, has no machine-checkable answer without exactly the curation burden this design avoids — so there is no orchestrator fine-tune.
- Training data is generated in the **all-local corner** and filtered by that reward: the local 31B orchestrates, the 12B workers produce many candidate investigation traces per incident, and the traces that actually found the planted fault become training examples — rejection sampling with ground truth as the filter — supplemented by a small hand-curated gold set. The corner choice is deliberate: if a frontier model wrote the subtask assignments the traces begin with, frontier output would have entered the training pipeline, violating the provenance stake. Run all-local, the question of distillation permissibility never arises at all. (A structural echo worth noticing: the sovereign configuration is also the self-improvement configuration.)
- Arc: trace generation & curation → **12B LoRA SFT** (BF16 parent) → preference/RL stage against the same verifiable reward → quantize or serve BF16 → full eval → redeploy into the lab stack.
- Precision rules: training operates on the BF16 parents. Adapters are served only over the base they were trained against. Deployment path: merge the adapter into the BF16 parent and quantize post-hoc to **FP8** — the same path that produced the production base models in Week 0, so the quantization tax appears identically on both sides of every before/after comparison and cancels out. 8-bit PTQ is expected to be near-lossless, and that expectation stays a committed prediction: the eval scores the fine-tune at both precisions, fine-tuned-BF16 vs. fine-tuned-FP8, tax expected ≈ 0. The capability question — against the un-fine-tuned production FP8 model driven by prompting alone, and against the frontier model in the worker chain — is read from the untaxed BF16 column. BF16 serving remains the fallback and fits trivially in 96 GB of pooled memory.
- **The bet**: the capability track's success criterion is a three-way comparison on the worker task distribution (the golden incident set's worker subtasks). Column one: the fine-tuned 12B. Column two: the frontier model occupying the same position in the worker chain — a Haiku-class model, deliberately, because that is the seat the fine-tune would take over; beating a frontier flagship is not the claim and never was. Column three: the prompted, un-fine-tuned production 12B, which prices what good prompting achieves for free — the fine-tune must beat this column merely to justify existing. Every outcome is a finding. If the fine-tune wins column two, the worker chain inverts to local-first: the sovereign configuration stops costing quality on worker tasks, and the model-rightsizing question has a measured answer instead of an opinion. If it loses to column two but beats column three, the frontier model keeps the preferred seat — but the fine-tune takes the local one: assuming the margin exceeds noise, the fine-tuned 12B replaces the prompted base as the worker chain's local model, raising the continuity floor even though the chain doesn't invert. The sovereign configuration improves either way the top of the chain resolves. And if it fails to beat even column three, fine-tuning added nothing over good prompting **at this training-set scale and curation effort** — a scoped data point in favor of the fine-tune-as-last-resort position, not a general verdict on fine-tuning: a production effort investing far more in dataset size, coverage, and curation could land differently, and distinguishing "fine-tuning didn't pay here" from "fine-tuning doesn't pay" is itself part of the when-not-to-fine-tune deliverable. The experiment cannot lose; it can only inform.
- **Prompt-sensitivity dimension:** fine-tuning relocates the task specification from context into weights — a detailed system prompt is a spec paid on every request (prefill tokens, latency) and maintained as model-tuned prose; SFT pays it once, at training time. To measure how completely the spec moved, a 2×2 runs alongside the bet, using the prompt-sensitivity method built in Phase 1: {fine-tune, prompted base} × {full production prompt, deliberately minimal prompt}. Committed prediction: the fine-tune's score barely moves across prompts while the base's collapses — the gap between those gaps is the internalization measure. The bet's headline columns stay like-for-like (same prompt for all three models); this is a separate operability finding: a fine-tune that runs on a minimal prompt cuts per-request prefill cost (measured), retires a maintenance-heavy prompt artifact, and leaves behind a prompt generic enough to survive model swaps.

**Mechanics track (breadth, nothing deployed)**

- One session-cluster: **31B QLoRA** run locally, and **31B BF16 LoRA under FSDP** run on a **rented multi-GPU node — by design, not as a compromise**. Sharding a model across GPUs for a half-day, a few times a year, is the textbook rental workload: *own for duty cycle, rent for peaks* — a principle that generalizes well beyond this session and quietly explains why the local platform doesn't need to fit every conceivable future workload. The rented node is also the skill's natural habitat — nobody shards on a workstation in production — and brings the incidental competencies along (standing up training on ephemeral hardware, moving checkpoints off a box you don't keep). Neither technique is needed by the capability track — a 12B LoRA fits locally with room to spare — but role completeness wants hands-on contact with the memory-constrained end of the training toolset. Deliverables are mechanics-shaped only: memory footprint, sharding behavior, throughput, checkpointing notes. Explicitly not model quality, and nothing gets deployed.
- Dataset: a ready-made, **clean-provenance** public dataset — human-authored or derived from permissively-licensed sources; many popular public instruction sets are frontier-model distillates and are excluded by the provenance stake. A known-good dataset also removes data quality as a confound: if the loss curve misbehaves, the training setup is at fault, which is exactly the signal a mechanics session wants. Coding is the preferred domain (provenance-clean options exist, and off-the-shelf benchmarks give a one-number before/after sanity check). Specific selection is a session-open task.

**Deliverable:** the capability-track eval table (fine-tune vs. frontier model vs. prompted base, both precisions), the redeployed worker if it wins, the mechanics measurements, and the when-not-to-fine-tune conclusion — whichever way the data points.

## Phase 6 — Capstone (Weeks 17–18)

- **Lab copilot v1**: end-to-end investigations of custom-fault incidents it has never seen, evaluated at the three chain corners, producing diagnoses and HITL remediation artifacts, fully reproducible from the public repos.
- The capstone is **terminal in public**: complete in its own right, never designed for portability elsewhere. Capstone report and program retrospective follow the predecessor's Week-16 pattern (summaries from journals, then the report), under the confidence-provenance contract with the evidential taxonomy.

---

## Named Non-Goals

Recorded so their absence reads as a decision, not an oversight:

- **High-volume telemetry analytics** (aggregation/batching/clustering at production scale) — a data-engineering-shaped problem the lab cannot faithfully reproduce; at most one elective session on batch/cluster-then-analyze patterns over synthetic telemetry.
- **Cloud managed-services deployment** (Bedrock-class hosting, serverless, IaC) — this program is deliberately self-hosted; containers, observability, API design, and provider abstraction transfer, the managed-service specifics don't. The self-hosting depth is the compensating differentiator.
- **Frontend/UX engineering** — the copilot gets a minimal interface at most.
- **Enterprise business-systems landscape** (CRM/marketing/support/iPaaS integration catalogs) — landscape familiarity, not lab-teachable engineering; the MCP-first tool surface is the *pattern* answer to what that requirement probes.
- **Full serving-stack HA** — failover is built at the model-endpoint level; infrastructure HA was the predecessor's territory.
- **Own-QAT during fine-tuning**; **orchestrator fine-tuning**; **inference-substrate optimization** (co-residency interference, MIG partitioning, larger-orchestrator deployments — deferred to the follow-on inference module per Week 0's black-box framing).

---

## Key Changes Log

*Maintained from day one. A row is warranted when a change alters what a week does — scope, sequence, deliverables, principles, or the week count. Editorial polish is not logged (git history covers it). Baseline: this v1.0 plan; each row's "Original" is the prior state at the time of the change.*

| Date | Original | Revised | Reason |
|---|---|---|---|
| 2026-07-06 | Week 0: 2× RTX A6000 (48 GB, NVLink), GPU Layout A (orchestrator TP=2 + one worker per card), regression vs. the predecessor's frozen baselines. Phase 3: two-worker pool behind a load-balancing front door. Phase 5: w4a16-first deployment with a BF16 single-occupancy escape hatch; FSDP on owned hardware. | Hardware finalized as 1× RTX PRO 6000 Blackwell Max-Q (96 GB, single die). Week 0: layout question dissolved; predict-from-spec platform characterization replaces regression-vs-history; inference substrate framed as a black box to this program. Phase 3: single worker endpoint (concurrency via continuous batching; worker-pool front door retired as IRS config drift). Roster: production precision moves from QAT w4a16 to FP8 quantized from the BF16 parents (QAT checkpoints demoted to fallback). Phase 5: FP8-first deployment via the same PTQ path as the base models (BF16 fallback); escape-hatch constraint dissolved; FSDP session on a rented multi-GPU node ("own for duty cycle, rent for peaks"). | Hardware decision finalized after the v1.0 commit. The program is a fresh start, so baseline parity carried no value; the two-worker pool was a topology artifact of isolated 24 GB cards, not an architectural preference; and the single-die card serves the post-program self-hosting horizon (100B-class at 4-bit float). The Gemma 4 roster freeze survives on eval-continuity grounds; its precision changes because QAT artifacts cannot be fine-tuned, and an FP8 default keeps the Phase 5 redeploy single-variable. |

---

*Program status: defined; start gated on the predecessor program's Week 16 conclusion.*
*Repos: `ai-engineering-training` (this), `ai-training-tools` (shared), `rca-lab` (lab), `inference-reference-stack` (serving).*
*Local hardware at start: 1× RTX PRO 6000 Blackwell Max-Q (96 GB GDDR7), single-host deployment; frontier models via API.*
