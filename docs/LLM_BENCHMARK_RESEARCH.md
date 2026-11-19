# LLM Configuration Feasibility and Benchmark Evidence

**Status:** Research proposal / documentation only  
**Research snapshot:** 2026-11-19  
**Purpose:** Explain the memory-feasibility mathematics, evidence rules, and candidate-model research needed before Brute Genius can produce defensible LLM configuration rankings.

This document does not claim that Brute Genius has benchmarked GPUs, executed remote workloads, or established a winning model. The current Brute Genius README describes sequential, in-process host scoring, a mock provider, and remote execution as future work. Its current numerical objectives are mathematical test functions, not measures of language-model quality.

## 1. Scope and terminology

A benchmark candidate is one complete configuration:

`(model, artifact, quantization, context, KV precision, runtime, backend, host)`

A measured outcome adds a workload and a specific execution:

`(configuration, host record, workload, workload specification, test-set revision, run)`

Keep these concepts separate:

- **Model:** immutable upstream model identity and revision.
- **Artifact:** exact downloadable file(s), format, size, and SHA-256.
- **Configuration:** inference settings that can be reused across hosts.
- **Host record:** hardware and software environment snapshot.
- **Run:** one execution of a configuration for a workload.
- **Workload result:** independent outcome for coding, reasoning, chat, or document ingestion.

A failed context length or workload does not invalidate every other configuration or workload for the same model.

## 2. Memory-feasibility model

The estimator is a pruning tool. It is not a substitute for observing actual allocations and process memory during execution.

### 2.1 Weight storage estimate

A first-order estimate is:

`W_est = P × (b / 8) × (1 + m)`

Where:

- `P` is the number of stored parameters.
- `b` is an assumed effective bits-per-weight value.
- `m` is an overhead fraction for quantization metadata, scales, tensor alignment, padding, and other format-specific storage.

This is only a rough estimate. Named formats such as `Q4_K_M` are mixed/block quantization formats; the label must not be interpreted as exactly four bits for every stored parameter. Embeddings, output tensors, selected higher-precision tensors, metadata, and alignment can change the result.

**Evidence precedence for weight size:**

1. Actual downloaded artifact byte size, recorded from the file.
2. Publisher or artifact-host file metadata, labeled as a vendor/host claim until downloaded.
3. Tensor/format-aware estimate, labeled `ESTIMATED`.
4. Parameter-count × nominal-bits estimate, labeled `ESTIMATED` with its assumptions recorded.

Do not replace an observed artifact size with the simplified estimate. For split or sharded models, record the sum of all required artifact files and their individual hashes.

### 2.2 KV-cache estimate

For a conventional transformer with uniform full attention, the basic estimate is:

`KV_est = B × T × Σ_l [H_kv,l × D_k,l × S_k,l + H_kv,l × D_v,l × S_v,l]`

For the common symmetric case where K and V have the same dimensions and element size:

`KV_est = 2 × B × T × L × H_kv × D_head × kv_bytes`

Where:

- `B` is the number of simultaneously resident sequences (not necessarily the prompt batch token count).
- `T` is the allocated/target context length per sequence.
- `L` is the number of layers contributing to the cache.
- `H_kv` is the number of key/value heads, **not** the query/attention-head count.
- `D_head` is the per-head dimension.
- `kv_bytes` is bytes per cached element.

Use separate K and V sizes when their precision differs. Use per-layer sums or an architecture-specific estimator for grouped-query attention, multi-query attention, sliding-window attention, hybrid/recurrent architectures, shared caches, cache quantization, or other non-uniform designs. Do not apply the simplified formula blindly to every architecture.

The selected context length and parallel sequence count matter. Record both. A model's advertised maximum context is not proof that a particular host can allocate that context.

### 2.3 Compute and activation buffers

`C_est = max(backend_floor, context_tokens × compute_coefficient)`

This can be useful as a coarse placeholder, but both terms must be backend-, architecture-, batch-, and configuration-specific. Real buffers can depend on batch size, microbatch size, attention implementation, graph planning, flash attention, tensor placement, and temporary workspaces.

Therefore:

- Use `C_est` only for early feasibility pruning.
- Store the estimator version and assumptions.
- Replace estimates with measured backend allocation data where available.
- Do not label a candidate resident-fit solely because a generic coefficient predicts it will fit.

### 2.4 Reserved memory and usable budget

Let:

`R = OS_reserve + display_reserve + backend_context_reserve + other_required_reserve`

Then:

`B_usable = B_physical_or_assigned - R`

Avoid counting the reserve twice. The equivalent resident test is:

`W + KV + C ≤ B_usable`

or:

`W + KV + C + R ≤ B_physical_or_assigned`

The reserve is not available to the model. Available memory at benchmark time may be lower than installed memory because other processes, the operating system, and runtime services consume resources.

### 2.5 Host topology changes the budget calculation

**Unified memory (for example, integrated/shared-memory systems):**

- Treat CPU and accelerator allocations as consumers of one shared physical-memory pool.
- Reserve memory for the OS, display, and other applications before assigning a model budget.
- Do not add nominal GPU memory to system RAM as if they were independent pools.

**Discrete GPU plus system RAM:**

- Maintain separate system-RAM and VRAM budgets.
- Account for weights, KV cache, compute buffers, and runtime allocations according to their actual placement.
- For partial offload, calculate the CPU-resident and GPU-resident portions separately, plus transfer/staging buffers.
- Do not assume all model weights or all KV cache are placed on the GPU.

**CPU-only:**

- Use the available system-RAM budget.
- Include the runtime's mapped/resident behavior and storage throughput in the candidate classification.

**Unknown host topology or missing measurements:**

- Set `topology: "unresolved"`.
- Set `config_state: "HOST_UNRESOLVED"`.
- Do not infer a machine from a generic hardware tier.

### 2.6 Feasibility states

Use the following as estimator states, not benchmark outcomes:

- `RESIDENT_FIT_ESTIMATED`: estimated weights + KV cache + compute buffers fit the applicable resident budget.
- `MAPPED_CANDIDATE`: the runtime supports memory mapping and the non-weight working set appears to fit, but execution has not proven sustained usability.
- `MEMORY_PRUNED`: estimated required allocations exceed the applicable budget under the selected placement.
- `HOST_UNRESOLVED`: required host/topology inputs are absent.
- `PLACEMENT_UNRESOLVED`: host is known but the runtime's actual allocation plan is not.

Memory mapping is not free memory. A mapped file can fault pages into RAM as inference accesses them. Under pressure, page faults, paging, storage throughput, and cache churn can make a theoretically loadable model unusably slow. The exact runtime loading mode and whether the backend supports mapping must be recorded. A mapped candidate becomes a measured fit only after successful execution and sustained-load measurements.

### 2.7 Storage gate

Before download:

`required_download_bytes ≤ free_storage_bytes - storage_reserve_bytes`

Include all shards, temporary download files, conversion/quantization output, and any retained source artifact. A resumable download may temporarily require both partial and final files. After download, verify exact file size and cryptographic hash before use.

## 3. Configuration identity and benchmark records

Use a stable configuration identifier derived from canonicalized configuration inputs. Keep the host and workload out of the reusable configuration identity; they belong to the execution identity.

Suggested identity split:

- `config_id`: immutable model/artifact identity + quantization + context + KV precision + runtime/backend + relevant flags and batch settings.
- `host_record_id`: immutable host snapshot identifier.
- `run_id`: unique execution identifier for a config on a host for a workload/test-set revision.

A hash-derived ID is useful only if canonical serialization is specified (field ordering, number representation, null handling, schema version, and string normalization).

### 3.1 Configuration record

```json
{
  "record_type": "config",
  "schema_version": "1",
  "config_id": "sha256-derived-stable-id",
  "model_id": "immutable-model-id",
  "model_revision": "immutable-upstream-revision",
  "artifact_sha256": null,
  "artifact_size_bytes": null,
  "quantization": "Q4_K_M",
  "context_tokens": 8192,
  "batch_size": 1,
  "kv_precision": "f16",
  "runtime": "llama.cpp",
  "runtime_version": null,
  "runtime_backend": "cpu",
  "runtime_flags": [],
  "host_record_id": "benchmark-host-id",
  "license_id": null,
  "license_evidence": "UNVERIFIED",
  "config_state": "HOST_UNRESOLVED",
  "failure_reason": "host inputs absent",
  "timestamp": null
}
```

### 3.2 Host record

The host record should include at least OS/version, CPU model and core count, total and available RAM, GPU/accelerator model, total and available VRAM or unified memory, topology, storage free bytes and device type, driver/runtime versions, power mode, and relevant thermal conditions. Record collection method and timestamp. Unknown values remain null.

### 3.3 Benchmark result record

```json
{
  "record_type": "benchmark_result",
  "schema_version": "1",
  "run_id": "unique-run-id",
  "config_id": "sha256-derived-stable-id",
  "host_record_id": "benchmark-host-id",
  "workload": "coding",
  "workload_spec_version": null,
  "test_set_revision": null,
  "threshold_profile_id": null,
  "threshold_revision": null,
  "config_state": "HOST_UNRESOLVED",
  "workload_result": null,
  "failure_reason": "not run",
  "metrics": {
    "load_success": null,
    "placement": null,
    "ttft_ms": null,
    "tokens_per_second": null,
    "peak_rss_mb": null,
    "peak_vram_mb": null,
    "sustained_load_seconds": null,
    "thermal_throttle_observed": null,
    "energy_wh": null
  },
  "evidence_class": "UNVERIFIED",
  "raw_log_sha256": null,
  "timestamp": null
}
```

Null means unknown or not measured. It must never be silently converted to zero, a pass, or a failure. Record prompt tokens, output tokens, sequence count, concurrency, warm-up policy, repetitions, and measurement method alongside the metrics or in a linked run-detail record.

### 3.4 Specification-claim record

```json
{
  "record_type": "spec_claim",
  "model_id": "immutable-model-id",
  "field": "declared_size",
  "value": "VENDOR_CLAIM_VALUE",
  "source": "https://example.invalid/model-card",
  "retrieved_at": null,
  "evidence_class": "VENDOR_CLAIM"
}
```

The example URL is deliberately illustrative and must be replaced with the actual source. Keep the retrieval timestamp and source revision when available.

## 4. Evidence classes

Evidence classes describe the origin/strength of a fact; they are not interchangeable benchmark outcomes.

| Class | Meaning | Example |
|---|---|---|
| `UNVERIFIED` | No supporting check has been recorded | Unknown artifact hash |
| `VENDOR_CLAIM` | Publisher/provider statement, not independently measured | Published model size or API price |
| `ESTIMATED` | Reproducible calculation from declared inputs | Predicted KV-cache size |
| `VERIFIED` | An integrity/provenance check was completed | Downloaded artifact matches expected SHA-256 |
| `MEASURED` | Actual execution with host, test, and method recorded | Peak RSS and tokens/sec |

A verified hash proves identity relative to the expected hash; it does not prove model quality, safety, or license rights. A measured run does not independently establish that a model's upstream license permits the intended use.

## 5. Gates and state transitions

For each configuration:

1. **License and intended-use gate:** identify upstream model license, artifact/quantizer license, additional terms, attribution requirements, and redistribution conditions. Unknown or incompatible terms remain unresolved or are pruned according to the project's policy; do not infer rights from the model name.
2. **Runtime/format gate:** verify that the selected runtime and backend support the artifact format and architecture at the required revision.
3. **Host topology gate:** determine unified, discrete, or CPU-only memory paths. Missing inputs produce `HOST_UNRESOLVED`.
4. **Storage gate:** prune before download if required space cannot be reserved.
5. **Memory gate:** estimate weight, KV, compute, and reserve requirements for each quantization/context/batch combination. Prune only the infeasible configuration.
6. **Placement gate:** classify full accelerator placement, partial CPU/GPU offload, or CPU execution; record unresolved placement where allocation cannot be predicted.
7. **Artifact gate:** fetch, verify size/hash/provenance, and quarantine mismatches. Retry interrupted/incomplete transfers separately. Never silently bless a changed artifact.
8. **Privacy/network gate:** prune configurations that violate offline-only or data-handling requirements. Free-tier and paid-tier data terms must be checked for the specific provider and product.
9. **Threshold registration:** freeze workload specifications, test-set revisions, and threshold profiles before measurements begin.
10. **Execution:** run each workload independently and retain all results, including failures.

Possible workload results include `PENDING`, `LOAD_FAILED`, `ACCURACY_FAILED`, `PERFORMANCE_FAILED`, `SUSTAINED_LOAD_FAILED`, `ELIGIBLE`, and `INCONCLUSIVE`. A workload is `ELIGIBLE` only when it actually ran and passed every pre-registered hard threshold for that workload.

A configuration may be eligible for coding and ineligible for document ingestion. Do not collapse independent workload outcomes into a single pass/fail bit.

## 6. Benchmark protocol

Pre-register thresholds per workload and host tier. At minimum, define:

- Accuracy/correctness metric and minimum threshold.
- TTFT maximum and throughput minimum.
- Peak RSS and VRAM ceilings.
- Required context length and context-stability test.
- Sustained-load duration and failure tolerance.
- Mobile thermal-throttle and energy/battery limits where applicable.
- Repetition count, warm-up policy, prompt/output token counts, concurrency, and power mode.

Recommended workload families:

- **Coding:** fixed repository tasks, tests, patch correctness, and reproducible grading.
- **Reasoning:** fixed problems with objective answer checks and rubric-scored explanations where needed.
- **Chat:** fixed prompt suite, instruction adherence, and human/rubric evaluation.
- **Document ingestion:** a separate pipeline test covering PDF extraction, OCR when needed, chunking, retrieval, source attribution, citation fidelity, and long-document coverage.

Document ingestion must not be merged into generic chat scoring. It has additional failure modes and should have its own test set and thresholds.

## 7. Online research snapshot: candidate discovery, not a ranking

The following sources were checked on 2026-11-19. They are discovery inputs and published claims, not independent model-quality measurements. Prices and model availability change; refresh them before deployment.

### 7.1 Local/open-weight candidates

**Gemma 4 (Google):** Google's model overview describes E2B/E4B small variants, 12B, 26B A4B MoE, and 31B dense models. Its published memory table gives approximate weight-loading footprints by precision, and explicitly warns that those figures exclude context-window/KV-cache and supporting software. It also explains that the 26B MoE still needs the full model weights resident even though fewer parameters are active per token. This is useful evidence for why parameter count or active parameters alone cannot determine host fit. Mobile variants use LiteRT-LM. The Gemma terms must be reviewed separately from permissive OSI-style licenses; do not label it Apache-2.0.

**Qwen3.8-27B:** The official Hugging Face model card identifies the model as Apache-2.0 and lists compatibility with Transformers, vLLM, and SGLang. Its architecture includes both gated-delta and gated-attention layers, with four KV heads for the gated-attention blocks. That mixed architecture is a concrete reason to avoid applying a single generic full-attention KV formula without architecture-aware accounting. The official model card's maximum context claim is not a promise that a given host can serve that context.

Neither candidate is a winner until tested on the target host and fixed workload suite.

### 7.2 Runtime and memory research

The llama.cpp documentation identifies GGUF as its supported model format and describes conversion/quantization workflows. Its current CLI/server documentation exposes selectable K/V cache types, memory-mapping/load modes, device placement, GPU-layer offload, and parallel sequence controls. These runtime settings materially affect memory placement and must be part of configuration identity.

The llama.cpp documentation also explains that mmap maps model files and can reduce eager loading, but it does not eliminate page faults or memory pressure. Mapped candidates therefore require real sustained-load tests and should not be marked resident-fit based on file size alone.

### 7.3 Hosted API candidates and published pricing

The Together AI pricing page, retrieved on 2026-11-19, listed these standard per-million-token prices at the time of research:

| Hosted model | Input / 1M tokens | Output / 1M tokens |
|---|---:|---:|
| Qwen3.8 Flash | $0.15 | $0.47 |
| GLM-5.3-Flash | $0.15 | $0.50 |
| DeepSeek V4.1 Flash | $0.30 | $1.20 |
| gpt-oss-120B | $0.15 | $0.60 |

These are vendor-published price claims, not cost-per-task measurements. Actual task cost depends on input/output token counts, caching, resolution or other settings, provider routing, and price changes. Calculate cost per successfully completed task from actual usage records. Do not assume that a hosted model is open-weight merely because a provider serves it.

For privacy-sensitive workloads, evaluate the exact provider product's data retention and training terms. Do not send confidential prompts to a free tier unless the governing terms and project policy explicitly permit it.

### 7.4 Source register

- Google, Gemma 4 model overview and inference memory table: https://ai.google.dev/gemma/docs/core
- Qwen, official Qwen3.8-27B model card and architecture: https://huggingface.co/Qwen/Qwen3.8-27B
- llama.cpp, model formats and quantization: https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md
- llama.cpp, server options including KV-cache precision and load modes: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp, completion options including mmap, mlock, device selection, and GPU-layer offload: https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md
- Together AI, current serverless inference pricing: https://www.together.ai/pricing

## 8. Brute Genius integration boundary

The current repository describes Brute Genius as a Cartesian-product enumerator with sequential host-side scoring. The connected provider is a mock; remote GPU execution is not implemented. Its current objective registry contains numerical functions such as sphere, Rastrigin, Rosenbrock, linear sum, and product. Those functions are useful for validating enumeration and ranking mechanics, but they do not measure LLM accuracy, inference speed, memory, privacy, or cost.

The appropriate future integration is:

1. Candidate/configuration enumerator.
2. Deterministic feasibility-pruning stage.
3. Artifact acquisition and integrity/provenance stage.
4. Runtime execution-provider interface.
5. Workload-specific benchmark runner.
6. Threshold evaluator using pre-registered profiles.
7. Append-only evidence ledger.
8. Multi-objective reporting that preserves hard-gate failures and per-workload outcomes.

Until a real LLM execution backend and evaluation objective exist, Brute Genius output must be described as a search plan or candidate enumeration—not as GPU benchmarking or a model leaderboard.

## 9. Acceptance criteria for a future implementation

- [ ] Canonical, versioned configuration identity.
- [ ] Separate host and run identities.
- [ ] Architecture-aware KV-cache estimator with explicit assumptions.
- [ ] Separate unified-memory, discrete-GPU, and CPU-only budget paths.
- [ ] Per-configuration pruning; no blanket model rejection.
- [ ] Mapped execution is a candidate state until measured.
- [ ] Artifact size, SHA-256, provenance, and license evidence recorded.
- [ ] Interrupted downloads distinguished from integrity failures.
- [ ] Threshold profiles and test-set revisions frozen before execution.
- [ ] Per-workload results retained independently.
- [ ] Missing metrics remain null.
- [ ] Raw logs or content-addressed evidence retained.
- [ ] No result marked `MEASURED` without an actual recorded run.
- [ ] No leaderboard generated from vendor claims or estimator outputs alone.

**Conclusion:** use the mathematics to prune candidates, vendor documentation to establish declared capabilities and constraints, and reproducible execution to establish what actually works. Keep those evidence classes separate throughout the system.


## 10. Explicit configuration lifecycle state machine

This lifecycle is the normative gate sequence for a future implementation. It distinguishes pre-download estimates from post-download verification and measured workload eligibility.

```text
CONFIGURATION CREATED
   |
   +-- LICENSE           --> LICENSE_PRUNED
   +-- COMPATIBILITY     --> COMPATIBILITY_PRUNED
   +-- PRIVACY/NETWORK   --> PRIVACY_PRUNED
   |
HOST IDENTIFICATION
   +-- unknown           --> HOST_UNRESOLVED
   |
STORAGE + MEMORY (estimate)
   +-- impossible        --> MEMORY_PRUNED
   +-- estimate ok       --> RESIDENT_ESTIMATE | MAPPED_ESTIMATE
   |
DOWNLOAD
   +-- fetch failed      --> FETCH_FAILED
   |
HASH / PROVENANCE
   +-- mismatch          --> INTEGRITY_QUARANTINED
   +-- no published hash --> INTEGRITY_UNVERIFIABLE
   +-- verified          --> re-run memory math with measured size
                              +-- still fits --> READY_FOR_BENCHMARK
                              +-- now exceeds --> MEMORY_PRUNED
   |
THRESHOLD PROFILE PRESENT
   +-- absent            --> UNRESOLVED
   |
FAN OUT: coding | reasoning | chat | doc-ingest
   per workload:
     LOAD_FAILED | HARNESS_FAILED | ACCURACY_FAILED
     | PERFORMANCE_FAILED | SUSTAINED_LOAD_FAILED
     | ELIGIBLE (requires LOCAL_MEASURED + thresholds + metrics)
```

### 10.1 Gate semantics

- **License, compatibility, and privacy/network** are independent hard gates. Their prune states apply to the affected configuration and intended use, not automatically to every quantization or deployment of the model.
- **Host identification** must complete before hardware feasibility can be decided. Missing OS, memory, accelerator topology, or required storage inputs produce `HOST_UNRESOLVED`; do not substitute a representative tier.
- **Storage and memory before download** are estimates. Record the estimator version, inputs, assumptions, and evidence class. The result is `RESIDENT_ESTIMATE` or `MAPPED_ESTIMATE`, not a measured fit.
- **Fetch failure** is distinct from integrity failure. Network errors, timeouts, interrupted transfers, and unavailable sources produce `FETCH_FAILED`; retry policy is separate.
- **Hash/provenance mismatch** produces `INTEGRITY_QUARANTINED`. Do not load or benchmark the artifact. Preserve the artifact identity, expected hash/source, observed hash, and quarantine reason.
- **No published expected hash** produces `INTEGRITY_UNVERIFIABLE`, not a hash mismatch. The downloaded bytes can still be hashed locally for identity and repeatability, but a locally computed hash alone does not prove publisher provenance. Project policy must decide whether unverifiable artifacts are blocked or allowed in a clearly marked lower-trust research lane.
- **Successful integrity verification** triggers a second memory-feasibility calculation using the actual downloaded artifact size and verified format metadata. If the actual size changes the feasibility decision, the configuration is pruned before benchmark execution.
- **Threshold profile** must exist and be revision-pinned before any benchmark begins. Missing profile means `UNRESOLVED`, not a default threshold or implicit pass.
- **Workload fan-out** creates independent results for coding, reasoning, chat, and document ingestion. A failure in one workload does not overwrite another workload's outcome.

### 10.2 Workload eligibility requirements

`ELIGIBLE` is a per-workload result and requires all of the following:

1. The exact artifact and configuration passed the required integrity policy.
2. The benchmark ran locally on the identified host using the recorded runtime/version/backend/flags.
3. The workload harness completed successfully; otherwise report `HARNESS_FAILED`.
4. The pre-registered workload and test-set revisions were used.
5. Required accuracy, latency, throughput, memory, and sustained-load thresholds were met.
6. Required metrics are present and valid; missing values remain null and cannot satisfy a threshold.
7. The result is classified `LOCAL_MEASURED` and linked to raw logs or a content-addressed evidence record.

Hosted API measurements must use a distinct evidence class and result path. They cannot satisfy a requirement explicitly marked `LOCAL_MEASURED`. A hosted result may be useful for online comparisons, but must not be represented as a local execution.

### 10.3 Suggested terminal and intermediate states

Configuration-level states:

- `LICENSE_PRUNED`
- `COMPATIBILITY_PRUNED`
- `PRIVACY_PRUNED`
- `HOST_UNRESOLVED`
- `MEMORY_PRUNED`
- `RESIDENT_ESTIMATE`
- `MAPPED_ESTIMATE`
- `FETCH_FAILED`
- `INTEGRITY_QUARANTINED`
- `INTEGRITY_UNVERIFIABLE`
- `READY_FOR_BENCHMARK`
- `UNRESOLVED`

Per-workload states:

- `PENDING`
- `LOAD_FAILED`
- `HARNESS_FAILED`
- `ACCURACY_FAILED`
- `PERFORMANCE_FAILED`
- `SUSTAINED_LOAD_FAILED`
- `ELIGIBLE`

Do not encode all workload outcomes into a single configuration terminal state. Preserve the configuration's gate history and the independent workload-result records.

### 10.4 Evidence-class requirement

Use `LOCAL_MEASURED` only when the run was actually executed on the identified local host and the measurement record is complete. Keep it distinct from generic `MEASURED` so that a cloud provider's benchmark cannot be mistaken for a local result. `VENDOR_CLAIM`, `ESTIMATED`, `VERIFIED`, and `UNVERIFIED` remain separate evidence classes and cannot independently produce `ELIGIBLE`.
