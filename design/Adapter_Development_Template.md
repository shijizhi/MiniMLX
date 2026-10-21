# MiniMLX adapter implementation template

Use this template for a new task, model-family, backend, or component provider.
An adapter should implement only the role-specific behavior it owns. A single
class may satisfy several roles, but discovery, capability declarations,
loading, and execution should remain independently testable.

## 1. Provider identity and scope

- Provider name:
- Role(s): task / family / backend / component / kernel
- Supported task(s):
- Input and output modalities:
- Model family / architecture evidence:
- Checkpoint formats and quantization schemes:
- Required and optional artifact components:
- Backend and version constraints:
- Hardware/features/memory constraints:
- Optional accelerations and compatibility requirements:
- Explicitly unsupported combinations:

## 2. Discovery contract

List each fact the provider needs, its source, and its confidence. Discovery
must not load model weights or import a device backend.

| Fact | Source | Required? | Missing/conflicting behavior |
|---|---|---:|---|
| | | | |

Keep raw/unknown metadata available in `ModelDescriptor.metadata`; do not infer
support from a repository name or a single config field when artifact evidence
can be checked.

Optional drafter/sidecar compatibility must be declared as a resolver requirement:
identify accepted artifact model types/architectures, required artifact roles,
and any target-to-artifact metadata fields that must match. A complete directory
is not proof that a sidecar can be paired with the selected target.

## 3. Declarative registration

Add an `AdapterRegistration` describing the provider's roles and constraints.
Use explicit priorities only when two valid providers are otherwise equivalent.
The resolver must return a structured `SupportDecision` and an `ExecutionPlan`
or a precise unsupported, ambiguous, missing-component, dependency, or hardware
status. It must not silently choose an incompatible fallback.

The resolved plan is authoritative at load and execution time. Do not re-read
environment variables inside the adapter to select a backend, acceleration, or
sidecar path; those values are inputs to environment discovery, and the resolver
must turn validated choices into plan fields.

## 4. Instance factory and lifecycle

Provide an instance factory that returns a clean, isolated adapter instance.
Loading is transactional: prepare and validate all required components before
publishing the instance as active. A failed load must release partial resources
and leave the currently active model usable. `unload()` should be idempotent,
drop model/session/component references, and release backend-owned caches where
appropriate. Optional adapter hooks must not be assumed by the generic runtime.
Keep the runtime environment snapshot used to build the loaded instance attached
to that instance so request-time resolution cannot claim a different loaded
backend after process configuration changes.

## 5. Execution contract

- Accepted typed request and option schema:
- Input preprocessing and validation:
- Backend invocation:
- Output payload type and postprocessing:
- Usage/metrics/diagnostic metadata:
- Streaming/cancellation semantics, if supported:
- Cache/session ownership and cleanup:
- Safe fallback and the conditions that enable it:

Backend objects (MLX arrays, PIL images, tokenizer internals, etc.) must not
escape through the public request/result or Rust IPC DTOs.

## 6. Required tests

1. Discovery fixtures: supported, incomplete, malformed, and ambiguous
   checkpoints; assert evidence and component manifest.
2. Resolver table: task × modality × format × backend × dependency × hardware;
   assert exact plan or exact failure status.
3. Factory/lifecycle: isolated instances, successful replacement, failed-load
   atomicity, unload idempotence, and partial-component cleanup.
4. Contract: validation errors, empty/minimum/maximum input, output schema,
   streaming/cancellation if claimed, and fallback equivalence.
5. Device correctness: fixed prompts/assets and deterministic reference outputs;
   compare token IDs, shapes, numerical tolerances, and cache state as applicable.
6. Performance only after correctness: record hardware, dependency versions,
   warmup, repetitions/statistic, prompt/input token counts, generated/output
   sizes, TTFT/latency, throughput, and peak memory.

## 7. Support-state record

Use the key `(task, input modalities, output modalities, family, format,
backend, hardware profile)`. Attach evidence as the combination progresses:

`discovered → classified → loadable → smoke_pass → task_correct → context_or_shape_pass → benchmarked`

Terminal or blocked states must be explicit: `unsupported`, `ambiguous`,
`missing_component`, `dependency_blocked`, or `hardware_blocked`. Record the
MiniMLX revision, adapter/provider version, backend dependency versions,
hardware, test command, result, and artifact/log reference. “Supported” is not
an alias for “config recognized” or “weights loaded”.
