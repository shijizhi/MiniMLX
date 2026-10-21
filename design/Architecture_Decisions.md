# MiniMLX architecture decisions (ADR-001)

- Status: accepted for the framework + Qwen3.5 milestone
- Scope: classification, routing, adapter boundaries, artifact composition, and extension policy
- Review basis: upstream source/documentation inspected on 2026-09-27. Links below point to moving upstream branches/docs and are evidence for the observed design, not dependency pins.

## Context

MiniMLX intends to cover model tasks and input/output modalities across MLX
families without encoding all behavior in one classifier or family-specific
runtime. The architecture must separate what a model can do, which checkpoint
components are present, what a request asks for, and which installed providers
can execute it. It also needs to distinguish unsupported work from missing
artifacts/dependencies/hardware and preserve a path from cheap metadata probing
to real task-level correctness tests.

## Upstream patterns inspected

| Project | Observed pattern | Useful lesson | Deliberately not copied |
|---|---|---|---|
| [oMLX discovery](https://github.com/jundot/omlx/blob/main/omlx/model_discovery.py) and [engine exports](https://github.com/jundot/omlx/blob/main/omlx/engine/__init__.py) | Discovery returns broad model/engine categories; separate engines cover batched LLM, VLM, embedding, reranking, STT, TTS, and STS. Detection combines config fields, architecture/model-type tables, checkpoint contents, and heuristics; helper/drafter artifacts are distinguished from serving models. | Task engines and helper components are genuinely different execution paths. Discovery must inspect more than a model name and must identify auxiliary checkpoints. | A single mutually-exclusive `model_type`/`engine_type` as the whole capability model, and an ever-growing hand-maintained routing table as the extension mechanism. MiniMLX records independent task/modality/structure/acceleration/artifact facts and lets provider registrations compose a route. |
| [LM Studio mlx-engine loader](https://github.com/lmstudio-ai/mlx-engine/blob/main/mlx_engine/generate.py) | One public loader chooses sequential, batched, or batched-vision model kits and integrates `mlx-lm`/`mlx-vlm` plus speculative decoding. The inspected loader selects the vision kit from `vision_config`; the text path can also probe batching/cache support. | A model-kit boundary is useful: scheduling/cache behavior can differ while exposing one generation API. Backend and concurrency constraints must be explicit. | Treating modality metadata as a reliable proxy for which package implements a text architecture. The loader's `vision_config` branch is simple but upstream issue [#357](https://github.com/lmstudio-ai/mlx-engine/issues/357) documents a text-only architecture available in `mlx-vlm` that this dispatch misses. MiniMLX resolves actual provider registrations and dependencies, not a `vision_config` if/else. |
| [vLLM plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/) | Entry-point plugins register models, platforms, IO processors, loggers, and endpoints. Registrations are explicit and role-specific; plugin code executes in runtime processes and is a trust boundary. | Make extensions declarative and role-scoped; keep model registration separate from IO preprocessing and platform/backend integration. | Automatically importing/running arbitrary installed model code. MiniMLX built-ins are explicit; third-party code must be explicitly enabled/trusted, with compatibility/version constraints and subprocess boundary considered before adding plugin loading. |
| [llama.cpp model-author guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md) and [multimodal guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md) | A model family can require coordinated conversion metadata/tensor mapping, architecture metadata, graph construction, backend-op availability, and tests across consuming binaries. Multimodal encoders/processors/projectors live behind a separate multimodal subsystem and may require architecture-specific components. | Treat family graph/weight mapping and modality components as first-class work; require a cross-layer test matrix, and do not equate text-backbone support with VLM support. | Requiring one monolithic family implementation to own every transport/task. MiniMLX keeps task semantics, family adaptation, backend, and modality components composable while allowing one concrete adapter to implement multiple roles where that is simpler. |

## Decisions

1. **Capability is a product of orthogonal dimensions.** `ModelDescriptor` records
   observed architecture/config/artifacts with provenance. `CapabilityProfile`
   declares task, input/output modality, structure, and acceleration independently.
   Model names and config heuristics are evidence, never the support verdict.
2. **Discovery is cheap and side-effect-free.** It reads configuration and artifact
   manifests only; it does not import MLX, load weights, or select a device backend.
3. **Route planning composes role-specific providers.** The resolver chooses task,
   family, backend, and any registered component providers, then validates
   dependencies and declared version constraints (currently exact pins), required artifacts, memory, hardware,
   requested acceleration, and backend conflicts. The result is an inspectable
   `SupportDecision` plus `ExecutionPlan`, or a typed failure state.
4. **Artifacts are independent components.** Tokenizer, processor, text/vision/audio
   weights, projector, vocoder, shards, and drafter can be reported and routed
   independently. A text-capable multimodal checkpoint is not automatically
   image-capable in a particular runtime route.
5. **Typed task protocol is backend-neutral.** Requests/results carry task-tagged
   payloads and transport-safe modality references; MLX arrays, PIL objects, model
   internals, and device caches stay behind Python adapters.
6. **Model load/replacement is transactional.** Registry factories create isolated
   candidates. A failed candidate is cleaned up and the active instance remains
   usable; the runtime switches to the candidate only after successful loading. Optional
   family cleanup hooks are not required from every provider.
7. **Correctness states are explicit and evidence-based.** Recognition, route
   resolution, loading, task smoke, task correctness, context/shape checks, and
   benchmark status are distinct. Unsupported, ambiguous, missing-component,
   dependency-blocked, and hardware-blocked are not silently collapsed.
   Transport errors preserve a legacy human-readable message and additionally
   carry a stable error code, retryability flag, and structured decision details;
   old string-only error responses remain decodable.
8. **Qwen3.5 is a vertical validation fixture, not the architecture template.**
   Its generation/MTP behavior proves end-to-end routing and legacy API
   compatibility; Qwen-specific cache/kernel details remain behind that family's
   adapter and do not change the generic task/modality model.
9. **Vision requests follow the upstream message-to-embedding pipeline.**
   The external request preserves ordered message content (`text` and `image`),
   the checkpoint's native chat template owns image-token expansion, and the
   VLM backend owns image loading/preprocessing, vision-feature construction,
   prefill injection, and ordinary token-id decode. MiniMLX must not synthesize
   a fallback `<image>` marker because upstream oMLX documents that this can
   produce a prompt/image-token mismatch. This follows oMLX's `vlm.py` and
   `VLMModelAdapter` split and LM Studio's separate batched-vision model kit.
10. **VLM families share one execution implementation.** The family-neutral
    `MLXVLMAdapter` owns `mlx-vlm` loading, ordered message/template processing,
    image validation, generation, and streaming. Qwen3.5 inherits this path;
    its family adapter only supplies the measured vision/preprocessing cache
    hooks. New VL families should use the shared route first, adding family
    code only for demonstrated correctness or performance needs.

## Consequences and open gates

- Current types, registry, resolver, typed JSON protocol, Python worker, Rust DTOs,
  adapter factories, and component-plan selection implement the base contracts.
- Built-in generation includes Qwen3.5, Qwen2 text, and the family-neutral
  `mlx-vlm` vision route. The latter has MiniMLX device smoke coverage on
  Qwen2.5-VL for text-only, single-image, multi-image, and streaming requests;
  this does not imply every upstream `mlx-vlm` architecture is validated.
  Other task families remain unsupported/dependency-blocked until their own
  executor and correctness evidence are delivered.
- VLM routes delegate the final generation loop to `mlx-vlm`; the native model
  `get_input_embeddings` path performs multimodal prefill and token decode. A
  MiniMLX-owned prefill adapter is still required before adding continuous
  batching or claiming scheduler-level parity.
- An opt-in bounded cache now stores prepared processor tensors separately from
  prefix/KV state. Its key includes model identity, rendered prompt, image
  identity, and local-file freshness metadata. It is intentionally not called
  a KV cache; projected vision-feature reuse is delegated to the upstream
  `mlx-vlm.VisionFeatureCache` described below.
- The Qwen VLM adapter now also passes an upstream `mlx-vlm.VisionFeatureCache`
  into generation. This is the actual projected-feature cache used by the
  backend; the MiniMLX processor cache remains a separate optional layer and
  is cleared with the model instance.
- Device-side Qwen3.5 MTP now produces the same target greedy token sequence
  under `--compare-mtp-target-ar` and reports `compatibility=PASS` through the
  new plan; forced-rejection continuation and EOS boundary tests also pass.

## Evidence links

- oMLX's `ModelType`/`EngineType`, family tables, artifact/helper detection and
  metadata discovery: [model_discovery.py](https://github.com/jundot/omlx/blob/main/omlx/model_discovery.py)
- oMLX task engine decomposition: [engine package](https://github.com/jundot/omlx/tree/main/omlx/engine)
- LM Studio model-kit dispatch: [generate.py](https://github.com/lmstudio-ai/mlx-engine/blob/main/mlx_engine/generate.py)
- LM Studio dispatch limitation example: [mlx-engine issue #357](https://github.com/lmstudio-ai/mlx-engine/issues/357)
- oMLX VLM request/embedding pipeline: [vlm.py](https://github.com/jundot/omlx/blob/main/omlx/engine/vlm.py), [VLMModelAdapter](https://github.com/jundot/omlx/blob/main/omlx/models/vlm.py)
- LM Studio batched-vision model-kit implementation: [batched vision kit](https://github.com/lmstudio-ai/mlx-engine/tree/main/mlx_engine/model_kit/batched_vision)
- vLLM role-based extension registration and trust boundary: [plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/)
- llama.cpp architecture addition and multimodal components: [model author guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md), [mtmd development guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)
