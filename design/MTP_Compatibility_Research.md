# Qwen3.5 MTP compatibility research

## Current MiniMLX boundary

- `Qwen35Adapter.generate()` uses a greedy single-token decode loop. It has no MTP proposal, target verification, accepted-token accounting, or speculative cache rollback path.
- The local `Qwen3.5-0.8B-MLX-4bit/config.json` declares `text_config.mtp_num_hidden_layers = 1` and `mtp_use_dedicated_embeddings = false`.
- The checkpoint's `model.safetensors.index.json` contains no `mtp` or `nextn` weights. The metadata alone therefore does not make this checkpoint an MTP-capable target/drafter.
- MiniMLX currently loads through `mlx_lm.load`; there is no `mlx_vlm` API integration in the runtime. The public Qwen MTP drafter family uses `model_type=qwen3_5_mtp`, which is not a normal `mlx_lm` model family.
- The project venv now has the explicitly selected stack `mlx==0.32.2`, `mlx-lm==0.31.3`, `mlx-vlm==0.7.2`, `transformers==5.17.0`; the user's `pip check` reported no broken requirements. This updates the earlier “mlx-vlm not installed” finding.

## Source-grounded integration direction

oMLX's `omlx/speculative/vlm_mtp.py` wraps MLX-VLM's MTP drafter loader and round helpers behind an adapter. Its caller contract makes the hybrid-model requirements explicit: target prefill must provide the final hidden state, post-prefill cache, shared K/V snapshots, and the first bonus token; the round loop then proposes, verifies, and rolls back cache state as needed. MiniMLX should adapt this contract behind its own interface rather than import oMLX code or place speculative logic in the generic generation loop.

MLX-VLM's public usage documentation likewise loads a drafter and passes it into `stream_generate`; its Qwen MTP drafter is separate from the target weights. MTP is therefore an opt-in decode backend with the existing autoregressive path retained as fallback.

### Concrete gap in MiniMLX (confirmed from current source)

MiniMLX's fast greedy path in `python/minimlx_runtime/adapters/qwen35.py` performs chunked prefill, obtains only the final logits, then feeds one sampled token per target forward. It does not request/retain the final pre-norm hidden state or shared K/V snapshots needed by the MTP round helper. Its `cache` is the hybrid cache returned by `model.make_cache()`, but MiniMLX has no transaction boundary for rolling back both full-attention KV positions and recurrent GDN/conv state after a rejected proposal.

The installed MLX-LM 0.31.3 implementation confirms the two cache families: `Qwen3_5TextModel.make_cache()` creates `KVCache` for full-attention layers and `ArraysCache(size=2)` for linear layers. `KVCache` supports `trim()` and tracks a token offset. In contrast, `ArraysCache` stores the convolution state and recurrent matrix as two arrays and has no `trim()` or speculative transaction methods. Thus truncating all caches uniformly is not a valid rollback. On a rejected block, a correctness-first fallback can restore a pre-round snapshot and replay only the accepted prefix through the target; a faster commit path needs an explicit, verified transaction for both ArraysCache entries. This is a concrete implementation decision, not an inference from timing.

MLX-VLM 0.7.2 does contain the MTP APIs and Qwen3.5 hooks: `load_drafter`, `_buffer_mtp_target_cache`, `_mtp_rounds`, `_mtp_rounds_batch`, plus its Qwen3.5 `speculative_verify_hidden/logits_from_hidden` and `rollback_speculative_cache`. These target hooks live on MLX-VLM's own Qwen3.5 `LanguageModel` and exact verifier; they are not automatically available on MLX-LM's `TextModel`. Installing the drafter library therefore resolves only the drafter side. MiniMLX still needs an equivalent target forward/cache adapter, or an explicit decision to use MLX-VLM's target executor for the MTP path. Do not claim MTP is integrated merely because the package imports or drafter weights load.

### Exact GDN transaction mechanism in MLX-VLM 0.7.2

The cache mismatch is now confirmed at source level, not inferred from API names:

- MLX-VLM's `models/cache.py::ArraysCache` adds `start_speculation(length)`, `update_recurrent`, and `update_window`. It snapshots the two original GDN cache slots and metadata. During the block, it retains per-token recurrent states and causal-convolution windows; `commit_speculation(accepted_lengths)` selects the state after each accepted prefix, while `abort_speculation()` restores the pre-block snapshot.
- MLX-VLM's Qwen3.5 exact verifier uses `cache.update_window(...)` for the convolution state and calls `gated_delta_update(..., cache=cache)`. That operator delegates recurrent state ownership/history to `ArraysCache.update_recurrent(...)`; it requests intermediate GDN states while speculation is active.
- The shared transaction also tracks ordinary append-only KV caches and has a special recording/replay transaction for rotating KV caches. This is why trimming the KV cache alone cannot correctly implement Qwen3.5 MTP rejection.
- MLX-LM 0.31.3's `models/cache.py::ArraysCache` only stores arrays and padding metadata; it has no speculation lifecycle or temporal-history methods. Its Qwen3.5 GDN implementation writes the final conv/recurrent arrays directly. The MLX-VLM verifier therefore cannot be called against MiniMLX's current MLX-LM caches as-is.

The practical implementation choices are consequently precise: (A) make an MTP-only target backend using MLX-VLM's Qwen3.5 model/cache/verifier end-to-end, or (B) port the needed Qwen3.5 target verify forward and bounded cache transaction semantics into a MiniMLX adapter over MLX-LM. Option B must keep the exact verifier math/quantized linear behavior and pass per-token equivalence plus forced accept/reject rollback tests before enabling MTP. These are not interchangeable small patches: the target model object, cache type, verifier forward, and drafter helpers must all agree. No backend switch should be made silently.

The tool session used for this task cannot initialize Metal, so importing `mlx_vlm` raises `No Metal device available`. Static inspection of the installed 0.7.2 files confirmed the symbols and transaction behavior above, but runtime imports/model loading must be checked from the user's interactive RustRover or terminal process with Apple GPU access.

The inspected oMLX wrapper delegates to MLX-VLM's `load_drafter` and private `_buffer_mtp_target_cache`, `_mtp_rounds`, `_mtp_rounds_batch` helpers. Current upstream MLX-VLM routes those round helpers through `mlx_vlm.speculative.utils`; its prefill contract explicitly requests `return_hidden=True` and `return_shared_kv=True`, and passes the final hidden state plus shared-KV snapshot into MTP. The verifier exposes speculative transaction commit/abort around the round. These are private/version-sensitive integration points, not a stable generic `mlx_lm` API. The target verification path also needs hidden-to-logits/argmax hooks and transactional cache commit/abort. An implementation that merely adds a draft-model call to MiniMLX's token loop would be incorrect: rejection must restore the complete hybrid state, not just truncate KV cache.

The current drafter installed locally was inspected as `model_type=qwen3_5_mtp`, `block_size=4`, affine Q4/group 64, with one MTP layer and actual MTP tensors in its safetensors index. It is paired with the local Qwen3.5 4B Q4 target; this is separate from the 0.8B target whose index has no MTP weights. The repository's earlier model-card note saying block size 2 is stale and must not be used for dispatch.

Implementation boundary: keep this behind an explicit opt-in and retain autoregressive fallback. The implementation selects one coherent target path: MLX-VLM Qwen3.5 target/cache/verifier end-to-end, paired with MLX-VLM's drafter and private MTP rounds. It is enabled only when `MINIMLX_MTP_DRAFT_MODEL` is set and requires exactly `mlx-vlm==0.7.2`; target/drafter compatibility is checked by MLX-VLM before generation. It does not mix MLX-VLM's verifier with MLX-LM's unmodified `ArraysCache`, import oMLX code, or depend on oMLX scheduler/proxy/Metal-pool patches.

## Candidate for a short MTP integration test

The best immediate candidate is `mlx-community/Qwen3.5-4B-MTP-4bit`: its model card explicitly identifies it as the separate Qwen3.5 4B drafter, with MTP block size 2, affine Q4/group 64, and 67.9 MB on disk. The matching 4B Q4 target already exists locally. The drafter checkpoint is not standalone and its model type is `qwen3_5_mtp`; it must be loaded through an MTP-capable runtime path, not passed to the normal target loader. This keeps target and drafter within the same model family/size and avoids changing the target quantization for the initial correctness test.

An alternate small-model candidate is `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`, which bundles a separate MTP head. It is mixed-precision and differs from MiniMLX's existing uniform-Q4 0.8B baseline, so it is not the first choice for the paired 4B integration test.

## Validation gates

1. [x] Inspect target and MTP tensor keys/shapes and confirm the drafter is paired with the target architecture/tokenizer; `block_size=4`.
2. [x] Select MLX-VLM's own Qwen3.5 target/cache/verifier as the isolated MTP backend; do not splice its cache into MLX-LM.
3. [x] Add the optional MLX-VLM drafter/round bridge behind an explicit version check and environment opt-in; leave normal import and greedy decode independent of it.
4. [x] Compare deterministic greedy token IDs against same-target autoregressive output: exact token equality passed on the 128-token “sky appears blue” prompt and 64-token “ocean looks blue” prompt, including naturally accepted and rejected proposal rounds. The 1-token output boundary matched with no speculative rounds. A controlled verifier-token override also forced EOS at output index 1 and forced a rejection at index 1; both outputs matched the same-target AR control.
5. [x] Natural rejected rounds and a controlled first-draft rejection were exercised. The forced-rejection case continued for five MTP rounds and remained exactly equal to the controlled same-target AR output, providing end-to-end evidence that the post-rejection cache state is usable. A separate field-by-field cache snapshot assertion is still absent.
6. [x] Record acceptance and peak memory after correctness: 128-token run drafted 134 / accepted 83 (61.9%), peak 3.242 GiB; 64-token run drafted 77 / accepted 38 (49.4%). One paired same-process run measured MTP decode 2706.824 ms versus target AR 4021.117 ms (about 1.49x), but this is a single sequential sample and not a stable performance claim.
7. [x] MTP remains disabled by default and requires `MINIMLX_MTP_DRAFT_MODEL`; autoregressive decode remains the fallback.

### Current validation result

MTP token-equivalence passed for the two natural-language prompts above at output lengths 128 and 64, and for the one-token boundary case. An earlier mismatch at output token index 112 was traced to sequence-shape-dependent results from batched `speculative_argmax_from_hidden`: singleton evaluation and replay both selected token 332, while batched verification selected 37951. Applying the argmax independently at each sequence position removed the mismatch; the same-target AR comparison then passed. This is the evidenced root cause and correction, not a sampling/parameter adjustment.

Status: correctness milestone passed for the tested greedy prompts and controlled EOS/rejection boundaries; MTP remains opt-in. The forced tests control target verifier outputs and validate stop/rollback control flow; they do not claim naturally generated EOS coverage or field-by-field cache introspection. The measured speedup is preliminary only.

## References

- oMLX MTP integration: https://github.com/jundot/omlx/blob/main/omlx/speculative/vlm_mtp.py
- MLX-VLM MTP helper exports and prefill contract: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/utils.py
- MLX-VLM drafter loader export: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/__init__.py
- MLX-VLM speculative decoding API: https://github.com/Blaizzy/mlx-vlm/blob/main/docs/usage.md
- Matching Qwen3.5 4B MTP drafter: https://huggingface.co/mlx-community/Qwen3.5-4B-MTP-4bit
- Qwen3.5 0.8B OptiQ checkpoint and bundled MTP head: https://huggingface.co/mlx-community/Qwen3.5-0.8B-OptiQ-4bit
