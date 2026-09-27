# Prefix cache research

## Why this is the next target

The current MiniMLX adapter creates a fresh model cache for every generation
request. That recomputes an identical system prompt or conversation prefix on
every turn. mlx-lm exposes prompt-cache primitives, and oMLX keeps an
LRU/trie cache for nearest-prefix reuse. MiniMLX should preserve this
capability with an independent implementation.

## First gate

`tools/experiment_prompt_cache.py` compares a cold forward over `prefix +
suffix` with a warm forward that reuses an independently cloned cache for
`prefix` and evaluates only `suffix`. Different prefill/decode execution
shapes can produce small floating-point logit differences, so the acceptance
gate is semantic: top-1 and top-5 agreement, while max absolute logit error is
reported diagnostically. Run it at 4K, 16K, and 32K prefixes on the target Mac
before changing the production path.

Initial 4K result on the target Mac: cold full forward was about 2.89 s,
cached suffix about 29.5 ms, or roughly 98x for a 16-token suffix. Top-1 and
top-5 both matched; max absolute logit error was 0.34375. This validates the
potential but does not yet justify production integration.

The first production-path extension test also passed: a 4096-token request
followed by the same prefix plus 16 new tokens hit the cache, reduced the
second request to about 321 ms, and reported about 6.4x total speedup. The
total includes generation time, so it is lower than the isolated prefill
speedup.

A 16K production-path comparison passed as well. Without reuse, the second
request took about 9296 ms; with reuse it took about 330 ms, including about
65 ms of prefill, for about 28.3x total speedup. The 16K snapshot occupied
about 221 MB and the expanded 16400-token snapshot occupied about 442 MB.

## Production requirements

- cache keys include model/tokenizer identity and exact token IDs;
- recurrent state and ordinary KV state are snapshotted together;
- requests never mutate an entry owned by the cache;
- exact hits and longest-prefix hits have separate correctness tests;
- memory limits and eviction are explicit; cold-request behavior stays intact;
- hit/miss and cache-byte telemetry are exposed before enabling a default UI policy.

## Intended runtime architecture

The production owner should have two separate tiers:

1. **Session cache** — the primary path. A conversation/session owns its
   token history and the corresponding model state. When a new turn appends to
   that history, the runtime reuses the session state directly. Editing or
   branching a conversation creates a new state lineage.
2. **Shared prefix cache** — optional secondary path. Stable system prompts or
   document prefixes are keyed by model fingerprint, tokenizer fingerprint,
   exact token IDs, and cache format. It uses bounded block snapshots and may
   later spill to disk.

The current `PrefixCache` is only the first shared-memory prototype. It is
useful for proving the speedup, but it is not yet the final session API: it is
process-local, uses a small entry list, and only accepts strict prefix
extension. The next implementation step is to add an optional `session_id` at
the protocol boundary and make session reuse explicit. A global cache must not
silently mix unrelated conversations just because their token prefixes happen
to match.

This separation follows the same broad capability split visible in oMLX: its
request model tracks cached tokens and prompt state, while its prefix-cache
implementation adds shared blocks and tiered persistence. MiniMLX will keep
the ownership and cache format independent.
