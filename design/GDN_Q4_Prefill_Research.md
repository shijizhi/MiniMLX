# GDN Q4 Prefill Projection Research

## Motivation

The current context matrix shows that MiniMLX is already faster than oMLX in
decode, while TTFT is still slightly behind at every measured context length.
The next prefill candidate is therefore a projection-level path, not another
decode or cache experiment.

Public behavior observed in oMLX indicates that long-prefill Qwen3.5 GDN
projections are routed through a specialized native Q4 matrix path. MiniMLX
currently has an inference-gated dense-BF16 path for Q4 MLP prefill, but its
GDN `qkv`, `z`, `beta`, `a`, and output projections still use the regular MLX
quantized matmul entry.

This is a behavioral and shape-level hypothesis only. No oMLX implementation
code is copied. The independent probe reuses MiniMLX's own correctness-first
affine-Q4 research kernel and does not change production dispatch.

## Experiment and acceptance gate

Run `tools/experiment_gdn_q4_prefill_projection.py` at 1K/2K/4K/8K/16K/32K
where memory permits. For every projection and the four-input fused call,
record:

1. max absolute error against MLX's native QMM;
2. median isolated latency;
3. whether the custom path is faster after warm-up;
4. whether the result remains valid for the real layer shape.

The candidate may enter production only if the kernel is numerically correct,
has strict shape/bit/group-size gates and fallback, and produces a stable
end-to-end TTFT improvement without decode regression. A faster isolated
projection alone is insufficient.

## First result: scalar affine-Q4 probe rejected

The 4K real-layer experiment completed with correctness within the expected
BF16 quantization accumulation range, but the kernel is not competitive:

| Projection | Native QMM | Probe kernel | Relative speed |
|---|---:|---:|---:|
| qkv | 17.653 ms | 101.275 ms | 0.174x |
| z | 6.090 ms | 33.941 ms | 0.179x |
| beta | 0.268 ms | 0.456 ms | 0.589x |
| a | 0.298 ms | 0.446 ms | 0.670x |
| out | 6.110 ms | 33.473 ms | 0.183x |
| four-input fused | 23.571 ms | 135.507 ms | 0.174x |

The probe is rejected for production. This specifically rejects a
scalar/simple-tile implementation; it does not test the optimized simdgroup
tiled Q4 design needed to compete with native QMM. The measured errors were
`0.015625–0.0625` in BF16 output units, so any new kernel must also include a
tighter numerical gate before performance evaluation.

## Next candidate

Only a new simdgroup-tiled design is worth testing: decode packed Q4 values
into register/tile fragments, amortize scale/bias loads across the sequence
tile, and keep the output projection's input layout contiguous. The next
probe must first pass a stricter error check and then beat native QMM on the
large `qkv`/`z` shapes; `beta/a` alone are too small to justify a custom path.

The next version was implemented as an isolated 8x8 probe in
`tools/experiment_q4_mlp_prefill.py` and reported alongside the GDN projection
experiment as `tiled_error`, `tiled`, and `tiled_speedup`.

## Second result: simdgroup probe rejected

The 4K run produced large numerical errors (`qkv` 11.25, `z` 4.75, `out`
3.33) and was dramatically slower than native QMM (`qkv/z/out` about
0.013–0.014x). This is an implementation-correctness failure in the fragment
layout/mapping, not a meaningful performance comparison. The scalar and first
simdgroup probes are both closed; neither is allowed into production.

The two JIT probes are now closed. A native-extension reimplementation would
be technically feasible, but it is deferred: current end-to-end evidence does
not justify taking on the MLX ABI, metallib, build, and version-compatibility
surface for this candidate.

## Corrected implementation target

The public oMLX implementation shows a compiled `Qwen35QAffineQmmTPrimitive`,
not a Python `mx.fast.metal_kernel`. Its dispatch selects a compiled
`BM/BK/BN` tile variant (variant 8 is `64 x 32 x 64`), validates contiguous
arrays, affine mode, group size 64/128, supported bit widths, and falls back
when the native path is unavailable. The Qwen GDN wrapper then routes the four
input projections and `out_proj` through that primitive while leaving the rest
of the GDN forward intact.

That explains why the two probes above were invalid comparisons: they tested a
different execution layer, different tile implementation, and incomplete
fallback contract. The native-extension route remains documented as a future
option, but is not the next experiment.

## oMLX implementation audit

The current public oMLX source confirms that the missing performance path is
not a wrapper around `mlx.quantized_matmul`. It constructs a lazy
`Qwen35QAffineQmmTPrimitive`, allocates the MLX output inside the primitive,
and launches a compiled Metal kernel selected by `BM/BK/BN` variant. The
classic group-size-64 kernel uses `qmm_t_impl<T, 64, bits, true, BM, BK, BN>`;
variant 8 is `BM=64, BK=32, BN=64`. It validates row-contiguous inputs,
packed `uint32` weights, matching scale/bias dtype, and the packed Q4 shape
before dispatch.

The oMLX build also depends on its shared `quantized_moe.h` Metal helpers and
ships a sibling `.metallib`; its CMake target links MLX and the Apple Metal
frameworks and uses `NB_DOMAIN mlx`. This is materially different from the
current isolated Python probe. The probe's zero-error result is therefore
only a binding/reference baseline and must not be treated as a native-kernel
speed result.

The next implementation step is to port only the classic group-size-64 Q4
primitive and its required Metal helper headers, then compare variant 8 at
2048/4096/16384 tokens. NAX kernels, ANE paths, and production dispatch remain
out of scope until that classic path passes correctness and end-to-end TTFT.

The exact upstream snapshot audited for this port is oMLX commit
`057847315a3a4b9a74650476020386eb9419dd20` (Apache-2.0). The minimal source
set is 5 files from `qwen35_prefill/csrc` plus the shared
`common/csrc/kernels/quantized_moe.h`; the latter is the substantial part of
the implementation and is why the current 20-line reference wrapper cannot
be promoted by simply changing its function name.

MiniMLX now has an independent first implementation in
`native/minimlx_q4_native`: a Q4-only Steel tile, a MiniMLX-owned lazy
primitive, and a separate metallib. It intentionally narrows the contract to
BF16, affine Q4, group size 64, and aligned `64 x 32 x 64` tiles. It does not
copy oMLX's generic loader, NAX path, ANE path, or multi-bit dispatch.

## Decision

Deferred. Current production code remains MLX-native and portable. The
reference probe is complete; production work resumes only with the classic
oMLX-compatible primitive described above, not by routing the current probe
into production.

Production dispatch remains unchanged.
