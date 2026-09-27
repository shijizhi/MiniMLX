# Prefill scheduling

## Scope

Prefill chunk size is a scheduling parameter, not a model algorithm. It trades
kernel reuse and launch overhead against peak working-set size and memory
pressure. The scheduler therefore must be workload- and hardware-aware, while
keeping an explicit override for reproducibility.

## Current implementation

`--prefill-step-size N` keeps the exact requested positive value. `0` is an
opt-in automatic mode with the initial conservative policy:

| Prompt tokens | Step size |
| ---: | ---: |
| `<= 4096` | 1024 |
| `4097..16384` | 2048 |
| `> 16384` | 4096 |

This is only a scheduling experiment and is not enabled by the default
configuration. It does not assume Qwen3.5 layer counts or attention types, and
it does not alter decode scheduling. The initial length-only policy did not
establish a measurable 4K improvement, so it remains available only for
diagnostics and future hardware-specific tables.

## Validation gate

Before changing the default, measure the same build and model at 1K, 4K, 8K,
16K and at least one 32K+ context. For each context, compare explicit 1024,
2048, 4096 and auto where feasible, using the same warm-up, generated-token
count and median-of-repeats. Record TTFT, TPOT, E2E, peak memory and whether
the request completes successfully. Auto is accepted only if it does not
regress the best explicit choice beyond measurement noise and remains safe at
the largest supported context. The current policy has not passed that gate.
