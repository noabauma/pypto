# InjectTracrBuffer Pass

Injects the `__tracr_buffer` GM parameter into every kernel that issues cross-device communication, so codegen has somewhere to write TraCR trace records. Runs immediately before `MaterializeDistTensorCtx`.

## Overview

Profiling a collective means observing the wait, and the wait happens *inside* the kernel: `pld.system.notify` and `pld.system.wait` lower to inline peer-offset arithmetic and `TNOTIFY`/`TWAIT` on the AICore. Nothing above the core can see them, so a marker has to be emitted by the kernel itself — and it needs a place to put the record.

The device tier cannot use TraCR's own recording runtime. That runtime has `thread_local` state, heap buffers and a filesystem flush, none of which exist under CCEC. What it uses instead is a plain GM region of 16-byte payloads that the host serializes into a `.bts` lane afterwards. This pass is what puts that region in reach of a generated kernel:

1. Finds every function whose body issues `pld.system.notify` or `pld.system.wait`.
2. Adds a fresh `__tracr_buffer` Out-tensor parameter to each such function.
3. Propagates the parameter upward through the call graph, so the buffer flows from Orchestration down to the kernel that writes it.
4. Stops at Orchestration functions — they do **not** receive the parameter. Instead the pass injects a per-call-site `tensor.create`, which the host materializes and reads back.

**A program with no communication is left byte-identical.** That presence check is the pass's only gate: there is no backend flag and no configuration. The overwhelming majority of compiled models never mention notify or wait, so they see no signature change and no allocation for a profiler they are not using.

`pld.system.defer_wait` is deliberately **not** a trigger. It hard-requires a Simpler runtime and lowers through a different path, so instrumenting it is a separate question.

### Buffer layout

The parameter is an `INT64` tensor of `2 + 2 * 4096` elements, roughly 64 KB, matching the contract in the runtime's `aicore/tracr_aicore_emit.h`:

| words | meaning |
| --- | --- |
| 0 | record count |
| 1 | dropped count |
| 2 + 2n, 3 + 2n | payload *n* (packed ids, then timestamp) |

The capacity is deliberately modest. Overflow is not a correctness problem: the emitter drops the record and increments the drop counter, and it never wraps — `tracr_process` requires each `.bts` to be sorted by timestamp, and a wrapped ring is rotated rather than ordered. So a small buffer only loses the tail of a very long run, against a real cost for a large one: nobody can read a million spans, and every record is a GM store on the critical path of the communication being measured.

## Placement, and why it is not next to InjectGMPipeBuffer

The two passes do the same mechanical job, but this one cannot sit at position 26. Its slot is pinned from four directions:

| constraint | why |
| --- | --- |
| after `LowerCompositeOps` | the last pass that *creates* notify ops in user IR. `LowerHostTensorCollectives` lowers to pre-written builtin kernels, which are hand-written and out of scope here |
| after `ExpandMixedKernel` | so a split AIC/AIV pair each get the parameter, rather than one inheriting it |
| after `AutoDeriveTaskDependencies` | **on purpose.** A trace buffer written by several tasks would otherwise become a dependency edge between them, and the profiler would serialize the very schedule it is trying to observe |
| before `MaterializeDistTensorCtx` | that pass appends `CommCtx` parameters as a trailing suffix and relies on being the last to widen a signature |

The third row is a deliberate trade, and the one worth understanding. Being invisible to dependency analysis protects the measurement, but it also means concurrent writers are not ordered for us. Today `get_block_num(args)` is 1 and a designated-writer predicate keeps a single core recording, so nothing races; if that changes, per-core sub-buffers are the answer rather than a dependency edge.

## Requirements

- Input IR must still contain `pld.system.notify` / `pld.system.wait` as ops (they are lowered at codegen, so they survive the whole pipeline).
- Must run before `MaterializeDistTensorCtx`.
- No backend gate. On a program without communication the pass returns its input unchanged.

**When to use**: the default pipeline already places it in the correct slot. There is no reason to run it standalone except in tests.

> **Note**: the call-graph walk is shared with [`InjectGMPipeBuffer`](25-inject_gm_pipe_buffer.md) and lives in `transform_utils::InjectGMBufferParamInPlace`. Each pass supplies only a `GMBufferInjectionSpec`: the trigger predicate, and the parameter's name, dtype and size. Adding a third GM workspace parameter should extend that spec rather than copy the walk a third time.

## API

```python
from pypto import passes

program = passes.inject_tracr_buffer()(program)
```

```cpp
#include "pypto/ir/transforms/passes.h"

ir::Pass p = ir::pass::InjectTracrBuffer();
```

## Pass properties

| | properties |
| --- | --- |
| required | `CommDomainScopesMaterialized`, `ReturnParamsExplicit` |
| produced | `CommDomainScopesMaterialized`, `ReturnParamsExplicit` |

The pass establishes nothing new — it only widens signatures and call argument lists. What it declares is what it must not break, which is exactly what `MaterializeDistTensorCtx` reads next.

## Idempotency

The parameter name is the key: a function already carrying `__tracr_buffer` is skipped. Running the pass twice yields structurally identical IR, which matters because the pass sits inside a pipeline that may be re-entered.

## See also

- [25-inject_gm_pipe_buffer.md](25-inject_gm_pipe_buffer.md) — the sibling pass this one shares its walk with.
- [48-materialize_dist_tensor_ctx.md](48-materialize_dist_tensor_ctx.md) — runs immediately after, and is why this pass must not be last.
- [52-insert_comm_fence.md](52-insert_comm_fence.md) — the other pass built around notify/wait, for the data-before-signal contract rather than for observation.
