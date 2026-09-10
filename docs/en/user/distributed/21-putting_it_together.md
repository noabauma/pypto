# Putting It Together: Broadcast + AllReduce + AllGather

Three collectives in one kernel — the capstone of the ladder, and the
bridge into real models.

> **Prerequisites:** [20-all_to_all](20-all_to_all.md), the previous step —
> plus the three collectives this page composes:
> [16-allreduce_reveal](16-allreduce_reveal.md) ·
> [17-broadcast](17-broadcast.md) · [18-allgather](18-allgather.md). Any
> number of devices ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 15 → **16** — this page is step 16.

## The idea

Every earlier step taught one abstraction in isolation. This step is the
first place a kernel does *more than one* collective — and the first place the
byte count is not the point. Real models do exactly this: weights are
broadcast, activations are allreduced, results are allgathered. The kernel
below is the picotron `model.py` idea in miniature.

The pipeline:

1. **Broadcast** (step 12) — root's weights `w` reach every rank.
2. **Allreduce** (steps 08–11) — every rank ends with `Σ_k x[k]`.
3. **Allgather** (step 13) — every rank ends with `concat(x[0], …, x[P-1])`.
4. **Local compute** — the gathered matrix is scaled by the shared weight `w`
   (a learned per-feature weight over gathered hidden states).

## Run it

```bash
# Two ranks:
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1

# Four ranks — the same source, only -d changes:
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1,2,3
```

Expected output:

```text
OK
```

The golden checks **both** stages: `allred[r] == Σ_k x[k]` on every rank, and
`gathered[r] == concat(x[0], …, x[P-1]) * w` — the allgather result scaled by
the broadcast weight, which also proves the weight reached every rank.

## Walkthrough

The kernel is short — three builtin calls plus a local multiply — because the
ladder did the work:

```python
@pl.function(type=pl.FunctionType.InCore)
def compose_step(self, x, w_in, allred, gathered, w_data, ar_data, ag_data,
                 sig_bcast, sig_ar, sig_ag):
    ctx = pld.get_comm_ctx(w_data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # 1 — Broadcast: root stages its weights, every rank gets them.
    if my_rank == ROOT_RANK:
        local_w = pl.load(w_in, [0, 0], [1, SIZE])
        w_data = pl.store(local_w, [0, 0], w_data)
    w_data = pld.tensor.broadcast(w_data, sig_bcast, root=ROOT_RANK)
    w = pl.load(w_data, [0, 0], [1, SIZE])

    # 2 — Allreduce: every rank ends with the element-wise sum.
    local_x = pl.load(x, [0, 0], [1, SIZE])
    ar_data = pl.store(local_x, [0, 0], ar_data)
    ar_data = pld.tensor.allreduce(ar_data, sig_ar, op=pld.ReduceOp.Sum, mode="mesh")
    total = pl.load(ar_data, [0, 0], [1, SIZE])
    allred = pl.store(total, [0, 0], allred)

    # 3 — Allgather: every rank ends with all ranks' raw slices.
    ag_data = pld.tensor.allgather(x, ag_data, sig_ag)

    # 4 — Local: scale the gathered matrix by the shared weight.
    for src in pl.range(nranks):
        chunk = pl.load(ag_data, [src, 0], [1, SIZE])
        chunk = pl.mul(chunk, w)
        gathered = pl.store(chunk, [0, src * SIZE], gathered)
    return gathered
```

- **Three signal windows here — but not because three are needed.** Every
  InCore composite ends with a self-clearing epilogue that subtracts its own
  credits back out, so the signal returns to zero and the next call starts at
  generation 1 again. One `[nr, 1]` window reused by all three collectives
  compiles and passes the golden at P=2 and P=4. The kernel uses
  `sig_bcast` / `sig_ar` / `sig_ag` so that each collective's barrier is
  separately visible in the IR diff — a teaching choice, not a requirement.
- **`mode="mesh"` is explicit** for the allreduce — the step-11 reveal made
  the mode a choice; here it is named so the reader sees the full call.
- **The allgather source is the plain `x` tensor** (step 13's rule), while
  broadcast and allreduce take windows — the three calls show the full
  surface of the `pld.tensor.*` API in one place.
- **The local step is where the collective meets the math.** `chunk * w` is
  an ordinary `pl.mul` on the gathered tile — the same vector op from step 01,
  now acting on data gathered from every rank in the world.

### The IR diff (the teaching artifact)

The lowered IR is three schedules in order — but only two of them match the
hand-rolled version you wrote: the broadcast's barrier + `pld.tile.get` from
the root (step 12) and the allreduce's mesh barrier + accumulate (step 08).
The third does not: the allgather expands to `pld.tile.put` per peer *then* a
barrier (step 13's reveal), a push where your step-13 kernel pulled. The three
compose because each lowering is self-contained — it opens and closes its own
credit generation on whatever signal window it is handed — which is exactly
why one window could serve all three.

**Cost card (per rank):** the sum of the pieces — `(P-1)·N` for the
broadcast, `(P-1)·N` for the allreduce, `(P-1)/P·N` for the allgather. Note
the middle term: this kernel pins `mode="mesh"`, so it pays mesh's step-08
traffic. The `2·(P-1)/P·N` figure you may remember belongs to the two-phase
and ring schedules of steps 09-10, not to mesh. For the first time the byte
count is not the point: the point is that three schedules compose into one
kernel.

## Edge cases

> **Fatal pitfall — sharing one window across two signal *layouts*.** Reusing
> a signal between back-to-back InCore collectives is safe; the credit
> protocol is self-clearing. What is not safe is pointing one window at two
> different conventions: mesh `[nr, 1]` and ring `[2*(nr-1), nr]` address
> their cells differently, so a window sized for one and addressed as the
> other is wrong. With a *static* rank count the compiler catches it for you —
> `ValidateMeshSignalShape` rejects a ring-shaped window with "signal shape[1]
> must be 1 (one cell per rank)". The check is skipped when `shape[1]` is
> symbolic, and that is the case where it goes silent. **Fix:** one window per
> *layout*, not per call.
> (The HOST builtin allreduce is a separate case — it is not yet
> self-clearing, so it remains rejected inside `for` / `while`.)

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `signal shape[1] must be 1 (one cell per rank)` at compile time | A ring-shaped `[2*(nr-1), nr]` window passed to a mesh-layout collective | Give each layout its own window; back-to-back reuse by InCore composites is safe (self-clearing) |
| Second/third collective passes early | A HOST-path signal reused in a loop, or a layout mismatch the compiler could not see because `shape[1]` is symbolic | Keep the column count static so the check fires, or one window per layout |
| `allred` wrong but `gathered` right | Allreduce source not staged / wrong op | Stage `x` into `ar_data`; `op=Sum`; `mode="mesh"` |
| `gathered` wrong but `allred` right | Broadcast weight not applied, or wrong row | `chunk = pl.mul(chunk, w)` for every row |
| `pld.tensor.allgather` source rejected | Tile passed instead of a tensor | Pass the plain `x` tensor |
| Non-root weights leak into output | Root staging missing | Stage `w_data` only under `if my_rank == ROOT_RANK` |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 16)
- [01-collectives](../distributed/01-collectives.md) — the whole collective zoo
- [17-broadcast](17-broadcast.md) / [18-allgather](18-allgather.md) /
  [20-all_to_all](20-all_to_all.md) — the pieces this kernel composes
- More advanced applications (not restated here): pypto-lib
  [#869](https://github.com/hw-native-sys/pypto-lib/pull/869) (AllGather-GEMM)
  and the DeepSeek-V4 distributed MoE dispatch/combine — the same patterns at
  model scale
- [04-debugging](04-debugging.md) — the canonical failure catalog for distributed programs
- This is the end of the ladder — the index lists everything, in order.
