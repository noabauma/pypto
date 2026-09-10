# 组合：Broadcast + AllReduce + AllGather

在一个内核中使用三种集合通信——本教程阶梯的收官之作，也是通往真实模型的桥梁。

> **前置：** [20-all_to_all](20-all_to_all.md)，即上一步——以及本页所组合的三种集合通信：[16-allreduce_reveal](16-allreduce_reveal.md) · [17-broadcast](17-broadcast.md) · [18-allgather](18-allgather.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 15 → **16** —— 本页是步骤 16。

## 思路

之前每一步都孤立地教授一个抽象。本步是第一次让一个内核做*不止一种*集合通信——也是第一次字节数不再是重点。真实模型正是这样做的：权重被广播、激活被 allreduce、结果被 allgather。下面的内核就是 picotron `model.py` 思想的缩影。

流水线：

1. **Broadcast**（步骤 12）——根的权重 `w` 到达每个 rank。
2. **Allreduce**（步骤 08–11）——每个 rank 得到 `Σ_k x[k]`。
3. **Allgather**（步骤 13）——每个 rank 得到 `concat(x[0], …, x[P-1])`。
4. **本地计算**——用共享权重 `w` 缩放 gather 后的矩阵（对 gather 到的隐藏状态施加一个学到的逐特征权重）。

## 运行

```bash
# 两个 rank：
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1

# 四个 rank——同一源码，只改 -d：
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1,2,3
```

预期输出：

```text
OK
```

golden 校验**两个**阶段：每个 rank 上 `allred[r] == Σ_k x[k]`，以及 `gathered[r] == concat(x[0], …, x[P-1]) * w`——被广播权重缩放的 allgather 结果，这也证明了权重到达了每个 rank。

## 走读

内核很短——三次内置调用加一次本地乘法——因为阶梯已经完成了工作：

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

- **这里用了三个信号 window——但并非因为需要三个。** 每个 InCore composite 都以自清理尾声结束，把自己那份信用额度再减回去，因此信号会归零，下一次调用重新从第 1 代开始。让三种集合通信复用同一个 `[nr, 1]` window 也能编译并在 P=2 与 P=4 下通过 golden。内核之所以使用 `sig_bcast` / `sig_ar` / `sig_ag`，是为了让每种集合通信的 barrier 在下面的 IR 对比中各自可见——这是教学选择，而非硬性要求。
- **`mode="mesh"` 显式给出**——步骤 11 的揭示让 mode 成为选择；这里点名写出，让读者看到完整调用。
- **allgather 的源是普通 `x` tensor**（步骤 13 的规则），而 broadcast 与 allreduce 接收 window——这三次调用在一处展示了 `pld.tensor.*` API 的完整面貌。
- **本地步骤是集合通信与数学相遇之处。** `chunk * w` 是 gather 后 tile 上的普通 `pl.mul`——步骤 01 的同一向量操作，如今作用于从全部 rank gather 而来的数据。

### IR 对比（教学工件）

Lowering 后的 IR 是你已熟悉的三个手工调度，按顺序：broadcast 的根 staging + barrier + 读取（步骤 12）、allreduce 的 mesh barrier + 累加（步骤 08）、以及 allgather 的 staging + barrier + 逐对端读取（步骤 13）。但其中只有两个与你手写的版本一致。第三个不一致：allgather 展开为逐对端 `pld.tile.put`，*然后*才是 barrier（步骤 13 的揭示），是 push，而你步骤 13 的内核是 pull。三者之所以能组合，是因为每个 lowering 都是自包含的——它在拿到的任意信号 window 上开启并关闭属于自己的信用代次——这也正是一个 window 就能服务三者的原因。

**成本卡（每 rank）：** 各组成部分之和——broadcast `(P-1)·N`、allreduce `(P-1)·N`、allgather `(P-1)/P·N`。注意中间这一项：本内核固定 `mode="mesh"`，因此付出的是 mesh 在步骤 08 的流量。你可能记得的 `2·(P-1)/P·N` 属于步骤 09-10 的两相与环形调度，而不属于 mesh。第一次，字节数不是重点：重点是三种调度组合进一个内核。

## 边界情况

> **致命陷阱——让同一个 window 跨两种信号*布局*使用。** 在连续的 InCore 集合通信之间复用信号是安全的，因为信用协议会自清理。不安全的是让同一个 window 对应两种不同的约定：mesh `[nr, 1]` 与 ring `[2*(nr-1), nr]` 的单元寻址方式不同，按其中一种分配却按另一种寻址就是错的。在 rank 数量*静态*时编译器会替你抓住它——`ValidateMeshSignalShape` 会以 “signal shape[1] must be 1 (one cell per rank)” 拒绝 ring 形状的 window。当 `shape[1]` 为符号时该检查会被跳过，那才是它变得静默的情形。**修复：** 每种*布局*一个 window，而不是每次调用一个。（HOST 内置 allreduce 属于另一种情况——它尚未自清理，因此仍然禁止出现在 `for` / `while` 内。）

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 编译期报 `signal shape[1] must be 1 (one cell per rank)` | 把 ring 形状 `[2*(nr-1), nr]` 的 window 传给了 mesh 布局的集合通信 | 每种布局各自一个 window；InCore composite 连续复用同一信号是安全的（自清理） |
| 第二/第三个集合通信提前通过 | 在循环中复用了 HOST 路径的信号，或因 `shape[1]` 是符号而使编译器无法发现布局不匹配 | 让列数保持静态以便检查生效，或每种布局一个 window |
| `allred` 错但 `gathered` 对 | allreduce 源未 staging / op 错误 | 把 `x` staging 进 `ar_data`；`op=Sum`；`mode="mesh"` |
| `gathered` 错但 `allred` 对 | 未应用广播权重，或行错误 | 对每行 `chunk = pl.mul(chunk, w)` |
| `pld.tensor.allgather` 源被拒绝 | 传的是 tile 而非 tensor | 传普通 `x` tensor |
| 非根权重泄漏到输出 | 根未 staging | 仅在 `if my_rank == ROOT_RANK` 下 staging `w_data` |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 16 行）
- [01-collectives](../distributed/01-collectives.md) — 整个集合通信动物园
- [17-broadcast](17-broadcast.md) / [18-allgather](18-allgather.md) / [20-all_to_all](20-all_to_all.md) — 本内核所组合的组件
- 更高级的应用（此处不重述）：pypto-lib [#869](https://github.com/hw-native-sys/pypto-lib/pull/869)（AllGather-GEMM）与 DeepSeek-V4 分布式 MoE dispatch/combine——模型规模下的相同模式
- [04-debugging](04-debugging.md) — 分布式程序的规范故障目录
- 这是阶梯的终点——索引按顺序列出了全部内容。
