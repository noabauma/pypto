# InjectTracrBuffer Pass

为每个发出跨设备通信的 kernel 注入 `__tracr_buffer` GM 参数，使 codegen 有地方写 TraCR 追踪记录。紧邻 `MaterializeDistTensorCtx` 之前运行。

## 概述

给一次集合通信（collective）做 profiling，本质是观察那个 wait；而 wait 发生在 kernel **内部**：`pld.system.notify` 与 `pld.system.wait` 会在 AICore 上被降低为内联的 peer 偏移运算加 `TNOTIFY`/`TWAIT`。核（core）之上的任何一层都看不到它们，所以 marker 只能由 kernel 自己发出——而它需要一个地方存放记录。

设备层无法使用 TraCR 自带的 recording runtime：那个 runtime 有 `thread_local` 状态、堆缓冲区和文件系统 flush，在 CCEC 下都不存在。它改用的是一块由 16 字节 payload 组成的普通 GM 区域，事后由 host 序列化成一条 `.bts` lane。本 pass 的作用就是把这块区域送到生成的 kernel 手里：

1. 找出函数体中发出 `pld.system.notify` 或 `pld.system.wait` 的每个函数。
2. 为每个这样的函数追加一个新的 `__tracr_buffer` Out tensor 参数。
3. 沿调用图向上传播该参数，使缓冲区从 Orchestration 一路流到真正写它的 kernel。
4. 在 Orchestration 函数处停止——它们**不**接收该参数，而是由本 pass 在每个调用点注入一个 `tensor.create`，由 host 负责实际分配并回读。

**不含通信的程序保持逐字节不变。** 这个"是否存在通信 op"的判断就是本 pass 唯一的开关：没有 backend flag，也没有任何配置项。绝大多数被编译的模型从不出现 notify 或 wait，因此它们不会为一个自己并不使用的 profiler 付出签名变化或额外分配。

`pld.system.defer_wait` 被有意排除在触发条件之外：它强依赖 Simpler runtime，且走的是另一条降低路径，为它插桩是另一个独立问题。

### 缓冲区布局

该参数是一个 `INT64` tensor，元素数为 `2 + 2 * 4096`（约 64 KB），与 runtime 中 `aicore/tracr_aicore_emit.h` 的约定一致：

| 字（word） | 含义 |
| --- | --- |
| 0 | 记录条数 |
| 1 | 丢弃条数 |
| 2 + 2n, 3 + 2n | 第 *n* 条 payload（先是打包的 id，然后是时间戳） |

容量刻意取小。溢出不是正确性问题：emitter 会丢弃该记录并递增丢弃计数，而且**从不回绕**——`tracr_process` 要求每个 `.bts` 按时间戳有序，而回绕的 ring 是被轮转过的、不是有序的。所以小缓冲区只会丢掉超长运行的尾部；而大缓冲区有真实代价：没人能读完上百万条 span，且每条记录都是一次 GM 存储，正落在被测量的那次通信的关键路径上。

## 位置，以及为何不紧邻 InjectGMPipeBuffer

两个 pass 做的机械工作相同，但本 pass 不能放在位置 26。它的槽位被四个方向同时钉住：

| 约束 | 原因 |
| --- | --- |
| 在 `LowerCompositeOps` 之后 | 它是最后一个在用户 IR 中**创建** notify op 的 pass。`LowerHostTensorCollectives` 降低到的是预先写好的 builtin kernel，属手写范畴，不在本 pass 处理范围内 |
| 在 `ExpandMixedKernel` 之后 | 这样拆分后的 AIC/AIV 成对函数各自都能拿到该参数，而不是其中一个继承 |
| 在 `AutoDeriveTaskDependencies` 之后 | **这是有意的。** 若多个 task 写同一块追踪缓冲区，它会变成它们之间的依赖边，于是 profiler 会把自己正要观察的那个调度序列化掉 |
| 在 `MaterializeDistTensorCtx` 之前 | 该 pass 以尾部后缀（trailing suffix）的形式追加 `CommCtx` 参数，并依赖自己是最后一个加宽签名的 pass |

第三行是一个有意的取舍，也是最值得理解的一条。对依赖分析不可见保护了测量本身，但也意味着并发写入不会被替我们排序。今天 `get_block_num(args)` 为 1，且有一个 designated-writer 判定让只有单个核在记录，所以尚不存在竞争；一旦这一点改变，答案是按核切分子缓冲区，而不是加一条依赖边。

## 前置条件

- 输入 IR 中 `pld.system.notify` / `pld.system.wait` 必须仍以 op 形式存在（它们在 codegen 阶段才被降低，因此会贯穿整条流水线）。
- 必须在 `MaterializeDistTensorCtx` 之前运行。
- 没有 backend 开关。对于不含通信的程序，本 pass 原样返回输入。

**何时使用**：默认流水线已经把它放在正确的槽位。除测试外没有理由单独运行。

> **注意**：调用图遍历与 [`InjectGMPipeBuffer`](25-inject_gm_pipe_buffer.md) 共享，实现位于 `transform_utils::InjectGMBufferParamInPlace`。每个 pass 只提供一个 `GMBufferInjectionSpec`：触发判定，以及参数的名字、dtype 和大小。若要再增加第三个 GM workspace 参数，应扩展该 spec，而不是第三次复制这段遍历。

## API

```python
from pypto import passes

program = passes.inject_tracr_buffer()(program)
```

```cpp
#include "pypto/ir/transforms/passes.h"

ir::Pass p = ir::pass::InjectTracrBuffer();
```

## Pass 属性

| | 属性 |
| --- | --- |
| required | `CommDomainScopesMaterialized`, `ReturnParamsExplicit` |
| produced | `CommDomainScopesMaterialized`, `ReturnParamsExplicit` |

本 pass 不建立任何新属性——它只加宽签名和调用实参列表。它声明的是自己**不能破坏**的东西，而这正是紧随其后的 `MaterializeDistTensorCtx` 要读取的内容。

## 幂等性

参数名就是判据：已经带有 `__tracr_buffer` 的函数会被跳过。连续运行两次得到结构上完全相同的 IR——这一点很重要，因为本 pass 位于一条可能被重复进入的流水线中。

## 参见

- [25-inject_gm_pipe_buffer.md](25-inject_gm_pipe_buffer.md) —— 与本 pass 共享遍历实现的兄弟 pass。
- [48-materialize_dist_tensor_ctx.md](48-materialize_dist_tensor_ctx.md) —— 紧随其后运行，也是本 pass 不能放在最后的原因。
- [52-insert_comm_fence.md](52-insert_comm_fence.md) —— 另一个围绕 notify/wait 构建的 pass，服务于 data-before-signal 约定而非观测。
