# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Unit tests for the ``InjectTracrBuffer`` pass.

The pass gives every kernel that issues ``pld.system.notify`` / ``pld.system.wait``
a ``__tracr_buffer`` Out-tensor parameter, so codegen has a GM region for AICore
TraCR records, and propagates it through callers. Orchestration functions
materialize one buffer per call site instead of taking a parameter.

The most important property is the **gate**: a program that never communicates
must come out byte-identical, because that is what keeps every non-distributed
model free of a signature change. That one is asserted structurally; the
injection cases assert on the resulting signature, where a hand-written
``Expected`` carrying an 8194-element parameter would obscure more than it pins.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.pypto_core import passes

N = 8

#: Must track the constants in ``src/ir/transforms/inject_tracr_buffer_pass.cpp``.
TRACR_PAYLOAD_CAPACITY = 4096
TRACR_HEADER_WORDS = 2
TRACR_WORDS_PER_PAYLOAD = 2
TRACR_BUFFER_ELEMS = TRACR_HEADER_WORDS + TRACR_WORDS_PER_PAYLOAD * TRACR_PAYLOAD_CAPACITY

BUFFER_NAME = "__tracr_buffer"


def _apply(program):
    """Run inject_tracr_buffer with verification disabled."""
    with passes.PassContext([]):
        return passes.inject_tracr_buffer()(program)


def _func(program, name):
    for gvar, func in program.functions.items():
        if func.name == name:
            return func
    raise AssertionError(f"function {name!r} not found in program")


def _param_names(func):
    return [p.name_hint for p in func.params]


def test_program_without_comm_is_untouched():
    """The gate. No notify, no wait, no change at all.

    This is what keeps the pass free for the overwhelming majority of compiled
    models: they never mention communication, so they must not pay a signature
    change or an allocation for a profiler they are not using.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(self, inp: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], out)

    ir.assert_structural_equal(_apply(Before), Before)


def test_notify_kernel_gains_the_buffer_param():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    after = _func(_apply(Before), "f")

    assert _param_names(after)[-1] == BUFFER_NAME, _param_names(after)
    # Appended, never inserted: every pre-existing parameter keeps its index, or
    # every call site in the program would have to be renumbered.
    assert _param_names(after)[:-1] == ["win", "signal", "peer"]


def test_wait_kernel_gains_the_buffer_param():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    after = _func(_apply(Before), "f")
    assert _param_names(after)[-1] == BUFFER_NAME, _param_names(after)


def test_injected_param_is_an_out_int64_tensor_of_the_expected_size():
    """Shape and direction are the contract the device emitter and host decoder share.

    ``Out`` because the kernel writes records; ``INT64`` because a TraCR payload is
    two int64 words; the element count is the ``[count][dropped]`` header plus two
    words per payload.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    after = _func(_apply(Before), "f")
    buf = after.params[-1]

    assert buf.name_hint == BUFFER_NAME
    assert buf.type.shape == [TRACR_BUFFER_ELEMS]
    assert buf.type.dtype == pl.INT64
    assert after.param_directions[-1] == ir.ParamDirection.Out


def test_pass_is_idempotent():
    """Re-running must not append a second buffer.

    The parameter name is the idempotency key, which matters because the pass
    runs inside a pipeline that may be re-entered.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    once = _apply(Before)
    twice = _apply(once)

    assert _param_names(_func(once, "f")).count(BUFFER_NAME) == 1
    assert _param_names(_func(twice, "f")).count(BUFFER_NAME) == 1
    ir.assert_structural_equal(twice, once)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
