/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file inject_tracr_buffer_pass.cpp
 * @brief Inject the __tracr_buffer GM parameter into every kernel that issues
 *        cross-device communication, so codegen has somewhere to write TraCR
 *        trace records.
 *
 * The device tier cannot use TraCR's own recording runtime: it has thread_local
 * state, heap buffers and a filesystem flush, none of which exist under CCEC.
 * What it uses instead is a plain GM region of 16-byte records that the host
 * serializes as a `.bts` lane afterwards. This pass is what puts that region in
 * reach of a generated kernel.
 *
 * A function needs the buffer when it issues `pld.system.notify` or
 * `pld.system.wait`; the parameter then propagates upward through callers, and
 * Orchestration functions materialize one per call site instead. The walk is
 * shared with InjectGMPipeBuffer (`transform_utils::InjectGMBufferParamInPlace`).
 *
 * **A program with no communication is left byte-identical.** That is the whole
 * gate: the overwhelming majority of compiled models never mention notify or
 * wait, so they see no signature change and no allocation.
 *
 * Placement, and why it is not next to InjectGMPipeBuffer:
 *
 *   - after LowerCompositeOps, which is the last pass that *creates* notify ops
 *     in user IR (LowerHostTensorCollectives lowers to pre-written builtin
 *     kernels, not to IR this pass would need to see);
 *   - after ExpandMixedKernel, so an AIC/AIV split pair each get the parameter
 *     rather than one inheriting it;
 *   - after AutoDeriveTaskDependencies **on purpose**. A trace buffer written by
 *     several tasks would otherwise become a dependency edge between them, and
 *     the profiler would serialize the very schedule it is trying to observe;
 *   - before MaterializeDistTensorCtx, because that pass appends CommCtx
 *     parameters as a trailing suffix and relies on being last to do so.
 */

#include <memory>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/inject_gm_buffer.h"

namespace pypto {
namespace ir {

namespace {

/// Well-known parameter name for the AICore TraCR trace buffer.
constexpr const char* kTracrBufferName = "__tracr_buffer";

/// Header words the device-side emitter reserves: [count][dropped].
/// Mirrors `kTracrHeaderWords` in the runtime's `aicore/tracr_aicore_emit.h`.
constexpr int64_t kTracrHeaderWords = 2;

/// int64 words per 16-byte TraCR payload. Mirrors `kTracrWordsPerPayload`.
constexpr int64_t kTracrWordsPerPayload = 2;

/// Records one buffer holds before the emitter starts dropping and counting.
///
/// Deliberately modest. Overflow is not a correctness problem — the emitter
/// drops the record and increments the drop counter, and never wraps, because
/// tracr_process requires each `.bts` to be sorted by timestamp and a wrapped
/// ring is rotated rather than ordered. So the only cost of a small buffer is
/// losing the tail of a very long run, against a real cost of a large one:
/// nobody can read a million spans, and every record is a GM store on the
/// critical path of the communication being measured.
constexpr int64_t kTracrPayloadCapacity = 4096;

constexpr int64_t kTracrBufferElems = kTracrHeaderWords + kTracrWordsPerPayload * kTracrPayloadCapacity;

/// True for the two ops whose lowering C1 wraps in a marker pair.
///
/// `pld.system.defer_wait` is excluded: it hard-requires a Simpler runtime and
/// lowers through a different path, so a marker there is a separate question.
bool IsCommMarkerSite(const CallPtr& call) {
  if (!call) return false;
  return IsOp(call, "pld.system.notify") || IsOp(call, "pld.system.wait");
}

}  // namespace

namespace pass {

Pass InjectTracrBuffer() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    for (const auto& [gvar, func] : program->functions_) {
      functions.push_back(func);
    }

    transform_utils::GMBufferInjectionSpec spec{.param_name = kTracrBufferName,
                                                .dtype = DataType::INT64,
                                                .elems = kTracrBufferElems,
                                                .is_trigger = IsCommMarkerSite,
                                                .pass_name = "InjectTracrBuffer"};
    transform_utils::InjectGMBufferParamInPlace(functions, spec);

    return std::make_shared<Program>(functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InjectTracrBuffer", kInjectTracrBufferProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
