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
 * @file inject_gm_pipe_buffer_pass.cpp
 * @brief Inject the __gm_pipe_buffer workspace parameter for cross-core pipes on backends
 *        that route slot data through GM (currently Ascend910B).
 *
 * On 910B, cross-core tpush/tpop rides through a shared GM buffer instead of a
 * direct inter-core fabric. This pass runs after ExpandMixedKernel has split
 * mixed InCore functions into AIC/AIV pairs, and:
 *
 *   1. Finds every function that issues initialize_pipe ops.
 *   2. Adds a fresh __gm_pipe_buffer Out-tensor parameter to each, propagating
 *      the parameter upward through callers except Orchestration functions,
 *      which instead get a per-call-site placeholder tensor.create. Codegen
 *      owns the real hardware workspace size and per-pipe offsets.
 *
 * The walk itself is shared with InjectTracrBuffer and lives in
 * ``transform_utils::InjectGMBufferParamInPlace``; this file supplies only the
 * spec that distinguishes the two.
 *
 * The pass is gated on BackendHandler::RequiresGMPipeBuffer(); other backends
 * see it as a no-op. Nothing else in the pipeline depends on its output, so it
 * produces the same IR properties it requires (MixedKernelExpanded).
 */

#include <memory>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/inject_gm_buffer.h"
#include "pypto/ir/transforms/utils/op_predicates.h"

namespace pypto {
namespace ir {

namespace {

/// Well-known parameter name for the GM slot buffer.
constexpr const char* kGMPipeBufferName = "__gm_pipe_buffer";

/// Codegen owns the real workspace size and per-pipe offsets, so the IR only
/// needs a parameter to exist; one element is enough to carry the type.
constexpr int64_t kGMPipeBufferPlaceholderElems = 1;

}  // namespace

namespace pass {

Pass InjectGMPipeBuffer() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!backend::BackendConfig::IsConfigured() ||
        !PassContext::Current()->GetBackendHandler()->RequiresGMPipeBuffer()) {
      return program;
    }
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    for (const auto& [gvar, func] : program->functions_) {
      functions.push_back(func);
    }

    transform_utils::GMBufferInjectionSpec spec{
        .param_name = kGMPipeBufferName,
        .dtype = DataType::FP32,
        .elems = kGMPipeBufferPlaceholderElems,
        .is_trigger = [](const CallPtr& call) { return op_predicates::IsInitializePipe(call); },
        .pass_name = "InjectGMPipeBuffer"};
    transform_utils::InjectGMBufferParamInPlace(functions, spec);

    return std::make_shared<Program>(functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InjectGMPipeBuffer", kInjectGMPipeBufferProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
