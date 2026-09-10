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

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"

namespace pypto {
namespace ir {
namespace transform_utils {

/// Describes one GM workspace parameter to thread through a program.
///
/// Two passes need the same plumbing and differ only in these fields, so the
/// walk lives in ``InjectGMBufferParamInPlace`` and each pass supplies a spec:
///
///   InjectGMPipeBuffer    cross-core tpush/tpop slot data on GM backends
///   InjectTracrBuffer     AICore TraCR trace records for comm markers
struct GMBufferInjectionSpec {
  /// Parameter name. Also the idempotency key: a function already carrying a
  /// parameter with this name is left alone, so the pass is safe to re-run.
  std::string param_name;

  /// Element type of the injected tensor parameter.
  DataType dtype;

  /// Element count of the injected tensor parameter.
  int64_t elems;

  /// True for a call that makes its enclosing function need the buffer.
  std::function<bool(const CallPtr&)> is_trigger;

  /// Pass name, used only in internal-error messages.
  std::string pass_name;
};

/// Add ``spec.param_name`` to every function that needs it, and thread it
/// through the call graph.
///
/// A function needs the buffer when its body issues a trigger call, or when it
/// calls a function that needs it. Propagation stops at Orchestration-like
/// functions: those materialize a per-call-site ``tensor.create`` placeholder
/// instead of taking a parameter, because they own the allocation.
///
/// Rewrites ``functions`` in place. A program with no trigger call anywhere is
/// left completely untouched, which is what keeps the pass free for the
/// programs it does not concern.
void InjectGMBufferParamInPlace(std::vector<FunctionPtr>& functions, const GMBufferInjectionSpec& spec);

}  // namespace transform_utils
}  // namespace ir
}  // namespace pypto
