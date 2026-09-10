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
 * @file inject_gm_buffer.cpp
 * @brief Shared plumbing for threading a GM workspace parameter through a program.
 *
 * Extracted from InjectGMPipeBuffer when InjectTracrBuffer needed the same
 * walk. The two differ only in the fields of GMBufferInjectionSpec: which call
 * makes a function need the buffer, and the parameter's name, dtype and size.
 */

#include "pypto/ir/transforms/utils/inject_gm_buffer.h"

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace transform_utils {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

using TriggerFn = std::function<bool(const CallPtr&)>;

/// Recurse into every nested body a trigger call could hide in.
bool HasTriggerOps(const std::vector<StmtPtr>& stmts, const TriggerFn& is_trigger) {
  for (const auto& stmt : stmts) {
    if (is_trigger(transform_utils::GetCallFromStmt(stmt))) return true;
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      if (HasTriggerOps(FlattenBody(for_stmt->body_), is_trigger)) return true;
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      if (HasTriggerOps(FlattenBody(if_stmt->then_body_), is_trigger)) return true;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        if (HasTriggerOps(FlattenBody(*else_body), is_trigger)) return true;
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      if (HasTriggerOps(FlattenBody(while_stmt->body_), is_trigger)) return true;
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      if (HasTriggerOps(FlattenBody(scope->body_), is_trigger)) return true;
    }
  }
  return false;
}

bool HasBufferParam(const FunctionPtr& func, const std::string& param_name) {
  for (const auto& param : func->params_) {
    if (param->name_hint_ == param_name) return true;
  }
  return false;
}

OpPtr GetCallLikeOpFromStmt(const StmtPtr& stmt) {
  ExprPtr expr;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    expr = assign->value_;
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    expr = eval->expr_;
  }
  if (auto call = As<Call>(expr)) return call->op_;
  if (auto submit = As<Submit>(expr)) return submit->op_;
  return nullptr;
}

StmtPtr CloneScopeWithBody(const ScopeStmtPtr& scope, const StmtPtr& body, const std::string& pass_name) {
  if (auto incore = As<InCoreScopeStmt>(scope)) {
    return std::make_shared<InCoreScopeStmt>(incore->split_, incore->name_hint_, body, incore->span_,
                                             incore->leading_comments_, incore->attrs_);
  }
  if (auto cluster = As<ClusterScopeStmt>(scope)) {
    return std::make_shared<ClusterScopeStmt>(cluster->name_hint_, body, cluster->span_,
                                              cluster->leading_comments_, cluster->attrs_);
  }
  if (auto hierarchy = As<HierarchyScopeStmt>(scope)) {
    return std::make_shared<HierarchyScopeStmt>(hierarchy->level_, hierarchy->role_, hierarchy->name_hint_,
                                                body, hierarchy->span_, hierarchy->leading_comments_,
                                                hierarchy->attrs_);
  }
  if (auto spmd = As<SpmdScopeStmt>(scope)) {
    return std::make_shared<SpmdScopeStmt>(spmd->core_num_, spmd->sync_start_, spmd->name_hint_, body,
                                           spmd->span_, spmd->leading_comments_, spmd->attrs_);
  }
  if (auto runtime = As<RuntimeScopeStmt>(scope)) {
    return std::make_shared<RuntimeScopeStmt>(runtime->manual_, runtime->name_hint_, body, runtime->span_,
                                              runtime->leading_comments_, runtime->attrs_);
  }
  INTERNAL_CHECK_SPAN(false, scope->span_) << "Unsupported ScopeStmt kind in " << pass_name;
  return scope;
}

void BuildCallGraphFromFunctions(const std::vector<FunctionPtr>& functions,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callers,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callees) {
  std::unordered_set<std::string> func_names;
  for (const auto& func : functions) func_names.insert(func->name_);
  for (const auto& func : functions) {
    std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& stmts) {
      for (const auto& stmt : stmts) {
        if (auto op = GetCallLikeOpFromStmt(stmt)) {
          auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
          if (gv && func_names.count(gv->name_)) {
            callees[func->name_].insert(gv->name_);
            callers[gv->name_].insert(func->name_);
          }
        }
        if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
          walk(FlattenBody(for_stmt->body_));
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
          walk(FlattenBody(if_stmt->then_body_));
          const auto& else_body = if_stmt->else_body_;
          if (else_body.has_value()) walk(FlattenBody(*else_body));
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
          walk(FlattenBody(while_stmt->body_));
        } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
          walk(FlattenBody(scope->body_));
        }
      }
    };
    if (func->body_) walk(FlattenBody(func->body_));
  }
}

TensorTypePtr MakeBufferType(const GMBufferInjectionSpec& spec) {
  return std::make_shared<TensorType>(std::vector<int64_t>{spec.elems}, spec.dtype, std::nullopt,
                                      std::nullopt);
}

FunctionPtr AddBufferParam(const FunctionPtr& func, const GMBufferInjectionSpec& spec) {
  auto gm_var = std::make_shared<Var>(spec.param_name, MakeBufferType(spec), func->span_);
  auto new_params = func->params_;
  new_params.push_back(gm_var);
  auto new_directions = func->param_directions_;
  new_directions.push_back(ParamDirection::Out);
  auto result = MutableCopy(func);
  result->params_ = new_params;
  result->param_directions_ = new_directions;
  return result;
}

/// Append ``gm_param`` to every call that targets a function in ``modified_funcs``.
StmtPtr RewriteCallsWithParam(const StmtPtr& body, const std::unordered_set<std::string>& modified_funcs,
                              const VarPtr& gm_param, const std::string& pass_name) {
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;
  auto should_rewrite = [&](const OpPtr& op) -> bool {
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
    return gv && modified_funcs.count(gv->name_);
  };
  auto try_rewrite_expr = [&](const ExprPtr& expr) -> ExprPtr {
    if (!expr) return nullptr;
    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) {
      if (!should_rewrite(call->op_)) return nullptr;
      auto new_call = MutableCopy(call);
      new_call->args_.push_back(gm_param);
      return new_call;
    }
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) {
      if (!should_rewrite(submit->op_)) return nullptr;
      auto new_submit = MutableCopy(submit);
      new_submit->args_.push_back(gm_param);
      return new_submit;
    }
    return nullptr;
  };
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto rw = try_rewrite_expr(assign->value_)) {
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto rw = try_rewrite_expr(eval->expr_)) {
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsWithParam(for_stmt->body_, modified_funcs, gm_param, pass_name);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(new_for);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsWithParam(if_stmt->then_body_, modified_funcs, gm_param, pass_name);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsWithParam(*else_body, modified_funcs, gm_param, pass_name);
      }
      bool body_changed = (nt != if_stmt->then_body_);
      if (!body_changed && ne.has_value() && else_body.has_value()) {
        body_changed = (*ne != *else_body);
      }
      if (body_changed) {
        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = nt;
        new_if->else_body_ = ne;
        new_stmts.push_back(new_if);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsWithParam(while_stmt->body_, modified_funcs, gm_param, pass_name);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(new_while);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteCallsWithParam(scope->body_, modified_funcs, gm_param, pass_name);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb, pass_name));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any_changed) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

CallPtr CreateBufferTensorCreate(const GMBufferInjectionSpec& spec, const Span& span) {
  auto shape_elem = std::make_shared<ConstInt>(spec.elems, DataType::INDEX, span);
  auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{shape_elem}, span);
  return OpRegistry::GetInstance().Create("tensor.create", {shape_tuple},
                                          {{"dtype", std::any(spec.dtype)},
                                           {"layout", std::any(TensorLayout::ND)},
                                           {"manual_dep", std::any(true)}},
                                          span);
}

/// Orchestration form: materialize one buffer per call site and pass it in.
StmtPtr RewriteCallsWithPerCallBuffer(const StmtPtr& body,
                                      const std::unordered_set<std::string>& modified_funcs,
                                      const GMBufferInjectionSpec& spec, const Span& span, int& counter) {
  auto gm_type = MakeBufferType(spec);
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;

  auto should_rewrite = [&](const OpPtr& op) -> bool {
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
    return gv && modified_funcs.count(gv->name_);
  };
  auto make_gm_create = [&]() -> std::pair<StmtPtr, VarPtr> {
    // Strip the leading underscores so the emitted local reads as an identifier.
    std::string base = spec.param_name;
    while (!base.empty() && base.front() == '_') base.erase(base.begin());
    std::string var_name = base + "_" + std::to_string(counter++);
    auto gm_var = std::make_shared<Var>(var_name, gm_type, span);
    auto create_call = CreateBufferTensorCreate(spec, span);
    auto create_stmt = std::make_shared<AssignStmt>(gm_var, create_call, span);
    return std::make_pair(StmtPtr(create_stmt), gm_var);
  };
  auto try_rewrite = [&](const ExprPtr& expr) -> std::pair<StmtPtr, ExprPtr> {
    if (!expr) return std::make_pair(StmtPtr{}, ExprPtr{});
    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) {
      if (!should_rewrite(call->op_)) return std::make_pair(StmtPtr{}, ExprPtr{});
      auto [create_stmt, gm_var] = make_gm_create();
      // Copy rather than reconstruct, so the original's kwargs_ and attrs_ survive.
      auto new_call = MutableCopy(call);
      new_call->args_.push_back(gm_var);
      return std::make_pair(create_stmt, ExprPtr(new_call));
    }
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) {
      if (!should_rewrite(submit->op_)) return std::make_pair(StmtPtr{}, ExprPtr{});
      auto [create_stmt, gm_var] = make_gm_create();
      auto new_submit = MutableCopy(submit);
      new_submit->args_.push_back(gm_var);
      return std::make_pair(create_stmt, ExprPtr(new_submit));
    }
    return std::make_pair(StmtPtr{}, ExprPtr{});
  };

  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto [create, rw] = try_rewrite(assign->value_);
      if (rw) {
        new_stmts.push_back(create);
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto [create, rw] = try_rewrite(eval->expr_);
      if (rw) {
        new_stmts.push_back(create);
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallBuffer(for_stmt->body_, modified_funcs, spec, span, counter);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(std::move(new_for));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsWithPerCallBuffer(if_stmt->then_body_, modified_funcs, spec, span, counter);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsWithPerCallBuffer(*else_body, modified_funcs, spec, span, counter);
      }
      bool body_changed = (nt != if_stmt->then_body_);
      if (!body_changed && ne.has_value() && else_body.has_value()) {
        body_changed = (*ne != *else_body);
      }
      if (body_changed) {
        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = nt;
        new_if->else_body_ = ne;
        new_stmts.push_back(std::move(new_if));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallBuffer(while_stmt->body_, modified_funcs, spec, span, counter);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(std::move(new_while));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallBuffer(scope->body_, modified_funcs, spec, span, counter);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb, spec.pass_name));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any_changed) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

}  // namespace

void InjectGMBufferParamInPlace(std::vector<FunctionPtr>& functions, const GMBufferInjectionSpec& spec) {
  std::unordered_map<std::string, FunctionPtr*> func_by_name;
  for (auto& func : functions) func_by_name[func->name_] = &func;

  std::unordered_map<std::string, std::unordered_set<std::string>> callers, callees;
  BuildCallGraphFromFunctions(functions, callers, callees);

  std::unordered_set<std::string> trigger_funcs;
  for (const auto& func : functions) {
    if (!HasBufferParam(func, spec.param_name) && func->body_ &&
        HasTriggerOps(FlattenBody(func->body_), spec.is_trigger)) {
      trigger_funcs.insert(func->name_);
    }
  }
  if (trigger_funcs.empty()) return;

  // Propagate upward, stopping at Orchestration boundaries (they materialize
  // the buffer locally instead of taking it as a parameter).
  std::unordered_set<std::string> needs_param = trigger_funcs;
  std::vector<std::string> worklist(trigger_funcs.begin(), trigger_funcs.end());
  while (!worklist.empty()) {
    std::string name = worklist.back();
    worklist.pop_back();
    auto it = callers.find(name);
    if (it == callers.end()) continue;
    for (const auto& caller_name : it->second) {
      auto fit = func_by_name.find(caller_name);
      if (fit == func_by_name.end()) continue;
      if (IsOrchestrationLike((*fit->second)->func_type_)) continue;
      if (needs_param.insert(caller_name).second) worklist.push_back(caller_name);
    }
  }

  for (auto& func : functions) {
    if (needs_param.count(func->name_) && !HasBufferParam(func, spec.param_name)) {
      func = AddBufferParam(func, spec);
    }
  }

  for (auto& func : functions) {
    if (!needs_param.count(func->name_)) continue;

    VarPtr gm_param;
    for (const auto& p : func->params_) {
      if (p->name_hint_ == spec.param_name) {
        gm_param = p;
        break;
      }
    }
    INTERNAL_CHECK_SPAN(gm_param, func->span_)
        << "Internal error: " << func->name_ << " should have " << spec.param_name;

    std::unordered_set<std::string> mod_callees;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_param.count(c)) mod_callees.insert(c);
      }
    }

    if (!mod_callees.empty()) {
      auto nb = RewriteCallsWithParam(func->body_, mod_callees, gm_param, spec.pass_name);
      auto updated = MutableCopy(func);
      updated->body_ = nb;
      func = updated;
    }
  }

  for (auto& func : functions) {
    // Must stay in lockstep with the boundary marking above: widening only one
    // side leaves a caller passing the old arity to a rewritten callee.
    if (!IsOrchestrationLike(func->func_type_)) continue;
    if (!func->body_) continue;
    if (needs_param.empty()) continue;

    int counter = 0;
    auto new_body = RewriteCallsWithPerCallBuffer(func->body_, needs_param, spec, func->span_, counter);
    auto updated = MutableCopy(func);
    updated->body_ = new_body;
    func = updated;
  }
}

}  // namespace transform_utils
}  // namespace ir
}  // namespace pypto
