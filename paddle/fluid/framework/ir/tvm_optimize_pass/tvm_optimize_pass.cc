/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/tvm_optimize_pass/tvm_optimize_pass.h"
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/ir/fusion_group/elementwise_group_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

void TVMOptimizePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  std::vector<fusion_group::SubGraph> subgraphs;
  std::unordered_set<Node*> all_nodes = graph->Nodes();
  for (Node* n : all_nodes) {
    bool is_found = false;
    for (auto& subgraph : subgraphs) {
      if (subgraph.Has(n)) {
        is_found = true;
        break;
      }
    }
    if (is_found) {
      continue;
    }

    fusion_group::SubGraph subgraph;
    fusion_group::ElementwiseGroupDetector detector;
    int num_operations = detector(n);
    if (num_operations >= 2) {
      subgraph = detector.GetSubgraph();
    }

    if (!subgraph.IsEmpty()) {
      subgraphs.push_back(subgraph);
    }
  }
  std::vector<tvm::relay::Function> functions = ConvertToTVMRelay(subgraphs);
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    InsertTVMOp(graph, &subgraphs[i], functions[i]);
  }
}

std::vector<tvm::relay::Function> TVMOptimizePass::ConvertToTVMRelay(
    const std::vector<fusion_group::SubGraph>& subgraphs) const {
  std::vector<tvm::relay::Function> functions;
  for (auto& subgraph : subgraphs) {
    std::vector<Node*> input_vars_of_subgraph = subgraph.GetInputVarNodes();
    std::vector<Node*> output_vars_of_subgraph = subgraph.GetOutputVarNodes();

    std::unordered_map<std::string, tvm::relay::Expr> var_name_to_expr;
    for (auto* n : input_vars_of_subgraph) {
      std::vector<int64_t> shape = n->Var()->GetShape();
      proto::VarType::Type data_type = n->Var()->GetDataType();
      std::vector<int32_t> shape_tmp(shape.begin(), shape.end());
      std::vector<tvm::PrimExpr> tensor_shape(shape_tmp.begin(),
                                              shape_tmp.end());
      auto tensor_type =
          tvm::relay::TensorType(tensor_shape, GetTVMDataType(data_type));
      var_name_to_expr[n->Name()] =
          tvm::relay::VarNode::make(n->Name(), tensor_type);
    }

    for (auto* n : subgraph.SortedNodes()) {
      if (n && n->IsOp() && n->Op()) {
        std::vector<tvm::relay::Expr> inputs;
        for (auto* in_var : n->inputs) {
          if (in_var && in_var->IsVar() && in_var->Var()) {
            if (var_name_to_expr.find(in_var->Name()) !=
                var_name_to_expr.end()) {
              inputs.push_back(var_name_to_expr.at(in_var->Name()));
            }
          }
        }
        // Assume we only has one output currently. It's OK in elementwise ops.
        std::string out_var_name = n->outputs[0]->Name();
        std::string op_name = GetTVMOpName(n->Op()->Type());
        auto relay_op = tvm::relay::Op::Get(op_name);
        var_name_to_expr[out_var_name] =
            tvm::relay::CallNode::make(relay_op, inputs, tvm::Attrs(), {});

        // get schedule
        auto reg = tvm::runtime::Registry::Get("relay.op._Register");
        auto s_i =
            tvm::runtime::Registry::Get("topi.generic.schedule_injective");
        if (!reg) {
          PADDLE_THROW("no op _Register");
        }
        if (!s_i) {
          PADDLE_THROW("no schedule _Register");
        }
        (*reg)(op_name, "FTVMSchedule", *s_i, 10);
      }
    }

    // Assume we only has one output currently
    auto out_expr = var_name_to_expr.at(output_vars_of_subgraph[0]->Name());
    auto func = tvm::relay::FunctionNode::make(
        tvm::relay::FreeVars(out_expr), out_expr, tvm::relay::Type(), {});
    functions.push_back(func);
  }

  return functions;
}

void TVMOptimizePass::InsertTVMOp(Graph* graph,
                                  fusion_group::SubGraph* subgraph,
                                  tvm::relay::Function func) const {
  std::string generated_codes = GenerateCode(func);
  std::vector<Node*> input_vars_of_subgraph = subgraph->GetInputVarNodes();
  std::vector<Node*> output_vars_of_subgraph = subgraph->GetOutputVarNodes();
  std::unordered_set<Node*> external_nodes;

  OpDesc op_desc;
  op_desc.SetType("tvm_op");

  std::vector<std::string> input_names;
  for (auto* n : input_vars_of_subgraph) {
    input_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetInput("Xs", input_names);

  std::vector<std::string> output_names;
  for (auto* n : output_vars_of_subgraph) {
    output_names.push_back(n->Name());
    external_nodes.insert(n);
  }
  op_desc.SetOutput("Outs", output_names);
  op_desc.SetAttr("generated_code", generated_codes);

  auto tvm_op_node = graph->CreateOpNode(&op_desc);
  for (auto* in : input_vars_of_subgraph) {
    IR_NODE_LINK_TO(in, tvm_op_node);
  }
  for (auto* out : output_vars_of_subgraph) {
    IR_NODE_LINK_TO(tvm_op_node, out);
  }

  std::unordered_set<const Node*> internal_nodes;
  for (auto* n : subgraph->Nodes()) {
    if (external_nodes.find(n) == external_nodes.end()) {
      internal_nodes.insert(n);
    }
  }
  GraphSafeRemoveNodes(graph, internal_nodes);
}

std::string TVMOptimizePass::GenerateCode(tvm::relay::Function func) const {
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  tvm::Map<tvm::Integer, tvm::Target> targets;
  tvm::Target llvm_tgt = tvm::Target::Create("llvm");
  tvm::Target cuda_tgt = tvm::Target::Create("cuda");
  targets.Set(2, cuda_tgt);
  build_f(func, targets, llvm_tgt);
  tvm::runtime::Module mod = mod_f();
  return mod.operator->()->GetSource();
}

tvm::runtime::DataType TVMOptimizePass::GetTVMDataType(
    proto::VarType::Type dtype) const {
  if (dtype == proto::VarType::FP32)
    return tvm::runtime::DataType::Float(32);
  else if (dtype == proto::VarType::FP16)
    return tvm::runtime::DataType::Float(16);
  else
    PADDLE_THROW(
        "Unsupported Var Type! We only support VarType::FP32 ans "
        "VarType::FP16.");
}

std::string TVMOptimizePass::GetTVMOpName(std::string op_type) const {
  return "conv2d";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
