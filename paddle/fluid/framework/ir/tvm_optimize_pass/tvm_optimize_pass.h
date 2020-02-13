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

#pragma once

#include <tvm/relay/expr.h>
#include <tvm/runtime/data_type.h>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/subgraph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class TVMOptimizePass : public Pass {
 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  tvm::relay::Function ConvertToTVMRelay(
      fusion_group::SubGraph* subgraph) const;
  std::string GetTVMOpName(std::string) const;
  tvm::runtime::DataType GetTVMDataType(proto::VarType::Type dtype) const;
  std::string GenerateCode(tvm::relay::Function func) const;
  void InsertTVMOp(Graph* graph, fusion_group::SubGraph* subgraph,
                   tvm::relay::Function func) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
