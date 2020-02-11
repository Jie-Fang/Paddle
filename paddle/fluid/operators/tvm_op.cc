/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/tvm_op.h"

namespace paddle {
namespace operators {

class TVMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    const size_t num_ins = ctx->Inputs("Inputs").size();
    const size_t num_outs = ctx->Outputs("Outs").size();

    PADDLE_ENFORCE_GE(
        num_ins, 1UL,
        platform::errors::InvalidArgument(
            "Expected the number of inputs >= 1. Received %d.", num_ins));
    PADDLE_ENFORCE_GE(
        num_outs, 1UL,
        platform::errors::InvalidArgument(
            "Expected the number of outputs >= 1. Recived %d.", num_outs));

    std::vector<framework::DDim> x_dims = ctx->GetInputsDim("Inputs");
    for (size_t i = 1; i < num_ins; ++i) {
      PADDLE_ENFORCE_EQ(x_dims[0], x_dims[i],
                        platform::errors::InvalidArgument(
                            "All the inputs' dims should be the same."));
    }
    std::vector<framework::DDim> out_dims;
    for (size_t j = 0; j < num_outs; ++j) {
      out_dims.push_back(x_dims[0]);
    }
    ctx->SetOutputsDim("Outputs", out_dims);

    // Only lod of Inputs[0] would be shared with Outs.
    for (size_t j = 0; j < num_outs; ++j) {
      ctx->ShareLoD("Inputs", /*->*/ "Outputs", 0, j);
    }
  }
};

class TVMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Inputs",
             "(std::vector<LoDTensor>), The input tensors of tvm operator.")
        .AsDuplicable();
    AddOutput("Outputs",
              "(std::vector<LoDTensor>), The output tensors of tvm operator.")
        .AsDuplicable();
    AddAttr<std::string>("generated_codes", "TVM generated codes.")
        .SetDefault("");
    AddComment(R"DOC(
TVM Operator.

It is used to execute TVM generated codes)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(tvm_op, ops::TVMOp, ops::TVMOpMaker);
