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

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/dynload/nvrtc.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TVMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("Inputs");
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Outputs");
    std::string ptx = ctx.Attr<std::string>("generated_code");

    size_t num_ins = ins.size();
    size_t num_outs = outs.size();

    auto place = ctx.GetPlace();
    for (size_t i = 0; i < num_outs; ++i) {
      outs[i]->mutable_data<T>(place);
    }

    size_t n = ins[0]->numel();
    std::vector<void*> args;
    args.push_back(&n);
    std::vector<const T*> ptrs(num_ins + num_outs);
    for (size_t i = 0; i < num_ins; ++i) {
      ptrs[i] = ins[i]->data<T>();
      args.push_back(&ptrs[i]);
    }
    for (size_t j = 0; j < num_outs; ++j) {
      ptrs[num_ins + j] = outs[j]->data<T>();
      args.push_back(&ptrs[num_ins + j]);
    }

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    platform::dynload::cuModuleLoadData(&module, ptx.c_str());
    platform::dynload::cuModuleGetFunction(&kernel, module, "tvm_op");

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / 1024, 1);
    int num_blocks = std::min(
        max_blocks, (static_cast<int>(n) + max_threads - 1) / max_threads);
    PADDLE_ENFORCE_EQ(
        platform::dynload::cuLaunchKernel(kernel, num_blocks, 1, 1,  // grid dim
                                          1024, 1, 1,        // block dim
                                          0,                 // shared memory
                                          dev_ctx.stream(),  // stream
                                          args.data(),       // arguments
                                          nullptr),
        CUDA_SUCCESS,
        platform::errors::External(
            "Fail to launch tvm generated kernel in cuLaunchKernel."));
  }
};

}  // namespace operators
}  // namespace paddle
