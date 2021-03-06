// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;
using paddle::ConvertToACPrecision;

extern "C" {
// PaddleTensor
PD_Tensor* PD_NewPaddleTensor() { return new PD_Tensor; }

void PD_DeletePaddleTensor(PD_Tensor* tensor) {
  if (tensor) {
    delete tensor;
    tensor = nullptr;
  }
}

void PD_SetPaddleTensorName(PD_Tensor* tensor, char* name) {
  tensor->tensor.name = std::string(name);
}

void PD_SetPaddleTensorDType(PD_Tensor* tensor, PD_DataType dtype) {
  tensor->tensor.dtype = paddle::ConvertToPaddleDType(dtype);
}

void PD_SetPaddleTensorData(PD_Tensor* tensor, PD_PaddleBuf* buf) {
  tensor->tensor.data = buf->buf;
}

void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size) {
  tensor->tensor.shape.assign(shape, shape + size);
}

const char* PD_GetPaddleTensorName(const PD_Tensor* tensor) {
  return tensor->tensor.name.c_str();
}

PD_DataType PD_GetPaddleTensorDType(const PD_Tensor* tensor) {
  return ConvertToPDDataType(tensor->tensor.dtype);
}

PD_PaddleBuf* PD_GetPaddleTensorData(const PD_Tensor* tensor) {
  PD_PaddleBuf* ret = PD_NewPaddleBuf();
  ret->buf = tensor->tensor.data;
  return ret;
}

int* PD_GetPaddleTensorShape(const PD_Tensor* tensor, int** size) {
  std::vector<int> shape = tensor->tensor.shape;
  int s = shape.size();
  *size = &s;
  return shape.data();
}

}  // extern "C"
